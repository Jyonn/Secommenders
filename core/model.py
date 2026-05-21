import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models import build_backbone
from utils import function

from .encoders import LLMSequenceEncoder, ScratchSequenceEncoder


class SequentialRecModel(nn.Module):
    def __init__(self, compiled, config):
        super().__init__()
        self.compiled = compiled
        self.config = config
        self.backbone_def = build_backbone(
            config.model,
            [''],
            max_length_override=config.model_max_length,
        )
        freeze_default = compiled.model_kind == 'llm'
        self.freeze_backbone = function.coerce_bool(config.freeze_backbone, default=freeze_default)
        self.use_lora = function.coerce_bool(config.use_lora, default=compiled.model_kind == 'llm')

        if compiled.model_kind == 'llm':
            self.encoder = LLMSequenceEncoder(
                compiled.model_key,
                freeze_backbone=self.freeze_backbone,
                use_lora=self.use_lora,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                lora_target_modules=config.lora_target_modules,
                model_dtype=config.model_dtype,
            )
        else:
            self.encoder = ScratchSequenceEncoder(
                vocab_size=compiled.model_vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_length=compiled.meta['model_max_length'],
            )

        hidden_size = self.encoder.hidden_size
        self.type_marker_embedding = nn.Embedding(len(self.compiled.special_vocab['tokens']), hidden_size)
        self.uid_embedding = nn.Embedding(compiled.num_items, hidden_size)
        self.sid_embedding = nn.Embedding(max(compiled.sid_vocab_size, 1), hidden_size)
        self.embedding_projection = None
        self.embedding_head = None

        if compiled.embedding_matrix is not None:
            self.register_buffer('embedding_matrix', compiled.embedding_matrix)
            self.embedding_projection = nn.Linear(compiled.embedding_matrix.shape[1], hidden_size, bias=False)
        else:
            self.register_buffer('embedding_matrix', torch.empty(0))

        if config.task_type == 'uid':
            self.target_head = nn.Linear(hidden_size, compiled.num_items)
        elif config.task_type == 'sid':
            if not compiled.sid_vocab_size or not compiled.sid_num_quantizers:
                raise ValueError('sid task requires sid vocab metadata in compiled artifacts')
            self.target_head = nn.Linear(hidden_size, compiled.sid_vocab_size)
        elif config.task_type == 'embedding':
            if compiled.embedding_matrix is None:
                raise ValueError('embedding task requires compiled embedding view and source matrix')
            self.embedding_head = nn.Linear(hidden_size, compiled.embedding_matrix.shape[1], bias=False)
            self.target_head = None
        else:
            raise ValueError(f'Unsupported task type: {config.task_type}')

        self.compute_dtype = getattr(self.encoder, 'compute_dtype', torch.float32)
        self.type_marker_embedding.to(dtype=self.compute_dtype)
        self.uid_embedding.to(dtype=self.compute_dtype)
        self.sid_embedding.to(dtype=self.compute_dtype)
        if self.embedding_projection is not None:
            self.embedding_projection.to(dtype=self.compute_dtype)
        if self.embedding_head is not None:
            self.embedding_head.to(dtype=self.compute_dtype)
        if self.target_head is not None:
            self.target_head.to(dtype=self.compute_dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    def trainable_state_dict(self):
        state = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: tensor.detach().cpu() for name, tensor in state.items() if name in trainable_names}

    def _type_marker_index(self, marker_name: str):
        return int(self.compiled.special_vocab['marker_to_index'][marker_name])

    def _render_history_item(self, uid: int):
        if self.config.repr_combine == 'add':
            if 'embedding' not in self.compiled.item_views:
                raise ValueError('repr.combine=add requires compiled embedding view')
            emb_index = int(self.compiled.item_views['embedding'][uid])
            return [
                ('type_marker', 'uid+embedding'),
                ('uid_embedding_add', (uid, emb_index)),
            ]

        specs = []
        for repr_type in self.config.compile_config.repr_types:
            if repr_type == 'uid':
                specs.append(('type_marker', 'uid'))
                specs.append(('uid', uid))
            elif repr_type == 'text':
                specs.append(('type_marker', 'text'))
                token_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views['text'][uid])]
                specs.append(('model_tokens', token_ids))
            elif repr_type == 'sid':
                specs.append(('type_marker', 'sid'))
                sid_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views['sid'][uid])]
                specs.append(('sid', sid_ids))
            elif repr_type == 'embedding':
                specs.append(('type_marker', 'embedding'))
                emb_index = int(self.compiled.item_views['embedding'][uid])
                specs.append(('embedding', emb_index))
            else:
                raise ValueError(f'Unsupported repr type: {repr_type}')
        return specs

    def _render_single_view_item(self, uid: int, view_name: str):
        if view_name == 'uid':
            return [('type_marker', 'uid'), ('uid', uid)]
        if view_name == 'text':
            token_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views['text'][uid])]
            return [('type_marker', 'text'), ('model_tokens', token_ids)]
        if view_name == 'sid':
            sid_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views['sid'][uid])]
            return [('type_marker', 'sid'), ('sid', sid_ids)]
        if view_name == 'embedding':
            emb_index = int(self.compiled.item_views['embedding'][uid])
            return [('type_marker', 'embedding'), ('embedding', emb_index)]
        raise ValueError(f'Unsupported alignment source view: {view_name}')

    def _build_sample_specs(self, sample):
        specs = [('model_tokens', [int(token_id) for token_id in self.compiled.prompt_main['history_prefix_ids']])]
        history_uids = sample['history_uids']
        for index, uid in enumerate(history_uids):
            specs.extend(self._render_history_item(uid))
            if index != len(history_uids) - 1:
                specs.append(('model_tokens', [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]))
        specs.append(('model_tokens', [int(token_id) for token_id in self.compiled.prompt_main['query_prefix_ids']]))
        specs.append(('type_marker', self.config.task_type))
        return specs

    def _embed_spec(self, kind: str, value):
        if kind == 'model_tokens':
            token_ids = torch.tensor(value, dtype=torch.long, device=self.device)
            return self.encoder.embed_model_tokens(token_ids)
        if kind == 'type_marker':
            marker_index = torch.tensor([self._type_marker_index(value)], dtype=torch.long, device=self.device)
            return self.type_marker_embedding(marker_index)
        if kind == 'uid':
            token_ids = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            return self.uid_embedding(token_ids)
        if kind == 'sid':
            token_ids = torch.tensor(value, dtype=torch.long, device=self.device)
            return self.sid_embedding(token_ids)
        if kind == 'embedding':
            emb_index = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            projected = self.embedding_projection(self.embedding_matrix[emb_index].to(dtype=self.compute_dtype))
            return projected
        if kind == 'uid_embedding_add':
            uid, emb_index = value
            uid_tensor = torch.tensor([int(uid)], dtype=torch.long, device=self.device)
            emb_tensor = torch.tensor([int(emb_index)], dtype=torch.long, device=self.device)
            uid_embed = self.uid_embedding(uid_tensor)
            content_embed = self.embedding_projection(self.embedding_matrix[emb_tensor].to(dtype=self.compute_dtype))
            return uid_embed + content_embed
        raise ValueError(f'Unknown spec kind: {kind}')

    def _build_batch_inputs(self, batch):
        sample_embeddings = []
        for sample in batch:
            specs = self._build_sample_specs(sample)
            pieces = [self._embed_spec(kind, value) for kind, value in specs]
            sample_embeddings.append(torch.cat(pieces, dim=0))

        padded = pad_sequence(sample_embeddings, batch_first=True)
        padded = padded.to(dtype=self.compute_dtype)
        lengths = torch.tensor([emb.shape[0] for emb in sample_embeddings], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(padded.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, attention_mask.long(), lengths

    def ranking_ks(self):
        max_k = max(1, int(self.config.sid_beam_width))
        ks = [k for k in (5, 10, 20) if k <= max_k]
        if max_k not in ks:
            ks.append(max_k)
        return sorted(set(ks))

    def sid_ranking_ks(self):
        return self.ranking_ks()

    def _build_sid_generation_batch_inputs(self, sample, sid_prefixes: list[list[int]]):
        sample_embeddings = []
        for sid_prefix in sid_prefixes:
            specs = self._build_sample_specs(sample)
            if sid_prefix:
                specs.append(('sid', [int(code) for code in sid_prefix]))
            pieces = [self._embed_spec(kind, value) for kind, value in specs]
            sample_embeddings.append(torch.cat(pieces, dim=0))

        padded = pad_sequence(sample_embeddings, batch_first=True)
        padded = padded.to(dtype=self.compute_dtype)
        lengths = torch.tensor([emb.shape[0] for emb in sample_embeddings], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(padded.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, attention_mask.long(), lengths

    def _predict_sid_step_logits(self, sample, sid_prefixes: list[list[int]], slot_index: int):
        inputs_embeds, attention_mask, lengths = self._build_sid_generation_batch_inputs(sample, sid_prefixes)
        hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths - 1]
        return self.target_head(pooled)

    def _pick_sid_item(self, sid_sequence: tuple[int, ...]):
        candidates = self.compiled.sid_sequence_to_items.get(sid_sequence, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])
        digest = hashlib.sha256(
            f'{self.config.seed}:{"-".join(str(code) for code in sid_sequence)}'.encode('utf-8')
        ).hexdigest()
        return int(candidates[int(digest[:8], 16) % len(candidates)])

    def _beam_search_sid_items(self, sample):
        beam_width = max(1, int(self.config.sid_beam_width))
        beams: list[tuple[tuple[int, ...], float]] = [(tuple(), 0.0)]

        for slot_index in range(int(self.compiled.sid_num_quantizers)):
            sid_prefixes = [list(prefix) for prefix, _ in beams]
            step_logits = self._predict_sid_step_logits(sample, sid_prefixes, slot_index)
            step_log_probs = F.log_softmax(step_logits.float(), dim=-1)
            candidates = []

            for beam_index, (prefix, score) in enumerate(beams):
                allowed_codes = self.compiled.sid_prefix_to_next.get(prefix, [])
                if not allowed_codes:
                    continue
                allowed_indices = torch.tensor(allowed_codes, dtype=torch.long, device=self.device)
                allowed_scores = step_log_probs[beam_index, allowed_indices]
                top_k = min(beam_width, allowed_scores.shape[0])
                top_scores, top_positions = torch.topk(allowed_scores, k=top_k)
                for top_position, top_score in zip(top_positions.tolist(), top_scores.tolist()):
                    next_code = int(allowed_codes[top_position])
                    candidates.append((prefix + (next_code,), score + float(top_score)))

            if not candidates:
                break
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[:beam_width]

        return beams

    def _decode_sid_beams_to_items(self, beams):
        ranked_items = []
        seen_items = set()
        for sid_sequence, score in beams:
            item_uid = self._pick_sid_item(sid_sequence)
            if item_uid is None or item_uid in seen_items:
                continue
            seen_items.add(item_uid)
            ranked_items.append((item_uid, float(score), sid_sequence))
        return ranked_items

    def _compute_sid_ranking_metrics(self, batch):
        ks = self.ranking_ks()
        totals = {f'hr@{k}': 0.0 for k in ks}
        totals.update({f'ndcg@{k}': 0.0 for k in ks})
        totals['mrr'] = 0.0
        totals['beam_unique_items'] = 0.0

        with torch.no_grad():
            for sample in batch:
                ranked_items = self._decode_sid_beams_to_items(self._beam_search_sid_items(sample))
                ranked_uids = [uid for uid, _, _ in ranked_items]
                totals['beam_unique_items'] += float(len(ranked_uids))
                target_uid = int(sample['target_uid'])
                rank = ranked_uids.index(target_uid) + 1 if target_uid in ranked_uids else None
                if rank is not None:
                    totals['mrr'] += 1.0 / rank
                for k in ks:
                    if rank is not None and rank <= k:
                        totals[f'hr@{k}'] += 1.0
                        totals[f'ndcg@{k}'] += 1.0 / math.log2(rank + 1)

        batch_size = max(len(batch), 1)
        return {key: value / batch_size for key, value in totals.items()}

    def _compute_sid_loss(self, batch):
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        token_correct = 0.0
        seq_correct = 0.0
        token_total = 0

        for sample in batch:
            target_codes = [
                int(token_id) for token_id in function.to_list(self.compiled.item_views['sid'][sample['target_uid']])
            ]
            sample_preds = []
            sample_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for slot_index, label in enumerate(target_codes):
                prefix = target_codes[:slot_index]
                logits = self._predict_sid_step_logits(sample, [prefix], slot_index)
                label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
                sample_loss = sample_loss + F.cross_entropy(logits.float(), label_tensor)
                pred = int(logits.argmax(dim=-1).item())
                sample_preds.append(pred)
                token_correct += float(pred == label)
                token_total += 1

            total_loss = total_loss + sample_loss / max(len(target_codes), 1)
            seq_correct += float(sample_preds == target_codes)

        batch_size = max(len(batch), 1)
        return total_loss / batch_size, {
            'sid_token_acc': token_correct / max(token_total, 1),
            'sid_seq_acc': seq_correct / batch_size,
        }

    def _compute_ranking_metrics_from_logits(self, logits: torch.Tensor, target_indices: torch.Tensor):
        ks = self.ranking_ks()
        totals = {f'hr@{k}': 0.0 for k in ks}
        totals.update({f'ndcg@{k}': 0.0 for k in ks})
        totals['mrr'] = 0.0

        ranking = torch.argsort(logits.float(), dim=-1, descending=True)
        targets = target_indices.view(-1, 1)
        match_positions = (ranking == targets).nonzero(as_tuple=False)
        if match_positions.shape[0] != logits.shape[0]:
            raise RuntimeError('failed to recover target rank from logits')
        ranks = match_positions[:, 1] + 1

        for k in ks:
            hits = (ranks <= k).float()
            totals[f'hr@{k}'] = float(hits.mean().item())
            ndcg = hits / torch.log2(ranks.float() + 1)
            totals[f'ndcg@{k}'] = float(ndcg.mean().item())
        totals['mrr'] = float((1.0 / ranks.float()).mean().item())
        return totals

    def _target_token_values(self, target_uid: int):
        if self.config.task_type == 'embedding':
            raise ValueError('embedding task uses query-anchor supervision instead of target tokens')
        if self.config.task_type == 'uid':
            return [int(self.compiled.item_views['uid'][target_uid])]
        sid_values = [int(token_id) for token_id in function.to_list(self.compiled.item_views['sid'][target_uid])]
        expected = int(self.compiled.sid_num_quantizers)
        if expected and len(sid_values) != expected:
            raise ValueError(
                f'sid target length mismatch for uid={target_uid}: '
                f'expected {expected} tokens, got {len(sid_values)}'
            )
        return sid_values

    def _target_embedding_index(self, target_uid: int):
        return int(self.compiled.item_views['embedding'][target_uid])

    def _build_finetune_sample_inputs(self, sample):
        prefix_ids = [int(token_id) for token_id in self.compiled.prompt_main['history_prefix_ids']]
        separator_ids = [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]
        query_ids = [int(token_id) for token_id in self.compiled.prompt_main['query_prefix_ids']]
        sequence_uids = sample['sequence_uids']
        history_uids = sequence_uids[:-1]
        target_uids = sequence_uids[1:]

        embeddings = []
        position_ids = []
        history_block_ends = []

        def append_embedded(tensor: torch.Tensor, positions: list[int]):
            start = sum(piece.shape[0] for piece in embeddings)
            embeddings.append(tensor)
            position_ids.extend(positions)
            return start, start + tensor.shape[0] - 1

        append_embedded(
            self._embed_spec('model_tokens', prefix_ids),
            list(range(len(prefix_ids))),
        )
        logical_cursor = len(prefix_ids)
        history_region_end = len(prefix_ids) - 1

        for index, uid in enumerate(history_uids):
            block_specs = []
            if index > 0 and separator_ids:
                block_specs.append(('model_tokens', separator_ids))
            block_specs.extend(self._render_history_item(uid))

            for kind, value in block_specs:
                piece = self._embed_spec(kind, value)
                piece_len = piece.shape[0]
                positions = list(range(logical_cursor, logical_cursor + piece_len))
                _, block_end = append_embedded(piece, positions)
                logical_cursor += piece_len
                history_region_end = block_end
            history_block_ends.append(history_region_end)

        prediction_blocks = []
        target_positions = []
        target_labels = []
        target_slots = []

        for target_index, target_uid in enumerate(target_uids):
            query_base = history_block_ends[target_index] + 1
            block_start = sum(piece.shape[0] for piece in embeddings)
            query_piece = self._embed_spec('model_tokens', query_ids)
            query_start, query_end = append_embedded(query_piece, list(range(query_base, query_base + len(query_ids))))
            marker_piece = self._embed_spec('type_marker', self.config.task_type)
            marker_start, marker_end = append_embedded(marker_piece, [query_base + len(query_ids)])

            if self.config.task_type == 'embedding':
                target_start, target_end = marker_start, marker_end
                target_positions.append(marker_end)
                target_labels.append(self._target_embedding_index(target_uid))
            else:
                token_values = self._target_token_values(target_uid)
                target_kind = 'uid' if self.config.task_type == 'uid' else 'sid'
                target_value = token_values[0] if target_kind == 'uid' else token_values
                target_start, target_end = append_embedded(
                    self._embed_spec(target_kind, target_value),
                    list(range(query_base + len(query_ids) + 1, query_base + len(query_ids) + 1 + len(token_values))),
                )
                if self.config.task_type == 'uid':
                    target_positions.append(marker_end)
                    target_labels.append(int(token_values[0]))
                else:
                    target_positions.append(marker_end)
                    target_labels.append(int(token_values[0]))
                    target_slots.append(0)
                    for slot_index, label in enumerate(token_values[1:], start=1):
                        target_positions.append(target_start + slot_index - 1)
                        target_labels.append(int(label))
                        target_slots.append(slot_index)

            prediction_blocks.append((block_start, target_end, history_block_ends[target_index]))

        sample_embeddings = torch.cat(embeddings, dim=0)
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=self.device)
        seq_len = sample_embeddings.shape[0]
        loss_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        if target_positions:
            loss_mask[torch.tensor(target_positions, dtype=torch.long, device=self.device)] = True
        min_dtype = torch.finfo(self.compute_dtype).min
        attention_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=self.compute_dtype, device=self.device)

        if history_region_end >= 0:
            history_mask = torch.triu(
                torch.full((history_region_end + 1, history_region_end + 1), fill_value=min_dtype, dtype=self.compute_dtype, device=self.device),
                diagonal=1,
            )
            attention_mask[:history_region_end + 1, :history_region_end + 1] = history_mask

        for block_start, block_end, visible_history_end in prediction_blocks:
            if visible_history_end >= 0:
                attention_mask[block_start:block_end + 1, :visible_history_end + 1] = 0
            block_len = block_end - block_start + 1
            attention_mask[block_start:block_end + 1, block_start:block_end + 1] = torch.triu(
                torch.full((block_len, block_len), fill_value=min_dtype, dtype=self.compute_dtype, device=self.device),
                diagonal=1,
            )

        return {
            'inputs_embeds': sample_embeddings,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'target_positions': target_positions,
            'target_labels': target_labels,
            'target_slots': target_slots,
        }

    def _build_finetune_batch_inputs(self, batch):
        packed_samples = [self._build_finetune_sample_inputs(sample) for sample in batch]
        max_len = max(sample['inputs_embeds'].shape[0] for sample in packed_samples)
        batch_size = len(packed_samples)
        hidden_size = packed_samples[0]['inputs_embeds'].shape[-1]
        padded_embeds = torch.zeros((batch_size, max_len, hidden_size), dtype=self.compute_dtype, device=self.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        min_dtype = torch.finfo(self.compute_dtype).min
        padded_attention_mask = torch.full(
            (batch_size, 1, max_len, max_len),
            fill_value=min_dtype,
            dtype=self.compute_dtype,
            device=self.device,
        )
        padded_loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

        target_labels = []
        target_slots = []

        for batch_index, sample in enumerate(packed_samples):
            seq_len = sample['inputs_embeds'].shape[0]
            padded_embeds[batch_index, :seq_len] = sample['inputs_embeds']
            padded_position_ids[batch_index, :seq_len] = sample['position_ids']
            padded_attention_mask[batch_index, 0, :seq_len, :seq_len] = sample['attention_mask']
            padded_loss_mask[batch_index, :seq_len] = sample['loss_mask']
            for pad_pos in range(seq_len, max_len):
                padded_attention_mask[batch_index, 0, pad_pos, pad_pos] = 0

            target_labels.extend(sample['target_labels'])
            target_slots.extend(sample['target_slots'])

        return (
            padded_embeds,
            padded_attention_mask,
            padded_position_ids,
            padded_loss_mask,
            torch.tensor(target_labels, dtype=torch.long, device=self.device),
            torch.tensor(target_slots, dtype=torch.long, device=self.device),
        )

    def _build_alignment_sample_inputs(self, sample, source_view: str):
        prefix_ids = [int(token_id) for token_id in self.compiled.prompt_align['align_prefix_ids']]
        bridge_ids = [int(token_id) for token_id in self.compiled.prompt_align['align_bridge_ids']]
        target_uid = int(sample['target_uid'])

        specs = [('model_tokens', prefix_ids)]
        specs.extend(self._render_single_view_item(target_uid, source_view))
        specs.append(('model_tokens', bridge_ids))

        embeddings = []
        target_positions = []
        target_labels = []
        target_slots = []

        def append_embedded(tensor: torch.Tensor):
            start = sum(piece.shape[0] for piece in embeddings)
            embeddings.append(tensor)
            return start, start + tensor.shape[0] - 1

        for kind, value in specs:
            append_embedded(self._embed_spec(kind, value))

        marker_start, marker_end = append_embedded(self._embed_spec('type_marker', self.config.task_type))

        if self.config.task_type == 'embedding':
            target_positions.append(marker_end)
            target_labels.append(self._target_embedding_index(target_uid))
        else:
            token_values = self._target_token_values(target_uid)
            target_kind = 'uid' if self.config.task_type == 'uid' else 'sid'
            target_value = token_values[0] if target_kind == 'uid' else token_values
            target_start, _ = append_embedded(self._embed_spec(target_kind, target_value))
            if self.config.task_type == 'uid':
                target_positions.append(marker_end)
                target_labels.append(int(token_values[0]))
            else:
                target_positions.append(marker_end)
                target_labels.append(int(token_values[0]))
                target_slots.append(0)
                for slot_index, label in enumerate(token_values[1:], start=1):
                    target_positions.append(target_start + slot_index - 1)
                    target_labels.append(int(label))
                    target_slots.append(slot_index)

        sample_embeddings = torch.cat(embeddings, dim=0)
        seq_len = sample_embeddings.shape[0]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        loss_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        if target_positions:
            loss_mask[torch.tensor(target_positions, dtype=torch.long, device=self.device)] = True
        min_dtype = torch.finfo(self.compute_dtype).min
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=self.compute_dtype, device=self.device),
            diagonal=1,
        )

        return {
            'inputs_embeds': sample_embeddings,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'target_labels': target_labels,
            'target_slots': target_slots,
        }

    def _build_alignment_batch_inputs(self, batch, source_view: str):
        packed_samples = [self._build_alignment_sample_inputs(sample, source_view) for sample in batch]
        max_len = max(sample['inputs_embeds'].shape[0] for sample in packed_samples)
        batch_size = len(packed_samples)
        hidden_size = packed_samples[0]['inputs_embeds'].shape[-1]
        padded_embeds = torch.zeros((batch_size, max_len, hidden_size), dtype=self.compute_dtype, device=self.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        min_dtype = torch.finfo(self.compute_dtype).min
        padded_attention_mask = torch.full(
            (batch_size, 1, max_len, max_len),
            fill_value=min_dtype,
            dtype=self.compute_dtype,
            device=self.device,
        )
        padded_loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

        target_labels = []
        target_slots = []

        for batch_index, sample in enumerate(packed_samples):
            seq_len = sample['inputs_embeds'].shape[0]
            padded_embeds[batch_index, :seq_len] = sample['inputs_embeds']
            padded_position_ids[batch_index, :seq_len] = sample['position_ids']
            padded_attention_mask[batch_index, 0, :seq_len, :seq_len] = sample['attention_mask']
            padded_loss_mask[batch_index, :seq_len] = sample['loss_mask']
            for pad_pos in range(seq_len, max_len):
                padded_attention_mask[batch_index, 0, pad_pos, pad_pos] = 0
            target_labels.extend(sample['target_labels'])
            target_slots.extend(sample['target_slots'])

        return (
            padded_embeds,
            padded_attention_mask,
            padded_position_ids,
            padded_loss_mask,
            torch.tensor(target_labels, dtype=torch.long, device=self.device),
            torch.tensor(target_slots, dtype=torch.long, device=self.device),
        )

    def _compute_uid_loss(self, pooled: torch.Tensor, batch):
        logits = self.target_head(pooled)
        labels = torch.tensor([sample['target_uid'] for sample in batch], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits.float(), labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, {
            'uid_acc': accuracy.item(),
        }

    def _compute_embedding_loss(self, pooled: torch.Tensor, batch):
        target_indices = torch.tensor(
            [int(self.compiled.item_views['embedding'][sample['target_uid']]) for sample in batch],
            dtype=torch.long,
            device=self.device,
        )
        targets = self.embedding_matrix[target_indices].to(dtype=self.compute_dtype)
        predictions = self.embedding_head(pooled)
        norm_predictions = F.normalize(predictions.float(), dim=-1)
        norm_table = F.normalize(self.embedding_matrix.float(), dim=-1)
        logits = norm_predictions @ norm_table.T
        loss = F.cross_entropy(logits, target_indices)
        cosine = F.cosine_similarity(predictions.float(), targets.float(), dim=-1).mean()
        accuracy = (logits.argmax(dim=-1) == target_indices).float().mean()
        return loss, {
            'embedding_cosine': cosine.item(),
            'embedding_acc': accuracy.item(),
        }

    def forward_next_item_batch(self, batch):
        inputs_embeds, attention_mask, lengths = self._build_batch_inputs(batch)
        hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths - 1]

        if self.config.task_type == 'uid':
            loss, metrics = self._compute_uid_loss(pooled, batch)
            labels = torch.tensor([sample['target_uid'] for sample in batch], dtype=torch.long, device=self.device)
            logits = self.target_head(pooled)
            metrics.update(self._compute_ranking_metrics_from_logits(logits, labels))
            return loss, metrics
        if self.config.task_type == 'sid':
            loss, metrics = self._compute_sid_loss(batch)
            metrics.update(self._compute_sid_ranking_metrics(batch))
            return loss, metrics
        loss, metrics = self._compute_embedding_loss(pooled, batch)
        target_indices = torch.tensor(
            [int(self.compiled.item_views['embedding'][sample['target_uid']]) for sample in batch],
            dtype=torch.long,
            device=self.device,
        )
        predictions = self.embedding_head(pooled)
        norm_predictions = F.normalize(predictions.float(), dim=-1)
        norm_table = F.normalize(self.embedding_matrix.float(), dim=-1)
        logits = norm_predictions @ norm_table.T
        metrics.update(self._compute_ranking_metrics_from_logits(logits, target_indices))
        return loss, metrics

    def forward_finetune_batch(self, batch):
        (
            inputs_embeds,
            attention_mask,
            position_ids,
            loss_mask,
            target_labels,
            target_slots,
        ) = self._build_finetune_batch_inputs(batch)

        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        selected_hidden = hidden[loss_mask]

        if self.config.task_type == 'uid':
            logits = self.target_head(selected_hidden)
            loss = F.cross_entropy(logits.float(), target_labels)
            accuracy = (logits.argmax(dim=-1) == target_labels).float().mean()
            return loss, {
                'uid_acc': accuracy.item(),
            }

        if self.config.task_type == 'embedding':
            predictions = self.embedding_head(selected_hidden)
            targets = self.embedding_matrix[target_labels].to(dtype=self.compute_dtype)
            norm_predictions = F.normalize(predictions.float(), dim=-1)
            norm_table = F.normalize(self.embedding_matrix.float(), dim=-1)
            logits = norm_predictions @ norm_table.T
            loss = F.cross_entropy(logits, target_labels)
            cosine = F.cosine_similarity(predictions.float(), targets.float(), dim=-1).mean()
            accuracy = (logits.argmax(dim=-1) == target_labels).float().mean()
            return loss, {
                'embedding_cosine': cosine.item(),
                'embedding_acc': accuracy.item(),
            }

        logits = self.target_head(selected_hidden)
        loss = F.cross_entropy(logits.float(), target_labels)
        token_acc = (logits.argmax(dim=-1) == target_labels).float().mean()
        return loss, {
            'sid_token_acc': token_acc.item(),
        }

    def forward_alignment_batch(self, batch, source_view: str):
        (
            inputs_embeds,
            attention_mask,
            position_ids,
            loss_mask,
            target_labels,
            target_slots,
        ) = self._build_alignment_batch_inputs(batch, source_view)

        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        selected_hidden = hidden[loss_mask]

        if self.config.task_type == 'uid':
            logits = self.target_head(selected_hidden)
            loss = F.cross_entropy(logits.float(), target_labels)
            accuracy = (logits.argmax(dim=-1) == target_labels).float().mean()
            return loss, {
                'uid_acc': accuracy.item(),
            }

        if self.config.task_type == 'embedding':
            predictions = self.embedding_head(selected_hidden)
            targets = self.embedding_matrix[target_labels].to(dtype=self.compute_dtype)
            norm_predictions = F.normalize(predictions.float(), dim=-1)
            norm_table = F.normalize(self.embedding_matrix.float(), dim=-1)
            logits = norm_predictions @ norm_table.T
            loss = F.cross_entropy(logits, target_labels)
            cosine = F.cosine_similarity(predictions.float(), targets.float(), dim=-1).mean()
            accuracy = (logits.argmax(dim=-1) == target_labels).float().mean()
            return loss, {
                'embedding_cosine': cosine.item(),
                'embedding_acc': accuracy.item(),
            }

        logits = self.target_head(selected_hidden)
        loss = F.cross_entropy(logits.float(), target_labels)
        token_acc = (logits.argmax(dim=-1) == target_labels).float().mean()
        return loss, {
            'sid_token_acc': token_acc.item(),
        }

    def forward(self, batch, mode: str, source_view: str | None = None):
        if mode == 'finetune':
            return self.forward_finetune_batch(batch)
        if mode == 'alignment':
            if source_view is None:
                raise ValueError('alignment forward requires source_view')
            return self.forward_alignment_batch(batch, source_view=source_view)
        if mode == 'test':
            return self.forward_next_item_batch(batch)
        raise ValueError(f'Unsupported forward mode: {mode}')
