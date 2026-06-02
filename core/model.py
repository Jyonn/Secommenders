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
                lora_layers=config.lora_layers,
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
        input_embed_dim = getattr(self.encoder, 'input_embed_dim', hidden_size)
        self.type_marker_embedding = nn.Embedding(len(self.compiled.special_vocab['tokens']), input_embed_dim)
        self.uid_embedding = nn.Embedding(compiled.num_items, input_embed_dim)
        self.sid_embedding = nn.Embedding(max(compiled.sid_vocab_size, 1), input_embed_dim)
        self.embedding_projection = None
        self.embedding_head = None
        self.uid_head = None
        self.sid_head = None
        self.model_token_head = None

        if compiled.embedding_matrix is not None:
            self.register_buffer('embedding_matrix', compiled.embedding_matrix)
            self.embedding_projection = nn.Linear(compiled.embedding_matrix.shape[1], input_embed_dim, bias=False)
        else:
            self.register_buffer('embedding_matrix', torch.empty(0))

        supervised_repr_types = set(config.compile_config.repr_types)
        if 'uid' in supervised_repr_types or config.task_type == 'uid':
            self.uid_head = nn.Linear(hidden_size, compiled.num_items)
        if 'sid' in supervised_repr_types or config.task_type == 'sid':
            if not compiled.sid_vocab_size or not compiled.sid_num_quantizers:
                raise ValueError('sid task requires sid vocab metadata in compiled artifacts')
            self.sid_head = nn.Linear(hidden_size, compiled.sid_vocab_size)
        if 'text' in supervised_repr_types:
            self.model_token_head = nn.Linear(hidden_size, input_embed_dim, bias=False)
        if 'embedding' in supervised_repr_types or config.task_type == 'embedding':
            if compiled.embedding_matrix is None:
                raise ValueError('embedding supervision requires compiled embedding view and source matrix')
            self.embedding_head = nn.Linear(hidden_size, compiled.embedding_matrix.shape[1], bias=False)
        if config.task_type not in {'uid', 'sid', 'embedding'}:
            raise ValueError(f'Unsupported task type: {config.task_type}')

        self.compute_dtype = getattr(self.encoder, 'compute_dtype', torch.float32)
        self.sid_collision_loss_weight = float(getattr(config, 'sid_collision_loss_weight', 0.1))
        self.type_marker_embedding.to(dtype=self.compute_dtype)
        self.uid_embedding.to(dtype=self.compute_dtype)
        self.sid_embedding.to(dtype=self.compute_dtype)
        if self.embedding_projection is not None:
            self.embedding_projection.to(dtype=self.compute_dtype)
        if self.embedding_head is not None:
            self.embedding_head.to(dtype=self.compute_dtype)
        if self.uid_head is not None:
            self.uid_head.to(dtype=self.compute_dtype)
        if self.sid_head is not None:
            self.sid_head.to(dtype=self.compute_dtype)
        if self.model_token_head is not None:
            self.model_token_head.to(dtype=self.compute_dtype)

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

    def _text_logits(self, hidden_states: torch.Tensor):
        if self.model_token_head is None:
            raise ValueError('text supervision requested but model_token_head is not initialized')
        projected = self.model_token_head(hidden_states)
        token_table = self.encoder.get_input_embedding_weight().to(dtype=self.compute_dtype)
        return projected.float() @ token_table.float().T

    def _uid_logits(self, hidden_states: torch.Tensor):
        if self.uid_head is None:
            raise ValueError('uid supervision requested but uid_head is not initialized')
        return self.uid_head(hidden_states)

    def _sid_logits(self, hidden_states: torch.Tensor):
        if self.sid_head is None:
            raise ValueError('sid supervision requested but sid_head is not initialized')
        return self.sid_head(hidden_states)

    def _sid_slot_allowed_codes(self, slot_index: int):
        base_num_quantizers = int(self.compiled.sid_base_num_quantizers or 0)
        codebook_size = int(self.compiled.sid_codebook_size or 0)
        collision_offset = int(self.compiled.sid_collision_token_offset or 0)
        collision_vocab_size = int(self.compiled.sid_collision_vocab_size or 0)

        if slot_index < 0:
            raise ValueError(f'Invalid sid slot index: {slot_index}')
        if slot_index < base_num_quantizers:
            start = slot_index * codebook_size
            return list(range(start, start + codebook_size))
        if slot_index == base_num_quantizers:
            return list(range(collision_offset, collision_offset + collision_vocab_size))
        raise ValueError(
            f'SID slot index {slot_index} exceeds configured slots '
            f'(base={base_num_quantizers}, final={self.compiled.sid_num_quantizers})'
        )

    def _mask_sid_logits_for_slots(self, logits: torch.Tensor, slot_indices: torch.Tensor):
        masked_logits = torch.full_like(logits, fill_value=torch.finfo(logits.dtype).min)
        unique_slots = sorted({int(slot) for slot in slot_indices.tolist()})
        for slot_index in unique_slots:
            row_indices = (slot_indices == slot_index).nonzero(as_tuple=False).view(-1)
            if row_indices.numel() == 0:
                continue
            allowed_codes = self._sid_slot_allowed_codes(slot_index)
            allowed_tensor = torch.tensor(allowed_codes, dtype=torch.long, device=logits.device)
            masked_logits[row_indices[:, None], allowed_tensor[None, :]] = logits[row_indices[:, None], allowed_tensor[None, :]]
        return masked_logits

    def _sid_loss_weights(self, slot_indices: torch.Tensor):
        base_num_quantizers = int(self.compiled.sid_base_num_quantizers or 0)
        weights = torch.ones(slot_indices.shape[0], dtype=torch.float32, device=slot_indices.device)
        collision_mask = slot_indices >= base_num_quantizers
        if collision_mask.any():
            weights[collision_mask] = self.sid_collision_loss_weight
        return weights

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
        return self._sid_logits(pooled)

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
                slot_tensor = torch.tensor([slot_index], dtype=torch.long, device=self.device)
                masked_logits = self._mask_sid_logits_for_slots(logits, slot_tensor)
                label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
                token_loss = F.cross_entropy(masked_logits.float(), label_tensor, reduction='none')
                token_loss = token_loss * self._sid_loss_weights(slot_tensor)
                sample_loss = sample_loss + token_loss.squeeze(0)
                pred = int(masked_logits.argmax(dim=-1).item())
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

    def _repr_payload_labels(self, repr_type: str, uid: int):
        if repr_type == 'uid':
            return [int(self.compiled.item_views['uid'][uid])]
        if repr_type == 'sid':
            return [int(token_id) for token_id in function.to_list(self.compiled.item_views['sid'][uid])]
        if repr_type == 'text':
            return [int(token_id) for token_id in function.to_list(self.compiled.item_views['text'][uid])]
        if repr_type == 'embedding':
            return [self._target_embedding_index(uid)]
        raise ValueError(f'Unsupported repr type for supervision: {repr_type}')

    def _build_repr_supervision(self, repr_type: str, uid: int, marker_position: int, payload_positions: list[int], group: str):
        labels = self._repr_payload_labels(repr_type, uid)
        if not labels:
            return []
        if repr_type == 'embedding':
            return [{'kind': 'embedding', 'position': int(marker_position), 'label': int(labels[0]), 'group': group, 'slot': -1}]
        anchor_positions = [int(marker_position)] + [int(position) for position in payload_positions[:-1]]
        if repr_type == 'sid':
            return [
                {'kind': repr_type, 'position': anchor_position, 'label': int(label), 'group': group, 'slot': slot_index}
                for slot_index, (anchor_position, label) in enumerate(zip(anchor_positions, labels))
            ]
        return [
            {'kind': repr_type, 'position': anchor_position, 'label': int(label), 'group': group, 'slot': -1}
            for anchor_position, label in zip(anchor_positions, labels)
        ]

    def _build_finetune_sample_inputs(self, sample):
        separator_ids = [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]
        sequence_uids = sample['sequence_uids']
        embeddings = []
        supervision = []

        def append_embedded(tensor: torch.Tensor):
            start = sum(piece.shape[0] for piece in embeddings)
            embeddings.append(tensor)
            return start, start + tensor.shape[0] - 1

        for item_index, uid in enumerate(sequence_uids):
            if item_index > 0 and separator_ids:
                append_embedded(self._embed_spec('model_tokens', separator_ids))

            include_alignment_repr = item_index < len(sequence_uids) - 1
            repr_types = [self.config.task_type]
            if include_alignment_repr:
                repr_types.extend([repr_type for repr_type in self.config.compile_config.repr_types[1:]])

            for repr_type in repr_types:
                segment_specs = self._render_single_view_item(uid, repr_type)
                marker_position = None
                payload_positions = []
                for kind, value in segment_specs:
                    start, end = append_embedded(self._embed_spec(kind, value))
                    if kind == 'type_marker':
                        marker_position = start
                    else:
                        payload_positions.extend(range(start, end + 1))

                if repr_type == self.config.task_type and item_index > 0:
                    supervision.extend(
                        self._build_repr_supervision(
                            repr_type=repr_type,
                            uid=uid,
                            marker_position=marker_position,
                            payload_positions=payload_positions,
                            group='primary',
                        )
                    )
                elif repr_type != self.config.task_type and self.config.alignment_enable and include_alignment_repr:
                    supervision.extend(
                        self._build_repr_supervision(
                            repr_type=repr_type,
                            uid=uid,
                            marker_position=marker_position,
                            payload_positions=payload_positions,
                            group='alignment',
                        )
                    )

        sample_embeddings = torch.cat(embeddings, dim=0)
        return {
            'inputs_embeds': sample_embeddings,
            'supervision': supervision,
        }

    def _build_finetune_batch_inputs(self, batch):
        packed_samples = [self._build_finetune_sample_inputs(sample) for sample in batch]
        max_len = max(sample['inputs_embeds'].shape[0] for sample in packed_samples)
        batch_size = len(packed_samples)
        hidden_size = packed_samples[0]['inputs_embeds'].shape[-1]
        padded_embeds = torch.zeros((batch_size, max_len, hidden_size), dtype=self.compute_dtype, device=self.device)
        lengths = torch.tensor([sample['inputs_embeds'].shape[0] for sample in packed_samples], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        supervision_batch_indices = []
        supervision_positions = []
        supervision_kinds = []
        supervision_labels = []
        supervision_groups = []
        supervision_slots = []

        for batch_index, sample in enumerate(packed_samples):
            seq_len = sample['inputs_embeds'].shape[0]
            padded_embeds[batch_index, :seq_len] = sample['inputs_embeds']
            for entry in sample['supervision']:
                supervision_batch_indices.append(batch_index)
                supervision_positions.append(int(entry['position']))
                supervision_kinds.append(entry['kind'])
                supervision_labels.append(int(entry['label']))
                supervision_groups.append(entry['group'])
                supervision_slots.append(int(entry.get('slot', -1)))

        return (
            padded_embeds,
            attention_mask.long(),
            lengths,
            {
                'batch_indices': torch.tensor(supervision_batch_indices, dtype=torch.long, device=self.device),
                'positions': torch.tensor(supervision_positions, dtype=torch.long, device=self.device),
                'kinds': supervision_kinds,
                'labels': torch.tensor(supervision_labels, dtype=torch.long, device=self.device),
                'groups': supervision_groups,
                'slots': torch.tensor(supervision_slots, dtype=torch.long, device=self.device),
            },
        )

    def _compute_uid_loss(self, pooled: torch.Tensor, batch):
        logits = self._uid_logits(pooled)
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

    def _compute_mixed_supervision_loss(self, selected_hidden: torch.Tensor, supervision: dict):
        kinds = supervision['kinds']
        labels = supervision['labels']
        groups = supervision['groups']
        slots = supervision['slots']
        primary_losses = []
        alignment_losses = []
        metrics = {}

        for kind in ['uid', 'sid', 'text', 'embedding']:
            mask_indices = [index for index, entry_kind in enumerate(kinds) if entry_kind == kind]
            if not mask_indices:
                continue
            index_tensor = torch.tensor(mask_indices, dtype=torch.long, device=self.device)
            kind_hidden = selected_hidden[index_tensor]
            kind_labels = labels[index_tensor]
            kind_slots = slots[index_tensor]
            group_mask = [groups[index] for index in mask_indices]

            if kind == 'uid':
                logits = self._uid_logits(kind_hidden)
                losses = F.cross_entropy(logits.float(), kind_labels, reduction='none')
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == kind_labels).float().mean().item()
                metrics['uid_acc'] = accuracy
            elif kind == 'sid':
                logits = self._sid_logits(kind_hidden)
                masked_logits = self._mask_sid_logits_for_slots(logits, kind_slots)
                losses = F.cross_entropy(masked_logits.float(), kind_labels, reduction='none')
                losses = losses * self._sid_loss_weights(kind_slots)
                predictions = masked_logits.argmax(dim=-1)
                accuracy = (predictions == kind_labels).float().mean().item()
                metrics['sid_token_acc'] = accuracy
            elif kind == 'text':
                logits = self._text_logits(kind_hidden)
                losses = F.cross_entropy(logits, kind_labels, reduction='none')
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == kind_labels).float().mean().item()
                metrics['text_token_acc'] = accuracy
            else:
                predictions = self.embedding_head(kind_hidden)
                targets = self.embedding_matrix[kind_labels].to(dtype=self.compute_dtype)
                norm_predictions = F.normalize(predictions.float(), dim=-1)
                norm_table = F.normalize(self.embedding_matrix.float(), dim=-1)
                logits = norm_predictions @ norm_table.T
                losses = F.cross_entropy(logits, kind_labels, reduction='none')
                accuracy = (logits.argmax(dim=-1) == kind_labels).float().mean().item()
                cosine = F.cosine_similarity(predictions.float(), targets.float(), dim=-1).mean().item()
                metrics['embedding_acc'] = accuracy
                metrics['embedding_cosine'] = cosine

            for local_index, loss_value in enumerate(losses):
                if group_mask[local_index] == 'primary':
                    primary_losses.append(loss_value)
                else:
                    alignment_losses.append(loss_value)

        if not primary_losses:
            raise RuntimeError('no primary supervision entries were constructed for finetune batch')

        primary_loss = torch.stack(primary_losses).mean()
        if alignment_losses:
            alignment_loss = torch.stack(alignment_losses).mean()
            total_loss = primary_loss + self.config.alignment_weight * alignment_loss
            metrics['alignment_loss'] = float(alignment_loss.item())
        else:
            alignment_loss = None
            total_loss = primary_loss
        metrics['primary_loss'] = float(primary_loss.item())
        return total_loss, metrics, primary_loss, alignment_loss

    def forward_next_item_batch(self, batch):
        inputs_embeds, attention_mask, lengths = self._build_batch_inputs(batch)
        hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths - 1]

        if self.config.task_type == 'uid':
            loss, metrics = self._compute_uid_loss(pooled, batch)
            labels = torch.tensor([sample['target_uid'] for sample in batch], dtype=torch.long, device=self.device)
            logits = self._uid_logits(pooled)
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
            lengths,
            supervision,
        ) = self._build_finetune_batch_inputs(batch)

        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        selected_hidden = hidden[supervision['batch_indices'], supervision['positions']]
        loss, metrics, primary_loss, alignment_loss = self._compute_mixed_supervision_loss(selected_hidden, supervision)
        return loss, metrics

    def forward(self, batch, mode: str, source_view: str | None = None):
        if mode == 'finetune':
            return self.forward_finetune_batch(batch)
        if mode == 'test':
            return self.forward_next_item_batch(batch)
        raise ValueError(f'Unsupported forward mode: {mode}')
