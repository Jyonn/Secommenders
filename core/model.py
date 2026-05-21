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
            self.target_head = nn.Linear(hidden_size, compiled.sid_vocab_size * compiled.sid_num_quantizers)
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

    def _compute_uid_loss(self, pooled: torch.Tensor, batch):
        logits = self.target_head(pooled)
        labels = torch.tensor([sample['target_uid'] for sample in batch], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits.float(), labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, {
            'uid_acc': accuracy.item(),
        }

    def _compute_sid_loss(self, pooled: torch.Tensor, batch):
        sid_targets = [
            [int(token_id) for token_id in function.to_list(self.compiled.item_views['sid'][sample['target_uid']])]
            for sample in batch
        ]
        labels = torch.tensor(sid_targets, dtype=torch.long, device=self.device)
        logits = self.target_head(pooled).view(len(batch), self.compiled.sid_num_quantizers, self.compiled.sid_vocab_size)
        loss = F.cross_entropy(logits.float().reshape(-1, self.compiled.sid_vocab_size), labels.reshape(-1))
        token_acc = (logits.argmax(dim=-1) == labels).float().mean()
        seq_acc = (logits.argmax(dim=-1) == labels).all(dim=-1).float().mean()
        return loss, {
            'sid_token_acc': token_acc.item(),
            'sid_seq_acc': seq_acc.item(),
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
            return self._compute_uid_loss(pooled, batch)
        if self.config.task_type == 'sid':
            return self._compute_sid_loss(pooled, batch)
        return self._compute_embedding_loss(pooled, batch)

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

        logits = self.target_head(selected_hidden).view(-1, self.compiled.sid_num_quantizers, self.compiled.sid_vocab_size)
        selected_logits = logits[torch.arange(logits.shape[0], device=self.device), target_slots]
        loss = F.cross_entropy(selected_logits.float(), target_labels)
        token_acc = (selected_logits.argmax(dim=-1) == target_labels).float().mean()
        return loss, {
            'sid_token_acc': token_acc.item(),
        }

    def forward(self, batch, mode: str):
        if mode == 'finetune':
            return self.forward_finetune_batch(batch)
        if mode == 'test':
            return self.forward_next_item_batch(batch)
        raise ValueError(f'Unsupported forward mode: {mode}')
