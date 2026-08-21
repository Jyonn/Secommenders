import hashlib
import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models import build_backbone
from utils import function
from utils.multi_decoding import fuse_candidate_scores, normalize_candidate_scores, uid_frequency_gate

from .encoders import LLMSequenceEncoder, ScratchLlamaSequenceEncoder, ScratchSequenceEncoder
from .uid_hierarchy import UIDHierarchyArtifacts


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
        task_types = set(config.task_types)
        self.uid_hierarchical_decoding = task_types == {'uid'} and str(getattr(config, 'uid_decoding', 'flat')).lower() == 'hierarchical'

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
        elif compiled.model_kind == 'scratch':
            self.encoder = ScratchLlamaSequenceEncoder(
                vocab_size=compiled.model_vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_length=compiled.meta['model_max_length'],
            )
        elif compiled.model_kind == 'scratchlegacy':
            self.encoder = ScratchSequenceEncoder(
                vocab_size=compiled.model_vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_length=compiled.meta['model_max_length'],
            )
        else:
            raise ValueError(f'Unsupported compiled model kind: {compiled.model_kind}')

        hidden_size = self.encoder.hidden_size
        input_embed_dim = getattr(self.encoder, 'input_embed_dim', hidden_size)
        history_repr_names = list(config.compile_config.representation_names)
        self.representation_pair_bias_enabled = bool(
            getattr(config, 'representation_pair_bias', False)
        )
        self.attention_representation_names = ['model', *history_repr_names]
        self.attention_representation_to_id = {
            name: index for index, name in enumerate(self.attention_representation_names)
        }
        self.representation_pair_bias = (
            nn.Parameter(torch.zeros(
                len(self.attention_representation_names),
                len(self.attention_representation_names),
            ))
            if self.representation_pair_bias_enabled else None
        )
        history_repr_types = {config.compile_config.representation_kind(name) for name in history_repr_names}
        self.uses_uid_path = 'uid' in history_repr_types or 'uid' in task_types or config.repr_combine == 'add'
        self.uses_sid_path = 'sid' in history_repr_types or 'sid' in task_types
        self.uses_hash_path = 'hash' in history_repr_types or 'hash' in task_types
        self.type_marker_embedding = nn.Embedding(len(self.compiled.special_vocab['tokens']), input_embed_dim)
        self.uid_embedding = nn.Embedding(compiled.num_items, input_embed_dim) if self.uses_uid_path else None
        self.sid_embeddings = nn.ModuleDict()
        self.sid_heads = nn.ModuleDict()
        for name in config.compile_config.names_for_kind('sid'):
            vocab_size = compiled.sid_vocab_size_for(name)
            if vocab_size <= 0:
                raise ValueError(f'sid representation {name} requires compiled vocabulary metadata')
            self.sid_embeddings[name] = nn.Embedding(vocab_size, input_embed_dim)
        self.hash_embedding = nn.Embedding(max(compiled.hash_vocab_size, 1), input_embed_dim) if self.uses_hash_path else None
        self.embedding_projection = None
        self.embedding_head = None
        self.embedding_tables = nn.ModuleDict()
        self.embedding_projections = nn.ModuleDict()
        self.embedding_heads = nn.ModuleDict()
        self.uid_head = None
        self.uid_hierarchy = None
        self.uid_node_heads = None
        self.hash_head = None
        self.model_token_head = None

        if config.compile_config.representation_graph:
            for name, matrix in compiled.embedding_matrices.items():
                self.embedding_tables[name] = nn.Embedding.from_pretrained(matrix, freeze=True)
                self.embedding_projections[name] = nn.Linear(matrix.shape[1], input_embed_dim, bias=False)
                if name in history_repr_names or name in config.compile_config.target_names:
                    self.embedding_heads[name] = nn.Linear(hidden_size, matrix.shape[1], bias=False)
            self.register_buffer('embedding_matrix', torch.empty(0))
        elif compiled.embedding_matrix is not None:
            self.register_buffer('embedding_matrix', compiled.embedding_matrix)
            self.embedding_projection = nn.Linear(compiled.embedding_matrix.shape[1], input_embed_dim, bias=False)
        else:
            self.register_buffer('embedding_matrix', torch.empty(0))

        supervised_repr_types = history_repr_types
        if self.uid_hierarchical_decoding:
            hierarchy_dir = getattr(config, 'uid_hierarchy_dir', None)
            if not hierarchy_dir:
                raise ValueError('uid hierarchical decoding requires prepared hierarchy artifacts')
            self.uid_hierarchy = UIDHierarchyArtifacts(hierarchy_dir, compiled, config.uid_cluster_topk)
            self.uid_node_heads = nn.ModuleList(
                [nn.Linear(hidden_size, child_count) for child_count in self.uid_hierarchy.node_child_counts]
            )
        elif 'uid' in supervised_repr_types or 'uid' in task_types:
            self.uid_head = nn.Linear(hidden_size, compiled.num_items)
        if 'sid' in supervised_repr_types or 'sid' in task_types:
            for name in config.compile_config.names_for_kind('sid'):
                meta = compiled.sid_metadata_for(name)
                if not meta['vocab_size'] or not meta['num_quantizers']:
                    raise ValueError(f'sid representation {name} requires vocabulary metadata')
                self.sid_heads[name] = nn.Linear(hidden_size, meta['vocab_size'])
        if 'hash' in supervised_repr_types or 'hash' in task_types:
            if not compiled.hash_vocab_size or not compiled.hash_num_tokens:
                raise ValueError('hash task requires hash vocab metadata in compiled artifacts')
            self.hash_head = nn.Linear(hidden_size, compiled.hash_vocab_size)
        if 'text' in supervised_repr_types:
            self.model_token_head = nn.Linear(hidden_size, input_embed_dim, bias=False)
        if not config.compile_config.representation_graph and ('embedding' in supervised_repr_types or 'embedding' in task_types):
            if compiled.embedding_matrix is None:
                raise ValueError('embedding supervision requires compiled embedding view and source matrix')
            self.embedding_head = nn.Linear(hidden_size, compiled.embedding_matrix.shape[1], bias=False)
        if not task_types.issubset({'uid', 'sid', 'hash', 'embedding'}):
            raise ValueError(f'Unsupported task type: {config.task_type}')

        self.compute_dtype = getattr(self.encoder, 'compute_dtype', torch.float32)
        self.code_collision_loss_weight = float(getattr(config, 'code_collision_loss_weight', 0.1))
        self._ranking_trace_enabled = False
        self._ranking_trace_records = []
        self._sid_item_code_buffer_names = {}
        for index, name in enumerate(config.compile_config.names_for_kind('sid')):
            sid_item_codes = compiled.item_views.get(name) or []
            if sid_item_codes:
                sid_item_codes_tensor = torch.tensor(
                    [[int(code) for code in function.to_list(codes)] for codes in sid_item_codes],
                    dtype=torch.long,
                )
            else:
                sid_item_codes_tensor = torch.empty((0, 0), dtype=torch.long)
            buffer_name = f'_sid_item_codes_{index}'
            self.register_buffer(buffer_name, sid_item_codes_tensor, persistent=False)
            self._sid_item_code_buffer_names[name] = buffer_name
        hash_item_codes = compiled.item_views.get('hash') or []
        if hash_item_codes:
            hash_item_codes_tensor = torch.tensor(
                [[int(code) for code in function.to_list(codes)] for codes in hash_item_codes],
                dtype=torch.long,
            )
        else:
            hash_item_codes_tensor = torch.empty((0, 0), dtype=torch.long)
        self.register_buffer('hash_item_codes', hash_item_codes_tensor, persistent=False)
        target_frequencies = torch.zeros(compiled.num_items, dtype=torch.float32)
        for sequence in compiled.finetune.get('sequence_uids', []):
            for uid in function.to_list(sequence)[1:]:
                target_frequencies[int(uid)] += 1.0
        self.register_buffer('item_target_frequencies', target_frequencies, persistent=False)
        self.type_marker_embedding.to(dtype=self.compute_dtype)
        if self.uid_embedding is not None:
            self.uid_embedding.to(dtype=self.compute_dtype)
        self.sid_embeddings.to(dtype=self.compute_dtype)
        if self.hash_embedding is not None:
            self.hash_embedding.to(dtype=self.compute_dtype)
        if self.embedding_projection is not None:
            self.embedding_projection.to(dtype=self.compute_dtype)
        if self.embedding_head is not None:
            self.embedding_head.to(dtype=self.compute_dtype)
        self.embedding_projections.to(dtype=self.compute_dtype)
        self.embedding_heads.to(dtype=self.compute_dtype)
        if self.uid_head is not None:
            self.uid_head.to(dtype=self.compute_dtype)
        if self.uid_node_heads is not None:
            for head in self.uid_node_heads:
                head.to(dtype=self.compute_dtype)
        self.sid_heads.to(dtype=self.compute_dtype)
        if self.hash_head is not None:
            self.hash_head.to(dtype=self.compute_dtype)
        if self.model_token_head is not None:
            self.model_token_head.to(dtype=self.compute_dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    def _primary_sid_name(self):
        if not hasattr(self.config, 'compile_config'):
            return 'sid'
        return (
            self.config.compile_config.primary_name('sid', targets=True)
            or self.config.compile_config.primary_name('sid')
        )

    def _resolve_sid_name(self, representation=None):
        name = representation or self._primary_sid_name()
        if not hasattr(self, 'sid_embeddings'):
            return name or 'sid'
        if not name or name not in self.sid_embeddings:
            raise ValueError(f'unknown SID representation: {name}')
        return name

    def _sid_meta(self, representation=None):
        name = self._resolve_sid_name(representation)
        if hasattr(self.compiled, 'sid_metadata_for'):
            return self.compiled.sid_metadata_for(name)
        return {
            'num_quantizers': getattr(self.compiled, 'sid_num_quantizers', 0),
            'base_num_quantizers': getattr(self.compiled, 'sid_base_num_quantizers', 0),
            'codebook_size': getattr(self.compiled, 'sid_codebook_size', 0),
            'collision_vocab_size': getattr(self.compiled, 'sid_collision_vocab_size', 0),
            'collision_token_offset': getattr(self.compiled, 'sid_collision_token_offset', 0),
            'recommended_decoding': getattr(self.compiled, 'sid_recommended_decoding', None),
        }

    def _sid_prefix_index(self, representation=None):
        name = self._resolve_sid_name(representation)
        indices = getattr(self.compiled, 'sid_prefix_to_next_by_name', None)
        return indices.get(name, {}) if indices is not None else self.compiled.sid_prefix_to_next

    def _sid_sequence_index(self, representation=None):
        name = self._resolve_sid_name(representation)
        indices = getattr(self.compiled, 'sid_sequence_to_items_by_name', None)
        return indices.get(name, {}) if indices is not None else self.compiled.sid_sequence_to_items

    @property
    def sid_embedding(self):
        name = self._primary_sid_name()
        return self.sid_embeddings[name] if name in self.sid_embeddings else None

    @property
    def sid_head(self):
        name = self._primary_sid_name()
        return self.sid_heads[name] if name in self.sid_heads else None

    def _sid_item_codes(self, representation=None):
        name = self._resolve_sid_name(representation)
        buffer_names = getattr(self, '_sid_item_code_buffer_names', {})
        if name in buffer_names:
            return getattr(self, buffer_names[name])
        legacy_codes = self.__dict__.get('sid_item_codes')
        if isinstance(legacy_codes, torch.Tensor):
            return legacy_codes
        raise ValueError(f'SID item-code buffer is unavailable for representation: {name}')

    @property
    def sid_item_codes(self):
        return self._sid_item_codes()

    def trainable_state_dict(self):
        state = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: tensor.detach().cpu() for name, tensor in state.items() if name in trainable_names}

    def enable_ranking_trace(self, enabled=True):
        self._ranking_trace_enabled = bool(enabled)
        self._ranking_trace_records = []

    def pop_ranking_trace_records(self):
        records = self._ranking_trace_records
        self._ranking_trace_records = []
        return records

    def _record_target_rank(self, sample, rank):
        if not self._ranking_trace_enabled:
            return
        self._ranking_trace_records.append({
            'target_uid': int(sample['target_uid']),
            'rank': int(rank) if rank is not None else None,
        })

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
        for repr_name in self.config.compile_config.representation_names:
            repr_type = self.config.compile_config.representation_kind(repr_name)
            if repr_type == 'uid':
                specs.append(('type_marker', repr_name))
                specs.append(('uid', uid))
            elif repr_type == 'text':
                specs.append(('type_marker', repr_name))
                token_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views[repr_name][uid])]
                specs.append(('model_tokens', token_ids))
            elif repr_type == 'sid':
                specs.append(('type_marker', repr_name))
                sid_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views[repr_name][uid])]
                specs.append(('sid', (repr_name, sid_ids)))
            elif repr_type == 'hash':
                specs.append(('type_marker', repr_name))
                hash_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views[repr_name][uid])]
                specs.append(('hash', hash_ids))
            elif repr_type == 'embedding':
                specs.append(('type_marker', repr_name))
                emb_index = int(self.compiled.item_views[repr_name][uid])
                specs.append(('embedding', (repr_name, emb_index)))
            else:
                raise ValueError(f'Unsupported repr type: {repr_type}')
        return specs

    def _render_single_view_item(self, uid: int, view_name: str):
        kind = self.config.compile_config.representation_kind(view_name)
        if kind == 'uid':
            return [('type_marker', view_name), ('uid', uid)]
        if kind == 'text':
            token_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views[view_name][uid])]
            return [('type_marker', view_name), ('model_tokens', token_ids)]
        if kind == 'sid':
            sid_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views[view_name][uid])]
            return [('type_marker', view_name), ('sid', (view_name, sid_ids))]
        if kind == 'hash':
            hash_ids = [int(token_id) for token_id in function.to_list(self.compiled.item_views[view_name][uid])]
            return [('type_marker', view_name), ('hash', hash_ids)]
        if kind == 'embedding':
            emb_index = int(self.compiled.item_views[view_name][uid])
            return [('type_marker', view_name), ('embedding', (view_name, emb_index))]
        raise ValueError(f'Unsupported alignment source view: {view_name}')

    def _build_sample_specs(self, sample):
        specs = [('model_tokens', [int(token_id) for token_id in self.compiled.prompt_main['history_prefix_ids']])]
        history_uids = sample['history_uids']
        for index, uid in enumerate(history_uids):
            specs.extend(self._render_history_item(uid))
            if index != len(history_uids) - 1:
                specs.append(('model_tokens', [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]))
        specs.append(('model_tokens', [int(token_id) for token_id in self.compiled.prompt_main['query_prefix_ids']]))
        specs.append(('type_marker', self._target_marker_name()))
        return specs

    def _target_marker_name(self):
        names = self.config.compile_config.target_names
        return names[0] if len(names) == 1 else 'decoder_' + '_'.join(names)

    def _embed_spec(self, kind: str, value):
        if kind == 'model_tokens':
            token_ids = torch.tensor(value, dtype=torch.long, device=self.device)
            return self.encoder.embed_model_tokens(token_ids)
        if kind == 'type_marker':
            marker_index = torch.tensor([self._type_marker_index(value)], dtype=torch.long, device=self.device)
            return self.type_marker_embedding(marker_index)
        if kind == 'uid':
            if self.uid_embedding is None:
                raise ValueError('uid embedding requested but uid path is not initialized')
            token_ids = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            return self.uid_embedding(token_ids)
        if kind == 'sid':
            if isinstance(value, tuple):
                name, token_values = value
            else:
                name, token_values = self._primary_sid_name(), value
            name = self._resolve_sid_name(name)
            token_ids = torch.tensor(token_values, dtype=torch.long, device=self.device)
            return self.sid_embeddings[name](token_ids)
        if kind == 'hash':
            if self.hash_embedding is None:
                raise ValueError('hash embedding requested but hash path is not initialized')
            token_ids = torch.tensor(value, dtype=torch.long, device=self.device)
            return self.hash_embedding(token_ids)
        if kind == 'embedding':
            if isinstance(value, tuple):
                name, index = value
                emb_index = torch.tensor([int(index)], dtype=torch.long, device=self.device)
                projected = self.embedding_projections[name](self.embedding_tables[name](emb_index).to(dtype=self.compute_dtype))
                return projected
            emb_index = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            projected = self.embedding_projection(self.embedding_matrix[emb_index].to(dtype=self.compute_dtype))
            return projected
        if kind == 'uid_embedding_add':
            if self.uid_embedding is None:
                raise ValueError('uid+embedding fused path requested but uid embedding is not initialized')
            if self.embedding_projection is None:
                raise ValueError('uid+embedding fused path requested but embedding projection is not initialized')
            uid, emb_index = value
            uid_tensor = torch.tensor([int(uid)], dtype=torch.long, device=self.device)
            emb_tensor = torch.tensor([int(emb_index)], dtype=torch.long, device=self.device)
            uid_embed = self.uid_embedding(uid_tensor)
            content_embed = self.embedding_projection(self.embedding_matrix[emb_tensor].to(dtype=self.compute_dtype))
            return uid_embed + content_embed
        raise ValueError(f'Unknown spec kind: {kind}')

    def _embed_specs_with_representation_ids(self, specs):
        pieces = []
        representation_ids = []
        pending_representation = None
        for kind, value in specs:
            representation = 'model'
            if kind == 'type_marker':
                representation = value if value in self.attention_representation_to_id else 'model'
                pending_representation = representation if representation != 'model' else None
            elif pending_representation is not None:
                representation = pending_representation
                pending_representation = None
            elif kind == 'sid' and isinstance(value, tuple):
                representation = value[0]
            elif kind == 'embedding' and isinstance(value, tuple):
                representation = value[0]
            piece = self._embed_spec(kind, value)
            pieces.append(piece)
            representation_id = self.attention_representation_to_id.get(representation, 0)
            representation_ids.extend([representation_id] * piece.shape[0])
        return torch.cat(pieces, dim=0), torch.tensor(
            representation_ids, dtype=torch.long, device=self.device
        )

    def _representation_attention_mask(self, attention_mask, representation_ids):
        if not self.representation_pair_bias_enabled:
            return attention_mask
        batch_size, seq_len = attention_mask.shape
        dtype = self.compute_dtype if self.compute_dtype.is_floating_point else torch.float32
        minimum = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), minimum, dtype=dtype, device=self.device),
            diagonal=1,
        )
        additive_mask = causal.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1).clone()
        additive_mask.masked_fill_(attention_mask[:, None, None, :] == 0, minimum)
        pair_bias = self.representation_pair_bias[
            representation_ids[:, :, None], representation_ids[:, None, :]
        ].to(dtype=dtype)
        return additive_mask + pair_bias.unsqueeze(1)

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

    def _uid_hierarchy_logits(self, node_id: int, hidden_states: torch.Tensor):
        if self.uid_node_heads is None:
            raise ValueError('hierarchical uid supervision requested but uid hierarchy heads are not initialized')
        return self.uid_node_heads[node_id](hidden_states)

    def _sid_logits(self, hidden_states: torch.Tensor, representation=None):
        name = self._resolve_sid_name(representation)
        if name not in self.sid_heads:
            raise ValueError(f'sid supervision requested but head {name} is not initialized')
        return self.sid_heads[name](hidden_states)

    def _hash_logits(self, hidden_states: torch.Tensor):
        if self.hash_head is None:
            raise ValueError('hash supervision requested but hash_head is not initialized')
        return self.hash_head(hidden_states)

    def _sid_slot_allowed_codes(self, slot_index: int, representation=None):
        meta = self._sid_meta(representation)
        base_num_quantizers = int(meta['base_num_quantizers'] or 0)
        codebook_size = int(meta['codebook_size'] or 0)
        collision_offset = int(meta['collision_token_offset'] or 0)
        collision_vocab_size = int(meta['collision_vocab_size'] or 0)

        if slot_index < 0:
            raise ValueError(f'Invalid sid slot index: {slot_index}')
        if slot_index < base_num_quantizers:
            start = slot_index * codebook_size
            return list(range(start, start + codebook_size))
        if slot_index == base_num_quantizers:
            return list(range(collision_offset, collision_offset + collision_vocab_size))
        raise ValueError(
            f'SID slot index {slot_index} exceeds configured slots '
            f'(base={base_num_quantizers}, final={meta["num_quantizers"]})'
        )

    def _mask_sid_logits_for_slots(self, logits: torch.Tensor, slot_indices: torch.Tensor, representation=None):
        masked_logits = torch.full_like(logits, fill_value=torch.finfo(logits.dtype).min)
        unique_slots = sorted({int(slot) for slot in slot_indices.tolist()})
        for slot_index in unique_slots:
            row_indices = (slot_indices == slot_index).nonzero(as_tuple=False).view(-1)
            if row_indices.numel() == 0:
                continue
            allowed_codes = self._sid_slot_allowed_codes(slot_index, representation)
            allowed_tensor = torch.tensor(allowed_codes, dtype=torch.long, device=logits.device)
            masked_logits[row_indices[:, None], allowed_tensor[None, :]] = logits[row_indices[:, None], allowed_tensor[None, :]]
        return masked_logits

    def _sid_allowed_logits_for_slot(self, logits: torch.Tensor, slot_index: int, representation=None):
        allowed_codes = self._sid_slot_allowed_codes(slot_index, representation)
        allowed_tensor = torch.tensor(allowed_codes, dtype=torch.long, device=logits.device)
        allowed_start = int(allowed_codes[0])
        return logits.index_select(dim=1, index=allowed_tensor), allowed_start

    def _sid_loss_weights(self, slot_indices: torch.Tensor, representation=None):
        meta = self._sid_meta(representation)
        base_num_quantizers = int(meta['base_num_quantizers'] or 0)
        weights = torch.ones(slot_indices.shape[0], dtype=torch.float32, device=slot_indices.device)
        collision_mask = slot_indices >= base_num_quantizers
        if collision_mask.any():
            weights[collision_mask] = float(self._sid_decoding_value(
                representation,
                'collision_loss_weight',
                self.code_collision_loss_weight,
            ))
        return weights

    def _hash_slot_allowed_codes(self, slot_index: int):
        base_num_tokens = int(self.compiled.hash_base_num_tokens or 0)
        slot_sizes = list(self.compiled.hash_slot_sizes or [])
        slot_offsets = list(self.compiled.hash_slot_offsets or [])
        collision_offset = int(self.compiled.hash_collision_token_offset or 0)
        collision_vocab_size = int(self.compiled.hash_collision_vocab_size or 0)

        if slot_index < 0:
            raise ValueError(f'Invalid hash slot index: {slot_index}')
        if slot_index < base_num_tokens:
            if slot_index >= len(slot_sizes) or slot_index >= len(slot_offsets):
                raise ValueError(f'Missing hash slot metadata for slot {slot_index}')
            start = slot_offsets[slot_index]
            size = slot_sizes[slot_index]
            return list(range(start, start + size))
        if slot_index == base_num_tokens:
            return list(range(collision_offset, collision_offset + collision_vocab_size))
        raise ValueError(
            f'Hash slot index {slot_index} exceeds configured slots '
            f'(base={base_num_tokens}, final={self.compiled.hash_num_tokens})'
        )

    def _mask_hash_logits_for_slots(self, logits: torch.Tensor, slot_indices: torch.Tensor):
        masked_logits = torch.full_like(logits, fill_value=torch.finfo(logits.dtype).min)
        unique_slots = sorted({int(slot) for slot in slot_indices.tolist()})
        for slot_index in unique_slots:
            row_indices = (slot_indices == slot_index).nonzero(as_tuple=False).view(-1)
            if row_indices.numel() == 0:
                continue
            allowed_codes = self._hash_slot_allowed_codes(slot_index)
            allowed_tensor = torch.tensor(allowed_codes, dtype=torch.long, device=logits.device)
            masked_logits[row_indices[:, None], allowed_tensor[None, :]] = logits[row_indices[:, None], allowed_tensor[None, :]]
        return masked_logits

    def _hash_loss_weights(self, slot_indices: torch.Tensor):
        base_num_tokens = int(self.compiled.hash_base_num_tokens or 0)
        weights = torch.ones(slot_indices.shape[0], dtype=torch.float32, device=slot_indices.device)
        collision_mask = slot_indices >= base_num_tokens
        if collision_mask.any():
            weights[collision_mask] = self.code_collision_loss_weight
        return weights

    def _sid_decoding_mode(self, representation=None):
        name = self._resolve_sid_name(representation)
        target = self.config.compile_config.target_spec(name) or {}
        decoding = target.get('decoding') or {}
        mode = str(decoding.get('mode', getattr(self.config, 'code_decoding', 'auto'))).strip().lower()
        if mode == 'auto':
            meta = self._sid_meta(name)
            mode = str(meta.get('recommended_decoding') or 'sequential').strip().lower()
        if mode not in {'sequential', 'parallel'}:
            raise ValueError(f'Unsupported code_decoding: {mode}')
        return mode

    def _sid_decoding_value(self, representation, key, fallback):
        name = self._resolve_sid_name(representation)
        if not hasattr(self.config, 'compile_config'):
            return fallback
        target = self.config.compile_config.target_spec(name) or {}
        return (target.get('decoding') or {}).get(key, fallback)

    def _sid_beam_width(self, representation=None):
        return max(1, int(self._sid_decoding_value(
            representation, 'beam_width', self.config.code_beam_width
        )))

    def _sid_beam_chunk_size(self, representation=None):
        return max(1, int(self._sid_decoding_value(
            representation, 'beam_chunk_size',
            getattr(self.config, 'code_beam_chunk_size', getattr(self.config, 'batch_size', 1)),
        )))

    def _build_batch_inputs(self, batch):
        sample_embeddings = []
        sample_representation_ids = []
        for sample in batch:
            specs = self._build_sample_specs(sample)
            embeddings, representation_ids = self._embed_specs_with_representation_ids(specs)
            sample_embeddings.append(embeddings)
            sample_representation_ids.append(representation_ids)

        padded = pad_sequence(sample_embeddings, batch_first=True)
        padded = padded.to(dtype=self.compute_dtype)
        lengths = torch.tensor([emb.shape[0] for emb in sample_embeddings], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(padded.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        padded_representation_ids = pad_sequence(sample_representation_ids, batch_first=True)
        attention_mask = self._representation_attention_mask(
            attention_mask.long(), padded_representation_ids
        )
        return padded, attention_mask, lengths

    def ranking_ks(self):
        max_k = (
            max(1, int(self.config.multi_output_topk))
            if self.config.is_multi_task
            else max(1, int(self.config.code_beam_width))
        )
        ks = [k for k in (5, 10, 20) if k <= max_k]
        if max_k not in ks:
            ks.append(max_k)
        return sorted(set(ks))

    def sid_ranking_ks(self):
        return self.ranking_ks()

    def _build_sid_generation_batch_inputs(self, sample, sid_prefixes: list[list[int]], representation=None):
        name = self._resolve_sid_name(representation)
        sample_embeddings = []
        sample_representation_ids = []
        for sid_prefix in sid_prefixes:
            specs = self._build_sample_specs(sample)
            if sid_prefix:
                specs.append(('sid', (name, [int(code) for code in sid_prefix])))
            embeddings, representation_ids = self._embed_specs_with_representation_ids(specs)
            sample_embeddings.append(embeddings)
            sample_representation_ids.append(representation_ids)

        padded = pad_sequence(sample_embeddings, batch_first=True)
        padded = padded.to(dtype=self.compute_dtype)
        lengths = torch.tensor([emb.shape[0] for emb in sample_embeddings], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(padded.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        padded_representation_ids = pad_sequence(sample_representation_ids, batch_first=True)
        return padded, self._representation_attention_mask(
            attention_mask.long(), padded_representation_ids
        ), lengths

    def _build_sid_generation_mixed_batch_inputs(self, work_items, representation=None):
        name = self._resolve_sid_name(representation)
        sample_embeddings = []
        sample_representation_ids = []
        for sample, sid_prefix in work_items:
            specs = self._build_sample_specs(sample)
            if sid_prefix:
                specs.append(('sid', (name, [int(code) for code in sid_prefix])))
            embeddings, representation_ids = self._embed_specs_with_representation_ids(specs)
            sample_embeddings.append(embeddings)
            sample_representation_ids.append(representation_ids)

        padded = pad_sequence(sample_embeddings, batch_first=True)
        padded = padded.to(dtype=self.compute_dtype)
        lengths = torch.tensor([emb.shape[0] for emb in sample_embeddings], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(padded.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        padded_representation_ids = pad_sequence(sample_representation_ids, batch_first=True)
        return padded, self._representation_attention_mask(
            attention_mask.long(), padded_representation_ids
        ), lengths

    def _predict_sid_step_logits(self, sample, sid_prefixes: list[list[int]], slot_index: int, representation=None):
        name = self._resolve_sid_name(representation)
        chunk_size = max(1, int(getattr(self.config, 'code_beam_chunk_size', self.config.batch_size)))
        logits_chunks = []
        for start in range(0, len(sid_prefixes), chunk_size):
            chunk_prefixes = sid_prefixes[start:start + chunk_size]
            inputs_embeds, attention_mask, lengths = self._build_sid_generation_batch_inputs(
                sample, chunk_prefixes, name
            )
            hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths - 1]
            logits_chunks.append(self._sid_logits(pooled, name))
        return torch.cat(logits_chunks, dim=0)

    def _sid_kv_cache_supported(self):
        return (
            not self.representation_pair_bias_enabled
            and
            isinstance(self.encoder, (LLMSequenceEncoder, ScratchLlamaSequenceEncoder))
            and hasattr(self.encoder, 'forward_with_cache')
            and bool(getattr(self.encoder, 'forward_accepts_use_cache', False))
        )

    def _sid_base_logits_and_cache(self, sample, representation=None):
        name = self._resolve_sid_name(representation)
        inputs_embeds, attention_mask, lengths = self._build_sid_generation_batch_inputs(sample, [[]], name)
        hidden, past_key_values = self.encoder.forward_with_cache(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if past_key_values is None:
            return None
        pooled = hidden[0, lengths[0] - 1]
        return self._sid_logits(pooled, name).squeeze(0), past_key_values

    def _sid_append_cached_tokens(
            self,
            past_key_values,
            codes: list[int],
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            representation=None,
    ):
        name = self._resolve_sid_name(representation)
        past_length = self.encoder.cache_seq_length(past_key_values)
        if past_length is None:
            return None
        if not codes:
            return None
        inputs_embeds = torch.stack(
            [self._embed_spec('sid', (name, [int(code)])) for code in codes],
            dim=0,
        )
        batch_size = len(codes)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, past_length + 1), dtype=torch.long, device=self.device)
        if position_ids is None:
            position_ids = torch.full((batch_size, 1), past_length, dtype=torch.long, device=self.device)
        hidden, next_past = self.encoder.forward_with_cache(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if next_past is None:
            return None
        return self._sid_logits(hidden[:, -1, :], name), next_past

    def _sid_append_cached_token(self, past_key_values, code: int, representation=None):
        result = self._sid_append_cached_tokens(
            past_key_values, [int(code)], representation=representation
        )
        if result is None:
            return None
        logits, next_past = result
        return logits.squeeze(0), next_past

    def _sid_timing_now(self):
        if not getattr(self, 'sid_decoding_timing_enabled', False):
            return None
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _sid_timing_add(self, key: str, started_at):
        if started_at is None:
            return
        elapsed = time.perf_counter() - started_at
        timings = getattr(self, '_sid_decoding_timings', None)
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + elapsed

    @staticmethod
    def _sid_cache_summary(past_key_values):
        if past_key_values is None:
            return 'cache=None'
        cache_type = f'{type(past_key_values).__module__}.{type(past_key_values).__name__}'
        details = [f'cache_type={cache_type}']
        for method_name in ('to_legacy_cache', 'batch_select_indices', 'reorder_cache', 'get_seq_length'):
            details.append(f'{method_name}={hasattr(past_key_values, method_name)}')
        if isinstance(past_key_values, (list, tuple)):
            details.append(f'layers={len(past_key_values)}')
            if past_key_values:
                first_layer = past_key_values[0]
                details.append(f'first_layer_type={type(first_layer).__name__}')
                if isinstance(first_layer, (list, tuple)):
                    entry_shapes = [
                        str(tuple(entry.shape)) if torch.is_tensor(entry) else type(entry).__name__
                        for entry in first_layer
                    ]
                    details.append(f'first_layer_entries={entry_shapes}')
        return ' '.join(details)

    def _sid_record_kv_diagnostic(self, reason: str, past_key_values=None, exception=None):
        diagnostics = getattr(self, '_sid_kv_diagnostics', None)
        if diagnostics is None:
            return
        parts = [f'reason={reason}', self._sid_cache_summary(past_key_values)]
        encoder_diagnostic = getattr(self.encoder, 'last_cache_diagnostic', None)
        if encoder_diagnostic:
            parts.append(f'encoder_cache={encoder_diagnostic}')
        if exception is not None:
            parts.append(f'exception={type(exception).__name__}:{exception}')
        diagnostic = ' '.join(parts)
        if diagnostic not in diagnostics and len(diagnostics) < 8:
            diagnostics.append(diagnostic)

    def _sid_base_batch_logits_and_cache(self, batch, representation=None):
        name = self._resolve_sid_name(representation)
        build_started = self._sid_timing_now()
        input_items = [(sample, []) for sample in batch]
        right_padded, _, lengths = self._build_sid_generation_mixed_batch_inputs(input_items, name)
        inputs_embeds = torch.zeros_like(right_padded)
        attention_mask = torch.zeros(
            right_padded.shape[:2],
            dtype=torch.long,
            device=self.device,
        )
        for row_index, length in enumerate(lengths.tolist()):
            inputs_embeds[row_index, -length:] = right_padded[row_index, :length]
            attention_mask[row_index, -length:] = 1
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        self._sid_timing_add('input_build', build_started)
        forward_started = self._sid_timing_now()
        hidden, past_key_values = self.encoder.forward_with_cache(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        self._sid_timing_add('history_forward', forward_started)
        if past_key_values is None:
            return None
        pooled = hidden[:, -1, :]
        return self._sid_logits(pooled, name), past_key_values, attention_mask, lengths

    def _pick_sid_item(self, sid_sequence: tuple[int, ...], representation=None):
        name = self._resolve_sid_name(representation)
        candidates = self._sid_sequence_index(name).get(sid_sequence, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])
        digest = hashlib.sha256(
            f'{self.config.seed}:{"-".join(str(code) for code in sid_sequence)}'.encode('utf-8')
        ).hexdigest()
        return int(candidates[int(digest[:8], 16) % len(candidates)])

    def _beam_search_sid_items(self, sample, representation=None):
        name = self._resolve_sid_name(representation)
        beam_width = self._sid_beam_width(name)
        beams: list[tuple[tuple[int, ...], float]] = [(tuple(), 0.0)]

        meta = self._sid_meta(name)
        prefix_index = self._sid_prefix_index(name)
        for slot_index in range(int(meta['num_quantizers'])):
            sid_prefixes = [list(prefix) for prefix, _ in beams]
            step_logits = self._predict_sid_step_logits(sample, sid_prefixes, slot_index, name)
            step_log_probs = F.log_softmax(step_logits.float(), dim=-1)
            candidates = []

            for beam_index, (prefix, score) in enumerate(beams):
                allowed_codes = prefix_index.get(prefix, [])
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

    def _beam_search_sid_items_with_kv_cache(self, sample, representation=None):
        name = self._resolve_sid_name(representation)
        if not self._sid_kv_cache_supported():
            return None

        try:
            base_result = (
                self._sid_base_logits_and_cache(sample)
                if name == self._primary_sid_name()
                else self._sid_base_logits_and_cache(sample, name)
            )
        except TypeError:
            return None
        if base_result is None:
            return None
        base_logits, batched_past = base_result

        beam_width = self._sid_beam_width(name)
        num_quantizers = int(self._sid_meta(name)['num_quantizers'])
        prefix_index = self._sid_prefix_index(name)
        prefixes = [tuple()]
        scores = [0.0]
        batched_logits = base_logits.unsqueeze(0)

        for slot_index in range(num_quantizers):
            candidates = []
            batched_log_probs = F.log_softmax(batched_logits.float(), dim=-1)
            for beam_index, (prefix, score) in enumerate(zip(prefixes, scores)):
                allowed_codes = prefix_index.get(prefix, [])
                if not allowed_codes:
                    continue
                allowed_indices = torch.tensor(allowed_codes, dtype=torch.long, device=self.device)
                allowed_scores = batched_log_probs[beam_index].index_select(
                    0,
                    allowed_indices,
                )
                top_k = min(beam_width, allowed_scores.shape[0])
                top_scores, top_positions = torch.topk(allowed_scores, k=top_k)
                for top_position, top_score in zip(top_positions.tolist(), top_scores.tolist()):
                    next_code = int(allowed_codes[top_position])
                    candidates.append(
                        (prefix + (next_code,), score + float(top_score), beam_index, next_code)
                    )

            if not candidates:
                break
            candidates.sort(key=lambda item: item[1], reverse=True)
            candidates = candidates[:beam_width]

            if slot_index == num_quantizers - 1:
                return [(prefix, score) for prefix, score, _, _ in candidates]

            parent_indices = torch.tensor(
                [parent_index for _, _, parent_index, _ in candidates],
                dtype=torch.long,
                device=self.device,
            )
            selected_past = self.encoder.reorder_cache(batched_past, parent_indices)
            if selected_past is None:
                return None
            try:
                append_args = (selected_past, [next_code for _, _, _, next_code in candidates])
                next_result = (
                    self._sid_append_cached_tokens(*append_args)
                    if name == self._primary_sid_name()
                    else self._sid_append_cached_tokens(*append_args, representation=name)
                )
            except (TypeError, ValueError):
                return None
            if next_result is None:
                return None
            batched_logits, batched_past = next_result
            prefixes = [prefix for prefix, _, _, _ in candidates]
            scores = [score for _, score, _, _ in candidates]

        return list(zip(prefixes, scores))

    def _beam_search_sid_items_batch_with_kv_cache(self, batch, representation=None):
        name = self._resolve_sid_name(representation)
        if not batch or not self._sid_kv_cache_supported():
            self._sid_record_kv_diagnostic('unsupported_or_empty')
            return None
        try:
            base_result = (
                self._sid_base_batch_logits_and_cache(batch)
                if name == self._primary_sid_name()
                else self._sid_base_batch_logits_and_cache(batch, name)
            )
        except TypeError as exc:
            self._sid_record_kv_diagnostic('base_forward_type_error', exception=exc)
            return None
        if base_result is None:
            self._sid_record_kv_diagnostic('base_forward_no_cache')
            return None
        batched_logits, batched_past, batched_attention_mask, batched_lengths = base_result

        beam_width = self._sid_beam_width(name)
        num_quantizers = int(self._sid_meta(name)['num_quantizers'])
        prefix_index = self._sid_prefix_index(name)
        beams_by_sample = [[(tuple(), 0.0, sample_index)] for sample_index in range(len(batch))]

        for slot_index in range(num_quantizers):
            select_started = self._sid_timing_now()
            batched_log_probs = F.log_softmax(batched_logits.float(), dim=-1)
            candidates_by_sample = []
            for beams in beams_by_sample:
                candidates = []
                for prefix, score, cache_row in beams:
                    allowed_codes = prefix_index.get(prefix, [])
                    if not allowed_codes:
                        continue
                    allowed_indices = torch.tensor(allowed_codes, dtype=torch.long, device=self.device)
                    allowed_scores = batched_log_probs[cache_row].index_select(0, allowed_indices)
                    top_k = min(beam_width, allowed_scores.shape[0])
                    top_scores, top_positions = torch.topk(allowed_scores, k=top_k)
                    for top_position, top_score in zip(top_positions.tolist(), top_scores.tolist()):
                        next_code = int(allowed_codes[top_position])
                        candidates.append(
                            (prefix + (next_code,), score + float(top_score), cache_row, next_code)
                        )
                candidates.sort(key=lambda item: item[1], reverse=True)
                candidates_by_sample.append(candidates[:beam_width])
            self._sid_timing_add(f'slot_{slot_index}_select', select_started)

            if not any(candidates_by_sample):
                break
            if slot_index == num_quantizers - 1:
                self._sid_record_kv_diagnostic('success', batched_past)
                return [
                    [(prefix, score) for prefix, score, _, _ in candidates]
                    for candidates in candidates_by_sample
                ]

            selected = [candidate for candidates in candidates_by_sample for candidate in candidates]
            cache_started = self._sid_timing_now()
            parent_indices = torch.tensor(
                [cache_row for _, _, cache_row, _ in selected],
                dtype=torch.long,
                device=self.device,
            )
            selected_past = self.encoder.reorder_cache(batched_past, parent_indices)
            if selected_past is None:
                self._sid_record_kv_diagnostic(f'slot_{slot_index}_reorder_returned_none', batched_past)
                return None
            parent_attention_mask = batched_attention_mask.index_select(0, parent_indices)
            next_attention_mask = torch.cat(
                [
                    parent_attention_mask,
                    torch.ones((len(selected), 1), dtype=torch.long, device=self.device),
                ],
                dim=1,
            )
            parent_lengths = batched_lengths.index_select(0, parent_indices)
            next_position_ids = parent_lengths.unsqueeze(1)
            self._sid_timing_add(f'slot_{slot_index}_cache', cache_started)
            forward_started = self._sid_timing_now()
            try:
                append_kwargs = {
                    'attention_mask': next_attention_mask,
                    'position_ids': next_position_ids,
                }
                if name != self._primary_sid_name():
                    append_kwargs['representation'] = name
                next_result = self._sid_append_cached_tokens(
                    selected_past,
                    [next_code for _, _, _, next_code in selected],
                    **append_kwargs,
                )
            except (TypeError, ValueError) as exc:
                self._sid_record_kv_diagnostic(
                    f'slot_{slot_index}_append_exception',
                    selected_past,
                    exception=exc,
                )
                return None
            if next_result is None:
                self._sid_record_kv_diagnostic(f'slot_{slot_index}_append_returned_none', selected_past)
                return None
            self._sid_timing_add(f'slot_{slot_index}_forward', forward_started)
            batched_logits, batched_past = next_result
            batched_attention_mask = next_attention_mask
            batched_lengths = parent_lengths + 1

            next_beams_by_sample = []
            cache_row = 0
            for candidates in candidates_by_sample:
                next_beams = []
                for prefix, score, _, _ in candidates:
                    next_beams.append((prefix, score, cache_row))
                    cache_row += 1
                next_beams_by_sample.append(next_beams)
            beams_by_sample = next_beams_by_sample

        self._sid_record_kv_diagnostic('success', batched_past)
        return [[(prefix, score) for prefix, score, _ in beams] for beams in beams_by_sample]

    def _beam_search_sid_items_batch(self, batch, representation=None):
        name = self._resolve_sid_name(representation)
        beam_width = self._sid_beam_width(name)
        chunk_size = self._sid_beam_chunk_size(name)
        beams_by_sample: list[list[tuple[tuple[int, ...], float]]] = [[(tuple(), 0.0)] for _ in batch]

        meta = self._sid_meta(name)
        prefix_index = self._sid_prefix_index(name)
        for slot_index in range(int(meta['num_quantizers'])):
            work_items = []
            for sample_index, beams in enumerate(beams_by_sample):
                for prefix, score in beams:
                    work_items.append((sample_index, prefix, score))

            if not work_items:
                break

            candidates_by_sample: list[list[tuple[tuple[int, ...], float]]] = [[] for _ in batch]
            for start in range(0, len(work_items), chunk_size):
                chunk = work_items[start:start + chunk_size]
                input_items = [(batch[sample_index], list(prefix)) for sample_index, prefix, _ in chunk]
                inputs_embeds, attention_mask, lengths = self._build_sid_generation_mixed_batch_inputs(
                    input_items, name
                )
                hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths - 1]
                step_log_probs = F.log_softmax(self._sid_logits(pooled, name).float(), dim=-1)

                for local_index, (sample_index, prefix, score) in enumerate(chunk):
                    allowed_codes = prefix_index.get(prefix, [])
                    if not allowed_codes:
                        continue
                    allowed_indices = torch.tensor(allowed_codes, dtype=torch.long, device=self.device)
                    allowed_scores = step_log_probs[local_index, allowed_indices]
                    top_k = min(beam_width, allowed_scores.shape[0])
                    top_scores, top_positions = torch.topk(allowed_scores, k=top_k)
                    for top_position, top_score in zip(top_positions.tolist(), top_scores.tolist()):
                        next_code = int(allowed_codes[top_position])
                        candidates_by_sample[sample_index].append((prefix + (next_code,), score + float(top_score)))

            next_beams_by_sample = []
            for candidates in candidates_by_sample:
                candidates.sort(key=lambda item: item[1], reverse=True)
                next_beams_by_sample.append(candidates[:beam_width])
            beams_by_sample = next_beams_by_sample

            if not any(beams_by_sample):
                break

        return beams_by_sample

    def _decode_sid_beams_to_items(self, beams, representation=None):
        ranked_items = []
        seen_items = set()
        for sid_sequence, score in beams:
            item_uid = self._pick_sid_item(sid_sequence, representation)
            if item_uid is None or item_uid in seen_items:
                continue
            seen_items.add(item_uid)
            ranked_items.append((item_uid, float(score), sid_sequence))
        return ranked_items

    def _compute_sid_ranking_metrics(self, batch, representation=None):
        name = self._resolve_sid_name(representation)
        profile_enabled = bool(getattr(self, 'sid_decoding_timing_enabled', False))
        if profile_enabled:
            self._sid_decoding_timings = {}
            self._sid_kv_diagnostics = []
        decoding_started = self._sid_timing_now()
        ks = self.ranking_ks()
        totals = self._init_ranking_totals(ks)
        totals['beam_unique_items'] = 0.0
        totals['kv_cache_used'] = 0.0

        with torch.inference_mode():
            beam_width = self._sid_beam_width(name)
            active_beam_limit = max(
                beam_width,
                self._sid_beam_chunk_size(name),
            )
            sample_chunk_size = max(1, active_beam_limit // beam_width)
            for start in range(0, len(batch), sample_chunk_size):
                sample_chunk = batch[start:start + sample_chunk_size]
                beams_by_sample = self._beam_search_sid_items_batch_with_kv_cache(sample_chunk, name)
                used_kv_cache = beams_by_sample is not None
                if beams_by_sample is None:
                    fallback_started = self._sid_timing_now()
                    beams_by_sample = self._beam_search_sid_items_batch(sample_chunk, name)
                    self._sid_timing_add('fallback_decode', fallback_started)
                mapping_started = self._sid_timing_now()
                for sample, beams in zip(sample_chunk, beams_by_sample):
                    ranked_items = self._decode_sid_beams_to_items(beams, name)
                    ranked_uids = [uid for uid, _, _ in ranked_items]
                    totals['beam_unique_items'] += float(len(ranked_uids))
                    totals['kv_cache_used'] += float(used_kv_cache)
                    self._accumulate_ranking_metrics(totals, ks, ranked_uids, sample)
                self._sid_timing_add('item_mapping_metrics', mapping_started)

        batch_size = max(len(batch), 1)
        result = {key: value / batch_size for key, value in totals.items()}
        self._sid_timing_add('decoding_total', decoding_started)
        if profile_enabled:
            result.update({f'sid_time_{key}_ms': value * 1000.0 for key, value in self._sid_decoding_timings.items()})
            result['sid_kv_diagnostic'] = ' | '.join(self._sid_kv_diagnostics) or 'no KV-cache diagnostic recorded'
        return result

    def _multi_uid_gate(self, uid: int):
        frequency = float(self.item_target_frequencies[int(uid)].item())
        return uid_frequency_gate(
            frequency,
            mode=self.config.multi_fusion,
            uid_weight=self.config.multi_uid_weight,
            threshold=self.config.multi_frequency_threshold,
            smoothing=self.config.multi_frequency_smoothing,
        )

    def _fuse_multi_candidates(self, uid_scores: dict[int, float], sid_scores: dict[int, float]):
        frequencies = {
            int(uid): float(self.item_target_frequencies[int(uid)].item())
            for uid in set(uid_scores) | set(sid_scores)
        }
        return fuse_candidate_scores(
            uid_scores,
            sid_scores,
            frequencies,
            fusion_mode=self.config.multi_fusion,
            uid_weight=self.config.multi_uid_weight,
            score_normalization=self.config.multi_score_normalization,
            temperature_uid=self.config.multi_temperature_uid,
            temperature_sid=self.config.multi_temperature_sid,
            frequency_threshold=self.config.multi_frequency_threshold,
            frequency_smoothing=self.config.multi_frequency_smoothing,
            output_topk=self.config.multi_output_topk,
        )

    def _compute_multi_ranking_metrics(self, pooled: torch.Tensor, batch, sid_representations=None):
        if sid_representations is None or isinstance(sid_representations, str):
            sid_representations = [self._resolve_sid_name(sid_representations)]
        else:
            sid_representations = [self._resolve_sid_name(name) for name in sid_representations]
        candidate_topk = min(int(self.config.multi_candidate_topk), int(self.compiled.num_items))
        uid_logits = self._uid_logits(pooled).float()
        uid_values, uid_indices = torch.topk(uid_logits, k=candidate_topk, dim=-1)

        scores_by_representation = []
        for sid_name in sid_representations:
            if self._sid_decoding_mode(sid_name) == 'parallel':
                semantic_scores, collision_scores = self._sid_parallel_item_scores(pooled, sid_name)
                sid_values, sid_indices = torch.topk(
                    semantic_scores + collision_scores,
                    k=candidate_topk,
                    dim=-1,
                )
                per_sample = [
                    {
                        int(uid): float(score)
                        for uid, score in zip(indices.tolist(), values.tolist())
                    }
                    for indices, values in zip(sid_indices, sid_values)
                ]
            else:
                beams_by_sample = self._beam_search_sid_items_batch_with_kv_cache(batch, sid_name)
                if beams_by_sample is None:
                    beams_by_sample = self._beam_search_sid_items_batch(batch, sid_name)
                per_sample = [
                    {
                        int(uid): float(score)
                        for uid, score, _ in self._decode_sid_beams_to_items(beams, sid_name)[:candidate_topk]
                    }
                    for beams in beams_by_sample
                ]
            scores_by_representation.append(per_sample)

        sid_scores_by_sample = []
        for sample_index in range(len(batch)):
            normalized = [
                normalize_candidate_scores(
                    per_representation[sample_index],
                    self.config.multi_score_normalization,
                )
                for per_representation in scores_by_representation
            ]
            candidates = set().union(*(scores.keys() for scores in normalized))
            floors = [min(scores.values(), default=0.0) - 1.0 for scores in normalized]
            sid_scores_by_sample.append({
                uid: sum(scores.get(uid, floor) for scores, floor in zip(normalized, floors)) / len(normalized)
                for uid in candidates
            })

        ks = self.ranking_ks()
        totals = self._init_ranking_totals(ks)
        totals['multi_uid_gate_mean'] = 0.0
        totals['multi_candidates'] = 0.0
        for batch_index, (sample, per_sid) in enumerate(zip(batch, sid_scores_by_sample)):
            per_uid = {
                int(uid): float(score)
                for uid, score in zip(uid_indices[batch_index].tolist(), uid_values[batch_index].tolist())
            }
            for uid in per_sid:
                per_uid.setdefault(uid, float(uid_logits[batch_index, uid].item()))
            fused = self._fuse_multi_candidates(per_uid, per_sid)
            ranked_uids = [uid for uid, _ in fused]
            totals['multi_candidates'] += float(len(set(per_uid) | set(per_sid)))
            if ranked_uids:
                totals['multi_uid_gate_mean'] += sum(self._multi_uid_gate(uid) for uid in ranked_uids) / len(ranked_uids)
            self._accumulate_ranking_metrics(totals, ks, ranked_uids, sample)
        batch_size = max(len(batch), 1)
        return {key: value / batch_size for key, value in totals.items()}

    def _compute_sid_loss(self, batch, representation=None):
        name = self._resolve_sid_name(representation)
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        token_correct = 0.0
        seq_correct = 0.0
        token_total = 0

        for sample in batch:
            target_codes = [
                int(token_id) for token_id in function.to_list(self.compiled.item_views[name][sample['target_uid']])
            ]
            sample_preds = []
            sample_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for slot_index, label in enumerate(target_codes):
                prefix = target_codes[:slot_index]
                logits = self._predict_sid_step_logits(sample, [prefix], slot_index, name)
                slot_tensor = torch.tensor([slot_index], dtype=torch.long, device=self.device)
                masked_logits = self._mask_sid_logits_for_slots(logits, slot_tensor, name)
                label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
                token_loss = F.cross_entropy(masked_logits.float(), label_tensor, reduction='none')
                token_loss = token_loss * self._sid_loss_weights(slot_tensor, name)
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

    def _compute_sid_parallel_loss(self, pooled: torch.Tensor, batch, representation=None):
        name = self._resolve_sid_name(representation)
        sid_item_codes = self._sid_item_codes(name)
        if sid_item_codes.numel() == 0:
            raise ValueError('sid parallel supervision requires compiled sid item codes')

        target_uids = torch.tensor([int(sample['target_uid']) for sample in batch], dtype=torch.long, device=self.device)
        target_codes = sid_item_codes[target_uids]
        batch_size = target_codes.shape[0]
        num_slots = target_codes.shape[1]

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        token_correct = 0.0
        seq_predictions = []
        seq_targets = target_codes.tolist()
        logits = self._sid_logits(pooled, name)

        for slot_index in range(num_slots):
            allowed_logits, allowed_start = self._sid_allowed_logits_for_slot(logits, slot_index, name)
            slot_tensor = torch.full((batch_size,), slot_index, dtype=torch.long, device=self.device)
            labels = target_codes[:, slot_index]
            label_positions = labels - allowed_start
            slot_loss = F.cross_entropy(allowed_logits.float(), label_positions, reduction='none')
            slot_loss = slot_loss * self._sid_loss_weights(slot_tensor, name)
            total_loss = total_loss + slot_loss.mean()
            predictions = allowed_logits.argmax(dim=-1) + allowed_start
            token_correct += float((predictions == labels).float().sum().item())
            seq_predictions.append(predictions.tolist())

        predicted_sequences = list(zip(*seq_predictions)) if seq_predictions else []
        seq_correct = sum(int(list(predicted) == list(target)) for predicted, target in zip(predicted_sequences, seq_targets))
        token_total = batch_size * max(num_slots, 1)
        return total_loss / max(num_slots, 1), {
            'sid_token_acc': token_correct / max(token_total, 1),
            'sid_seq_acc': seq_correct / max(batch_size, 1),
        }

    def _compute_sid_parallel_ranking_metrics(self, pooled: torch.Tensor, batch, representation=None):
        name = self._resolve_sid_name(representation)
        sid_item_codes = self._sid_item_codes(name)
        if sid_item_codes.numel() == 0:
            raise ValueError('sid parallel ranking requires compiled sid item codes')

        batch_size = pooled.shape[0]
        semantic_scores, collision_scores = self._sid_parallel_item_scores(pooled, name)
        item_codes = sid_item_codes.to(device=self.device)

        ks = self.ranking_ks()
        totals = self._init_ranking_totals(ks)
        item_indices = torch.arange(item_codes.shape[0], dtype=torch.long, device=self.device)

        for batch_index, sample in enumerate(batch):
            self._accumulate_ranking_metrics_from_score_tensors(
                totals,
                ks,
                semantic_scores[batch_index],
                collision_scores[batch_index],
                sample,
                item_indices=item_indices,
            )

        return {key: value / max(batch_size, 1) for key, value in totals.items()}

    def _sid_parallel_item_scores(self, pooled: torch.Tensor, representation=None):
        name = self._resolve_sid_name(representation)
        sid_item_codes = self._sid_item_codes(name)
        if sid_item_codes.numel() == 0:
            raise ValueError('sid parallel item scoring requires compiled sid item codes')
        batch_size = pooled.shape[0]
        item_codes = sid_item_codes.to(device=self.device)
        semantic_scores = torch.zeros((batch_size, item_codes.shape[0]), dtype=torch.float32, device=self.device)
        collision_scores = torch.zeros((batch_size, item_codes.shape[0]), dtype=torch.float32, device=self.device)
        base_num_quantizers = int(self._sid_meta(name)['base_num_quantizers'] or 0)
        logits = self._sid_logits(pooled, name)

        for slot_index in range(int(item_codes.shape[1])):
            allowed_logits, allowed_start = self._sid_allowed_logits_for_slot(logits, slot_index, name)
            log_probs = F.log_softmax(allowed_logits.float(), dim=-1)
            slot_codes = item_codes[:, slot_index]
            slot_positions = slot_codes - allowed_start
            slot_scores = log_probs.index_select(dim=1, index=slot_positions)
            if slot_index < base_num_quantizers:
                semantic_scores = semantic_scores + slot_scores
            else:
                collision_scores = collision_scores + slot_scores

        return semantic_scores, collision_scores

    def _compute_ranking_metrics_from_logits(self, logits: torch.Tensor, target_indices: torch.Tensor, batch=None):
        ks = self.ranking_ks()
        totals = self._init_ranking_totals(ks)

        ranking = torch.argsort(logits.float(), dim=-1, descending=True)
        if batch is None:
            targets = target_indices.view(-1, 1)
            match_positions = (ranking == targets).nonzero(as_tuple=False)
            if match_positions.shape[0] != logits.shape[0]:
                raise RuntimeError('failed to recover target rank from logits')
            ranks = match_positions[:, 1] + 1

            for k in ks:
                hits = (ranks <= k).float()
                totals[f'hr@{k}'] = float(hits.mean().item())
                totals[f'pass@{k}'] = totals[f'hr@{k}']
                totals[f'recall@{k}'] = totals[f'hr@{k}']
                ndcg = hits / torch.log2(ranks.float() + 1)
                totals[f'ndcg@{k}'] = float(ndcg.mean().item())
            totals['mrr'] = float((1.0 / ranks.float()).mean().item())
            return totals

        for batch_index, sample in enumerate(batch):
            ranked_uids = [int(uid) for uid in ranking[batch_index].tolist()]
            self._accumulate_ranking_metrics(totals, ks, ranked_uids, sample)

        batch_size = max(len(batch), 1)
        for key in totals:
            totals[key] /= batch_size
        return totals

    @staticmethod
    def _sample_ground_truth_uids(sample):
        ground_truth_uids = sample.get('ground_truth_uids') or []
        if ground_truth_uids:
            return [int(uid) for uid in ground_truth_uids]
        return [int(sample['target_uid'])]

    @staticmethod
    def _init_ranking_totals(ks):
        totals = {f'hr@{k}': 0.0 for k in ks}
        totals.update({f'pass@{k}': 0.0 for k in ks})
        totals.update({f'recall@{k}': 0.0 for k in ks})
        totals.update({f'ndcg@{k}': 0.0 for k in ks})
        totals['mrr'] = 0.0
        return totals

    def _accumulate_ranking_metrics(self, totals, ks, ranked_uids, sample):
        candidate_uids = list(dict.fromkeys(self._sample_ground_truth_uids(sample)))
        if not candidate_uids:
            return

        rank_by_uid = {}
        for index, uid in enumerate(ranked_uids, start=1):
            uid = int(uid)
            if uid not in rank_by_uid:
                rank_by_uid[uid] = index

        self._record_target_rank(sample, rank_by_uid.get(int(sample['target_uid'])))

        matched_ranks = sorted(rank_by_uid[uid] for uid in candidate_uids if uid in rank_by_uid)
        if matched_ranks:
            totals['mrr'] += 1.0 / matched_ranks[0]

        for k in ks:
            hit_count = sum(rank <= k for rank in matched_ranks)
            hit_ratio = 1.0 if hit_count > 0 else 0.0
            totals[f'hr@{k}'] += hit_ratio
            totals[f'pass@{k}'] += hit_ratio
            totals[f'recall@{k}'] += hit_count / len(candidate_uids)
            if hit_count > 0:
                dcg = sum(1.0 / math.log2(rank + 1) for rank in matched_ranks if rank <= k)
                ideal_hits = min(len(candidate_uids), k)
                idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
                totals[f'ndcg@{k}'] += dcg / idcg if idcg > 0 else 0.0

    def _accumulate_ranking_metrics_from_score_tensors(
        self,
        totals,
        ks,
        semantic_scores,
        collision_scores,
        sample,
        *,
        item_indices=None,
    ):
        candidate_uids = list(dict.fromkeys(self._sample_ground_truth_uids(sample)))
        if not candidate_uids:
            return

        num_items = int(semantic_scores.shape[0])
        if item_indices is None:
            item_indices = torch.arange(num_items, dtype=torch.long, device=semantic_scores.device)
        matched_ranks = []
        primary_target_rank = None
        for uid in candidate_uids:
            uid = int(uid)
            if uid < 0 or uid >= num_items:
                continue
            target_semantic = semantic_scores[uid]
            target_collision = collision_scores[uid]
            higher = (
                (semantic_scores > target_semantic)
                | ((semantic_scores == target_semantic) & (collision_scores > target_collision))
                | (
                    (semantic_scores == target_semantic)
                    & (collision_scores == target_collision)
                    & (item_indices < uid)
                )
            )
            rank = int(higher.sum().item()) + 1
            matched_ranks.append(rank)
            if uid == int(sample['target_uid']):
                primary_target_rank = rank

        self._record_target_rank(sample, primary_target_rank)

        matched_ranks = sorted(matched_ranks)
        if matched_ranks:
            totals['mrr'] += 1.0 / matched_ranks[0]

        for k in ks:
            hit_count = sum(rank <= k for rank in matched_ranks)
            hit_ratio = 1.0 if hit_count > 0 else 0.0
            totals[f'hr@{k}'] += hit_ratio
            totals[f'pass@{k}'] += hit_ratio
            totals[f'recall@{k}'] += hit_count / len(candidate_uids)
            if hit_count > 0:
                dcg = sum(1.0 / math.log2(rank + 1) for rank in matched_ranks if rank <= k)
                ideal_hits = min(len(candidate_uids), k)
                idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
                totals[f'ndcg@{k}'] += dcg / idcg if idcg > 0 else 0.0

    def _target_token_values(self, target_uid: int):
        if self.config.is_multi_task:
            raise ValueError('multi-task supervision builds each target representation independently')
        if self.config.task_type == 'embedding':
            raise ValueError('embedding task uses query-anchor supervision instead of target tokens')
        if self.config.task_type == 'uid':
            return [int(self.compiled.item_views['uid'][target_uid])]
        if self.config.task_type == 'hash':
            return [int(token_id) for token_id in function.to_list(self.compiled.item_views['hash'][target_uid])]
        sid_name = self._primary_sid_name()
        sid_values = [int(token_id) for token_id in function.to_list(self.compiled.item_views[sid_name][target_uid])]
        expected = int(self._sid_meta(sid_name)['num_quantizers'])
        if expected and len(sid_values) != expected:
            raise ValueError(
                f'sid target length mismatch for uid={target_uid}: '
                f'expected {expected} tokens, got {len(sid_values)}'
            )
        return sid_values

    def _target_embedding_index(self, target_uid: int, representation='embedding'):
        return int(self.compiled.item_views[representation][target_uid])

    def _repr_payload_labels(self, repr_type: str, uid: int):
        kind = self.config.compile_config.representation_kind(repr_type)
        if kind == 'uid':
            return [int(self.compiled.item_views[repr_type][uid])]
        if kind in {'sid', 'hash', 'text'}:
            return [int(token_id) for token_id in function.to_list(self.compiled.item_views[repr_type][uid])]
        if kind == 'embedding':
            return [self._target_embedding_index(uid, repr_type)]
        raise ValueError(f'Unsupported repr type for supervision: {repr_type}')

    def _build_repr_supervision(self, repr_type: str, uid: int, marker_position: int, payload_positions: list[int], group: str):
        repr_kind = self.config.compile_config.representation_kind(repr_type)
        labels = self._repr_payload_labels(repr_type, uid)
        if not labels:
            return []
        if repr_kind == 'embedding':
            return [{'kind': repr_kind, 'representation': repr_type, 'position': int(marker_position), 'label': int(labels[0]), 'group': group, 'slot': -1}]
        if repr_kind == 'sid':
            if self._sid_decoding_mode(repr_type) == 'parallel':
                return [
                    {'kind': repr_kind, 'representation': repr_type, 'position': int(marker_position), 'label': int(label), 'group': group, 'slot': slot_index}
                    for slot_index, label in enumerate(labels)
                ]
            anchor_positions = [int(marker_position)] + [int(position) for position in payload_positions[:-1]]
            return [
                {'kind': repr_kind, 'representation': repr_type, 'position': anchor_position, 'label': int(label), 'group': group, 'slot': slot_index}
                for slot_index, (anchor_position, label) in enumerate(zip(anchor_positions, labels))
            ]
        if repr_kind == 'hash':
            return [
                {'kind': repr_kind, 'representation': repr_type, 'position': int(marker_position), 'label': int(label), 'group': group, 'slot': slot_index}
                for slot_index, label in enumerate(labels)
            ]
        anchor_positions = [int(marker_position)] + [int(position) for position in payload_positions[:-1]]
        return [
            {'kind': repr_kind, 'representation': repr_type, 'position': anchor_position, 'label': int(label), 'group': group, 'slot': -1}
            for anchor_position, label in zip(anchor_positions, labels)
        ]

    def _build_finetune_sample_inputs_add(self, sample):
        if self.config.task_type != 'uid':
            raise ValueError('repr.combine=add is only supported for uid targets')

        separator_ids = [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]
        sequence_uids = sample['sequence_uids']
        embeddings = []
        representation_ids = []
        supervision = []

        def append_embedded(tensor: torch.Tensor, representation='model'):
            embeddings.append(tensor)
            representation_ids.extend([
                self.attention_representation_to_id.get(representation, 0)
            ] * tensor.shape[0])
            return sum(piece.shape[0] for piece in embeddings[:-1]), sum(piece.shape[0] for piece in embeddings) - 1

        for item_index, uid in enumerate(sequence_uids):
            if item_index > 0 and separator_ids:
                append_embedded(self._embed_spec('model_tokens', separator_ids))

            if item_index == 0:
                for kind, value in self._render_history_item(uid):
                    append_embedded(self._embed_spec(kind, value))
                continue

            marker_position = None
            payload_positions = []
            for kind, value in self._render_single_view_item(uid, 'uid'):
                start, end = append_embedded(self._embed_spec(kind, value))
                if kind == 'type_marker':
                    marker_position = start
                else:
                    payload_positions.extend(range(start, end + 1))
            supervision.extend(
                self._build_repr_supervision(
                    repr_type='uid',
                    uid=uid,
                    marker_position=marker_position,
                    payload_positions=payload_positions,
                    group='primary',
                )
            )

            if item_index < len(sequence_uids) - 1:
                for kind, value in self._render_history_item(uid):
                    append_embedded(self._embed_spec(kind, value))

        sample_embeddings = torch.cat(embeddings, dim=0)
        return {
            'inputs_embeds': sample_embeddings,
            'representation_ids': torch.tensor(representation_ids, dtype=torch.long, device=self.device),
            'supervision': supervision,
        }

    def _build_finetune_sample_inputs(self, sample):
        if self.config.repr_combine == 'add':
            return self._build_finetune_sample_inputs_add(sample)
        if self.config.is_multi_task:
            return self._build_finetune_sample_inputs_multi(sample)
        separator_ids = [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]
        sequence_uids = sample['sequence_uids']
        embeddings = []
        representation_ids = []
        supervision = []

        def append_embedded(tensor: torch.Tensor, representation='model'):
            start = sum(piece.shape[0] for piece in embeddings)
            embeddings.append(tensor)
            representation_ids.extend([
                self.attention_representation_to_id.get(representation, 0)
            ] * tensor.shape[0])
            return start, start + tensor.shape[0] - 1

        for item_index, uid in enumerate(sequence_uids):
            if item_index > 0 and separator_ids:
                append_embedded(self._embed_spec('model_tokens', separator_ids))

            include_alignment_repr = item_index < len(sequence_uids) - 1
            repr_types = list(self.config.compile_config.target_names)
            if include_alignment_repr:
                repr_types.extend([
                    repr_type
                    for repr_type in self.config.compile_config.representation_names[len(self.config.compile_config.target_names):]
                ])

            for repr_type in repr_types:
                segment_specs = self._render_single_view_item(uid, repr_type)
                marker_position = None
                payload_positions = []
                for kind, value in segment_specs:
                    start, end = append_embedded(self._embed_spec(kind, value), repr_type)
                    if kind == 'type_marker':
                        marker_position = start
                    else:
                        payload_positions.extend(range(start, end + 1))

                if repr_type in self.config.compile_config.target_names and item_index > 0:
                    supervision.extend(
                        self._build_repr_supervision(
                            repr_type=repr_type,
                            uid=uid,
                            marker_position=marker_position,
                            payload_positions=payload_positions,
                            group='primary',
                        )
                    )
                elif repr_type not in self.config.compile_config.target_names and self.config.alignment_weight > 0 and include_alignment_repr:
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
            'representation_ids': torch.tensor(representation_ids, dtype=torch.long, device=self.device),
            'supervision': supervision,
        }

    def _build_finetune_sample_inputs_multi(self, sample):
        separator_ids = [int(token_id) for token_id in self.compiled.prompt_main['item_separator_ids']]
        embeddings = []
        representation_ids = []
        supervision = []

        def append_embedded(tensor: torch.Tensor, representation='model'):
            start = sum(piece.shape[0] for piece in embeddings)
            embeddings.append(tensor)
            representation_ids.extend([
                self.attention_representation_to_id.get(representation, 0)
            ] * tensor.shape[0])
            return start, start + tensor.shape[0] - 1

        for item_index, uid in enumerate(sample['sequence_uids']):
            if item_index > 0 and separator_ids:
                append_embedded(self._embed_spec('model_tokens', separator_ids))

            marker_position, _ = append_embedded(
                self._embed_spec('type_marker', self._target_marker_name())
            )
            payload_positions = {}
            for task_type in self.config.compile_config.target_names:
                positions = []
                for kind, value in self._render_single_view_item(uid, task_type):
                    if kind == 'type_marker':
                        continue
                    start, end = append_embedded(self._embed_spec(kind, value), task_type)
                    positions.extend(range(start, end + 1))
                payload_positions[task_type] = positions

            if item_index > 0:
                for task_type in self.config.compile_config.target_names:
                    supervision.extend(
                        self._build_repr_supervision(
                            repr_type=task_type,
                            uid=uid,
                            marker_position=marker_position,
                            payload_positions=payload_positions[task_type],
                            group='primary',
                        )
                    )

            if item_index < len(sample['sequence_uids']) - 1:
                for repr_type in self.config.compile_config.representation_names[len(self.config.compile_config.target_names):]:
                    segment_specs = self._render_single_view_item(uid, repr_type)
                    alignment_marker = None
                    alignment_positions = []
                    for kind, value in segment_specs:
                        start, end = append_embedded(self._embed_spec(kind, value), repr_type)
                        if kind == 'type_marker':
                            alignment_marker = start
                        else:
                            alignment_positions.extend(range(start, end + 1))
                    if self.config.alignment_weight > 0:
                        supervision.extend(
                            self._build_repr_supervision(
                                repr_type=repr_type,
                                uid=uid,
                                marker_position=alignment_marker,
                                payload_positions=alignment_positions,
                                group='alignment',
                            )
                        )

        return {
            'inputs_embeds': torch.cat(embeddings, dim=0),
            'representation_ids': torch.tensor(representation_ids, dtype=torch.long, device=self.device),
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
        padded_representation_ids = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        supervision_batch_indices = []
        supervision_positions = []
        supervision_kinds = []
        supervision_representations = []
        supervision_labels = []
        supervision_groups = []
        supervision_slots = []

        for batch_index, sample in enumerate(packed_samples):
            seq_len = sample['inputs_embeds'].shape[0]
            padded_embeds[batch_index, :seq_len] = sample['inputs_embeds']
            padded_representation_ids[batch_index, :seq_len] = sample['representation_ids']
            for entry in sample['supervision']:
                supervision_batch_indices.append(batch_index)
                supervision_positions.append(int(entry['position']))
                supervision_kinds.append(entry['kind'])
                supervision_representations.append(entry.get('representation', entry['kind']))
                supervision_labels.append(int(entry['label']))
                supervision_groups.append(entry['group'])
                supervision_slots.append(int(entry.get('slot', -1)))

        return (
            padded_embeds,
            self._representation_attention_mask(
                attention_mask.long(), padded_representation_ids
            ),
            lengths,
            {
                'batch_indices': torch.tensor(supervision_batch_indices, dtype=torch.long, device=self.device),
                'positions': torch.tensor(supervision_positions, dtype=torch.long, device=self.device),
                'kinds': supervision_kinds,
                'representations': supervision_representations,
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

    def _compute_uid_hierarchical_losses(self, hidden_states: torch.Tensor, target_uids: torch.Tensor):
        if self.uid_hierarchy is None:
            raise ValueError('hierarchical uid losses requested without uid hierarchy artifacts')

        item_node_ids = self.uid_hierarchy.item_node_ids.to(device=self.device)[target_uids]
        item_labels = self.uid_hierarchy.item_labels.to(device=self.device)[target_uids]
        batch_size = hidden_states.shape[0]
        depth = self.uid_hierarchy.depth

        losses_per_sample = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        path_correct = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        metrics = {}

        for level_index in range(depth):
            node_ids = item_node_ids[:, level_index]
            labels = item_labels[:, level_index]
            level_predictions = torch.full_like(labels, fill_value=-1)
            unique_node_ids = sorted({int(node_id) for node_id in node_ids.tolist()})
            for node_id in unique_node_ids:
                sample_indices = (node_ids == node_id).nonzero(as_tuple=False).view(-1)
                if sample_indices.numel() == 0:
                    continue
                logits = self._uid_hierarchy_logits(node_id, hidden_states[sample_indices])
                level_losses = F.cross_entropy(logits.float(), labels[sample_indices], reduction='none')
                losses_per_sample[sample_indices] += level_losses
                predictions = logits.argmax(dim=-1)
                level_predictions[sample_indices] = predictions

            level_correct = level_predictions == labels
            path_correct = path_correct & level_correct
            if level_index == 0:
                metrics['uid_cluster_acc'] = float(level_correct.float().mean().item())
            if level_index == depth - 1:
                metrics['uid_local_acc'] = float(level_correct.float().mean().item())

        metrics['uid_acc'] = float(path_correct.float().mean().item())
        return losses_per_sample / max(depth, 1), metrics

    def _compute_uid_hierarchical_loss(self, pooled: torch.Tensor, batch):
        target_uids = torch.tensor([int(sample['target_uid']) for sample in batch], dtype=torch.long, device=self.device)
        losses, metrics = self._compute_uid_hierarchical_losses(pooled, target_uids)
        return losses.mean(), metrics

    def _compute_uid_hierarchical_ranking_metrics(self, pooled: torch.Tensor, batch):
        if self.uid_hierarchy is None:
            raise ValueError('hierarchical uid ranking requested without uid hierarchy artifacts')

        ks = self.ranking_ks()
        totals = self._init_ranking_totals(ks)

        for batch_index, sample in enumerate(batch):
            hidden = pooled[batch_index:batch_index + 1]
            frontier = [(0, 0.0)]
            candidate_scores = {}
            for level_index in range(self.uid_hierarchy.depth):
                next_frontier = []
                for node_id, prefix_score in frontier:
                    logits = self._uid_hierarchy_logits(node_id, hidden).squeeze(0)
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    child_count = int(self.uid_hierarchy.node_child_counts[node_id])
                    topk_spec = self.uid_hierarchy.topk_per_level[level_index]
                    topk = child_count if topk_spec <= 0 else min(topk_spec, child_count)
                    top_scores, top_labels = torch.topk(log_probs, k=topk)
                    if level_index < self.uid_hierarchy.depth - 1:
                        child_nodes = self.uid_hierarchy.child_nodes[node_id]
                        for label, score in zip(top_labels.tolist(), top_scores.tolist()):
                            next_frontier.append((int(child_nodes[int(label)]), prefix_score + float(score)))
                    else:
                        leaf_items = self.uid_hierarchy.leaf_items[node_id]
                        for label, score in zip(top_labels.tolist(), top_scores.tolist()):
                            item_uid = int(leaf_items[int(label)])
                            candidate_scores[item_uid] = max(candidate_scores.get(item_uid, float('-inf')), prefix_score + float(score))
                frontier = next_frontier

            ranked_uids = [
                item_uid
                for item_uid, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            ]
            self._accumulate_ranking_metrics(totals, ks, ranked_uids, sample)

        batch_size = max(len(batch), 1)
        return {key: value / batch_size for key, value in totals.items()}

    def _embedding_decoder_components(self):
        if self.config.compile_config.representation_graph:
            name = self.config.compile_config.primary_name('embedding', targets=True)
            if not name:
                raise ValueError('embedding decoder target is not configured')
            return name, self.embedding_heads[name], self.embedding_tables[name].weight
        return 'embedding', self.embedding_head, self.embedding_matrix

    def _compute_embedding_loss(self, pooled: torch.Tensor, batch):
        name, head, table = self._embedding_decoder_components()
        target_indices = torch.tensor(
            [int(self.compiled.item_views[name][sample['target_uid']]) for sample in batch],
            dtype=torch.long,
            device=self.device,
        )
        targets = table[target_indices].to(dtype=self.compute_dtype)
        predictions = head(pooled)
        norm_predictions = F.normalize(predictions.float(), dim=-1)
        norm_table = F.normalize(table.float(), dim=-1)
        logits = norm_predictions @ norm_table.T
        loss = F.cross_entropy(logits, target_indices)
        cosine = F.cosine_similarity(predictions.float(), targets.float(), dim=-1).mean()
        accuracy = (logits.argmax(dim=-1) == target_indices).float().mean()
        return loss, {
            'embedding_cosine': cosine.item(),
            'embedding_acc': accuracy.item(),
        }

    def _compute_hash_parallel_loss(self, pooled: torch.Tensor, batch):
        if self.hash_item_codes.numel() == 0:
            raise ValueError('hash supervision requires compiled hash item codes')

        target_uids = torch.tensor([int(sample['target_uid']) for sample in batch], dtype=torch.long, device=self.device)
        target_codes = self.hash_item_codes[target_uids]
        batch_size = target_codes.shape[0]
        num_slots = target_codes.shape[1]

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        token_correct = 0.0
        seq_predictions = []
        seq_targets = target_codes.tolist()

        for slot_index in range(num_slots):
            logits = self._hash_logits(pooled)
            slot_tensor = torch.full((batch_size,), slot_index, dtype=torch.long, device=self.device)
            masked_logits = self._mask_hash_logits_for_slots(logits, slot_tensor)
            labels = target_codes[:, slot_index]
            slot_loss = F.cross_entropy(masked_logits.float(), labels, reduction='none')
            slot_loss = slot_loss * self._hash_loss_weights(slot_tensor)
            total_loss = total_loss + slot_loss.mean()
            predictions = masked_logits.argmax(dim=-1)
            token_correct += float((predictions == labels).float().sum().item())
            seq_predictions.append(predictions.tolist())

        predicted_sequences = list(zip(*seq_predictions)) if seq_predictions else []
        seq_correct = sum(int(list(predicted) == list(target)) for predicted, target in zip(predicted_sequences, seq_targets))
        token_total = batch_size * max(num_slots, 1)
        return total_loss / max(num_slots, 1), {
            'hash_token_acc': token_correct / max(token_total, 1),
            'hash_seq_acc': seq_correct / max(batch_size, 1),
        }

    def _compute_hash_parallel_ranking_metrics(self, pooled: torch.Tensor, batch):
        if self.hash_item_codes.numel() == 0:
            raise ValueError('hash parallel ranking requires compiled hash item codes')

        batch_size = pooled.shape[0]
        item_codes = self.hash_item_codes.to(device=self.device)
        semantic_scores = torch.zeros((batch_size, item_codes.shape[0]), dtype=torch.float32, device=self.device)
        collision_scores = torch.zeros((batch_size, item_codes.shape[0]), dtype=torch.float32, device=self.device)
        base_num_tokens = int(self.compiled.hash_base_num_tokens or 0)

        for slot_index in range(int(item_codes.shape[1])):
            logits = self._hash_logits(pooled)
            slot_tensor = torch.full((batch_size,), slot_index, dtype=torch.long, device=self.device)
            masked_logits = self._mask_hash_logits_for_slots(logits, slot_tensor)
            log_probs = F.log_softmax(masked_logits.float(), dim=-1)
            slot_codes = item_codes[:, slot_index]
            slot_scores = log_probs.index_select(dim=1, index=slot_codes)
            if slot_index < base_num_tokens:
                semantic_scores = semantic_scores + slot_scores
            else:
                collision_scores = collision_scores + slot_scores

        ks = self.ranking_ks()
        totals = self._init_ranking_totals(ks)

        semantic_scores_cpu = semantic_scores.detach().cpu()
        collision_scores_cpu = collision_scores.detach().cpu()
        for batch_index, sample in enumerate(batch):
            ranked_uids = sorted(
                range(item_codes.shape[0]),
                key=lambda uid: (
                    float(semantic_scores_cpu[batch_index, uid].item()),
                    float(collision_scores_cpu[batch_index, uid].item()),
                ),
                reverse=True,
            )
            self._accumulate_ranking_metrics(totals, ks, ranked_uids, sample)

        return {key: value / max(batch_size, 1) for key, value in totals.items()}

    def _compute_mixed_supervision_loss(self, selected_hidden: torch.Tensor, supervision: dict):
        kinds = supervision['kinds']
        representations = supervision.get('representations', kinds)
        labels = supervision['labels']
        groups = supervision['groups']
        slots = supervision['slots']
        primary_losses = {}
        alignment_losses = []
        metrics = {}

        for representation in dict.fromkeys(representations):
            kind = self.config.compile_config.representation_kind(representation)
            mask_indices = [index for index, entry in enumerate(representations) if entry == representation]
            if not mask_indices:
                continue
            index_tensor = torch.tensor(mask_indices, dtype=torch.long, device=self.device)
            kind_hidden = selected_hidden[index_tensor]
            kind_labels = labels[index_tensor]
            kind_slots = slots[index_tensor]
            group_mask = [groups[index] for index in mask_indices]

            if kind == 'uid':
                if self.uid_hierarchical_decoding and self.config.task_type == 'uid':
                    losses, uid_metrics = self._compute_uid_hierarchical_losses(kind_hidden, kind_labels)
                    metrics.update(uid_metrics)
                else:
                    logits = self._uid_logits(kind_hidden)
                    losses = F.cross_entropy(logits.float(), kind_labels, reduction='none')
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == kind_labels).float().mean().item()
                    metrics['uid_acc'] = accuracy
            elif kind == 'sid':
                logits = self._sid_logits(kind_hidden, representation)
                masked_logits = self._mask_sid_logits_for_slots(logits, kind_slots, representation)
                losses = F.cross_entropy(masked_logits.float(), kind_labels, reduction='none')
                losses = losses * self._sid_loss_weights(kind_slots, representation)
                predictions = masked_logits.argmax(dim=-1)
                accuracy = (predictions == kind_labels).float().mean().item()
                sid_names = self.config.compile_config.names_for_kind('sid')
                metric_name = 'sid_token_acc' if len(sid_names) == 1 else f'{representation}_token_acc'
                metrics[metric_name] = accuracy
            elif kind == 'hash':
                logits = self._hash_logits(kind_hidden)
                masked_logits = self._mask_hash_logits_for_slots(logits, kind_slots)
                losses = F.cross_entropy(masked_logits.float(), kind_labels, reduction='none')
                losses = losses * self._hash_loss_weights(kind_slots)
                predictions = masked_logits.argmax(dim=-1)
                accuracy = (predictions == kind_labels).float().mean().item()
                metrics['hash_token_acc'] = accuracy
            elif kind == 'text':
                logits = self._text_logits(kind_hidden)
                losses = F.cross_entropy(logits, kind_labels, reduction='none')
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == kind_labels).float().mean().item()
                metrics['text_token_acc'] = accuracy
            else:
                if self.config.compile_config.representation_graph:
                    head = self.embedding_heads[representation]
                    table = self.embedding_tables[representation].weight
                else:
                    head = self.embedding_head
                    table = self.embedding_matrix
                predictions = head(kind_hidden)
                targets = table[kind_labels].to(dtype=self.compute_dtype)
                norm_predictions = F.normalize(predictions.float(), dim=-1)
                norm_table = F.normalize(table.float(), dim=-1)
                logits = norm_predictions @ norm_table.T
                losses = F.cross_entropy(logits, kind_labels, reduction='none')
                accuracy = (logits.argmax(dim=-1) == kind_labels).float().mean().item()
                cosine = F.cosine_similarity(predictions.float(), targets.float(), dim=-1).mean().item()
                metric_prefix = representation if self.config.compile_config.representation_graph else 'embedding'
                metrics[f'{metric_prefix}_acc'] = accuracy
                metrics[f'{metric_prefix}_cosine'] = cosine

            for local_index, loss_value in enumerate(losses):
                if group_mask[local_index] == 'primary':
                    primary_losses.setdefault(representation, []).append(loss_value)
                else:
                    alignment_losses.append(loss_value)

        if not any(primary_losses.values()):
            raise RuntimeError('no primary supervision entries were constructed for finetune batch')

        named_sid_accuracy = [
            value for key, value in metrics.items()
            if key.endswith('_token_acc') and key != 'sid_token_acc'
        ]
        if named_sid_accuracy:
            metrics['sid_token_acc'] = sum(named_sid_accuracy) / len(named_sid_accuracy)

        task_loss_weights = {
            'uid': float(getattr(self.config, 'multi_uid_loss_weight', 1.0)),
            'sid': float(getattr(self.config, 'multi_sid_loss_weight', 1.0)),
        }
        primary_parts = []
        primary_parts_by_kind = {}
        target_kinds = [
            self.config.compile_config.representation_kind(name)
            for name in self.config.compile_config.target_names
        ]
        for representation, losses_for_kind in primary_losses.items():
            kind = self.config.compile_config.representation_kind(representation)
            kind_loss = torch.stack(losses_for_kind).mean()
            weight = task_loss_weights.get(kind, 1.0) if self.config.is_multi_task else 1.0
            weighted_loss = kind_loss * weight
            primary_parts.append(weighted_loss)
            primary_parts_by_kind.setdefault(kind, []).append(weighted_loss)
            if self.config.is_multi_task:
                metric_prefix = representation if target_kinds.count(kind) > 1 else kind
                metrics[f'{metric_prefix}_loss'] = float(kind_loss.item())
        primary_loss = (
            torch.stack([
                torch.stack(parts).mean()
                for parts in primary_parts_by_kind.values()
            ]).sum()
            if self.config.is_multi_task
            else primary_parts[0]
        )
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

        if self.config.is_multi_task:
            target_names = self.config.compile_config.target_names
            target_kinds = [
                self.config.compile_config.representation_kind(name)
                for name in target_names
            ]
            if all(kind == 'sid' for kind in target_kinds):
                losses = []
                per_target_metrics = []
                metrics = {}
                for name in target_names:
                    if self._sid_decoding_mode(name) == 'parallel':
                        sid_loss, sid_metrics = self._compute_sid_parallel_loss(pooled, batch, name)
                        sid_metrics.update(self._compute_sid_parallel_ranking_metrics(pooled, batch, name))
                    else:
                        sid_loss, sid_metrics = self._compute_sid_loss(batch, name)
                        sid_metrics.update(self._compute_sid_ranking_metrics(batch, name))
                    losses.append(sid_loss)
                    per_target_metrics.append(sid_metrics)
                    metrics.update({f'{name}_{key}': value for key, value in sid_metrics.items()})
                common_numeric_keys = set.intersection(*[
                    {key for key, value in values.items() if isinstance(value, (int, float))}
                    for values in per_target_metrics
                ])
                for key in common_numeric_keys:
                    metrics[key] = sum(float(values[key]) for values in per_target_metrics) / len(per_target_metrics)
                return torch.stack(losses).mean(), metrics

            sid_targets = [name for name, kind in zip(target_names, target_kinds) if kind == 'sid']
            if target_kinds.count('uid') != 1 or not sid_targets:
                raise ValueError(f'unsupported multi-target decoder kinds: {target_kinds}')
            uid_loss, uid_metrics = self._compute_uid_loss(pooled, batch)
            sid_losses = []
            sid_metrics = {}
            for name in sid_targets:
                if self._sid_decoding_mode(name) == 'parallel':
                    per_loss, per_metrics = self._compute_sid_parallel_loss(pooled, batch, name)
                else:
                    per_loss, per_metrics = self._compute_sid_loss(batch, name)
                sid_losses.append(per_loss)
                sid_metrics.update({f'{name}_{key}': value for key, value in per_metrics.items()})
            sid_loss = torch.stack(sid_losses).mean()
            loss = (
                float(self.config.multi_uid_loss_weight) * uid_loss
                + float(self.config.multi_sid_loss_weight) * sid_loss
            )
            metrics = dict(uid_metrics)
            metrics.update(sid_metrics)
            metrics.update(self._compute_multi_ranking_metrics(pooled, batch, sid_targets))
            return loss, metrics

        if self.config.task_type == 'uid':
            if self.uid_hierarchical_decoding:
                loss, metrics = self._compute_uid_hierarchical_loss(pooled, batch)
                metrics.update(self._compute_uid_hierarchical_ranking_metrics(pooled, batch))
            else:
                loss, metrics = self._compute_uid_loss(pooled, batch)
                labels = torch.tensor([sample['target_uid'] for sample in batch], dtype=torch.long, device=self.device)
                logits = self._uid_logits(pooled)
                metrics.update(self._compute_ranking_metrics_from_logits(logits, labels, batch=batch))
            return loss, metrics
        if self.config.task_type == 'sid':
            sid_name = self._primary_sid_name()
            if self._sid_decoding_mode(sid_name) == 'parallel':
                loss, metrics = self._compute_sid_parallel_loss(pooled, batch, sid_name)
                metrics.update(self._compute_sid_parallel_ranking_metrics(pooled, batch, sid_name))
            else:
                loss, metrics = self._compute_sid_loss(batch, sid_name)
                metrics.update(self._compute_sid_ranking_metrics(batch, sid_name))
            return loss, metrics
        if self.config.task_type == 'hash':
            loss, metrics = self._compute_hash_parallel_loss(pooled, batch)
            metrics.update(self._compute_hash_parallel_ranking_metrics(pooled, batch))
            return loss, metrics
        loss, metrics = self._compute_embedding_loss(pooled, batch)
        name, head, table = self._embedding_decoder_components()
        target_indices = torch.tensor(
            [int(self.compiled.item_views[name][sample['target_uid']]) for sample in batch],
            dtype=torch.long,
            device=self.device,
        )
        predictions = head(pooled)
        norm_predictions = F.normalize(predictions.float(), dim=-1)
        norm_table = F.normalize(table.float(), dim=-1)
        logits = norm_predictions @ norm_table.T
        metrics.update(self._compute_ranking_metrics_from_logits(logits, target_indices, batch=batch))
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
