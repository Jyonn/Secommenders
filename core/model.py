import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models import build_backbone

from .dataset import to_list
from .encoders import LLMSequenceEncoder, ScratchSequenceEncoder, coerce_bool


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
        self.freeze_backbone = coerce_bool(config.freeze_backbone, default=freeze_default)
        self.use_lora = coerce_bool(config.use_lora, default=compiled.model_kind == 'llm')

        if compiled.model_kind == 'llm':
            self.encoder = LLMSequenceEncoder(
                compiled.model_key,
                freeze_backbone=self.freeze_backbone,
                use_lora=self.use_lora,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                lora_target_modules=config.lora_target_modules,
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

    @property
    def device(self):
        return next(self.parameters()).device

    def trainable_state_dict(self):
        state = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: tensor.detach().cpu() for name, tensor in state.items() if name in trainable_names}

    def _render_history_item(self, uid: int):
        if self.config.repr_combine == 'add':
            if 'embedding' not in self.compiled.item_views:
                raise ValueError('repr.combine=add requires compiled embedding view')
            emb_index = int(self.compiled.item_views['embedding'][uid])
            return [('uid_embedding_add', (uid, emb_index))]

        specs = []
        for repr_type in self.config.compile_config.repr_types:
            if repr_type == 'uid':
                specs.append(('uid', uid))
            elif repr_type == 'text':
                token_ids = [int(token_id) for token_id in to_list(self.compiled.item_views['text'][uid])]
                specs.append(('model_tokens', token_ids))
            elif repr_type == 'sid':
                sid_ids = [int(token_id) for token_id in to_list(self.compiled.item_views['sid'][uid])]
                specs.append(('sid', sid_ids))
            elif repr_type == 'embedding':
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
        return specs

    def _embed_spec(self, kind: str, value):
        if kind == 'model_tokens':
            token_ids = torch.tensor(value, dtype=torch.long, device=self.device)
            return self.encoder.embed_model_tokens(token_ids)
        if kind == 'uid':
            token_ids = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            return self.uid_embedding(token_ids)
        if kind == 'sid':
            token_ids = torch.tensor(value, dtype=torch.long, device=self.device)
            return self.sid_embedding(token_ids)
        if kind == 'embedding':
            emb_index = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            projected = self.embedding_projection(self.embedding_matrix[emb_index])
            return projected
        if kind == 'uid_embedding_add':
            uid, emb_index = value
            uid_tensor = torch.tensor([int(uid)], dtype=torch.long, device=self.device)
            emb_tensor = torch.tensor([int(emb_index)], dtype=torch.long, device=self.device)
            uid_embed = self.uid_embedding(uid_tensor)
            content_embed = self.embedding_projection(self.embedding_matrix[emb_tensor])
            return uid_embed + content_embed
        raise ValueError(f'Unknown spec kind: {kind}')

    def _build_batch_inputs(self, batch):
        sample_embeddings = []
        for sample in batch:
            specs = self._build_sample_specs(sample)
            pieces = [self._embed_spec(kind, value) for kind, value in specs]
            sample_embeddings.append(torch.cat(pieces, dim=0))

        padded = pad_sequence(sample_embeddings, batch_first=True)
        lengths = torch.tensor([emb.shape[0] for emb in sample_embeddings], dtype=torch.long, device=self.device)
        attention_mask = torch.arange(padded.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, attention_mask.long(), lengths

    def _compute_uid_loss(self, pooled: torch.Tensor, batch):
        logits = self.target_head(pooled)
        labels = torch.tensor([sample['target_uid'] for sample in batch], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, {
            'uid_acc': accuracy.item(),
        }

    def _compute_sid_loss(self, pooled: torch.Tensor, batch):
        sid_targets = [
            [int(token_id) for token_id in to_list(self.compiled.item_views['sid'][sample['target_uid']])]
            for sample in batch
        ]
        labels = torch.tensor(sid_targets, dtype=torch.long, device=self.device)
        logits = self.target_head(pooled).view(len(batch), self.compiled.sid_num_quantizers, self.compiled.sid_vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, self.compiled.sid_vocab_size), labels.reshape(-1))
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
        targets = self.embedding_matrix[target_indices]
        predictions = self.embedding_head(pooled)
        loss = F.mse_loss(predictions, targets)
        cosine = F.cosine_similarity(predictions, targets, dim=-1).mean()
        return loss, {
            'embedding_cosine': cosine.item(),
        }

    def forward_batch(self, batch):
        inputs_embeds, attention_mask, lengths = self._build_batch_inputs(batch)
        hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = hidden[torch.arange(hidden.shape[0], device=self.device), lengths - 1]

        if self.config.task_type == 'uid':
            return self._compute_uid_loss(pooled, batch)
        if self.config.task_type == 'sid':
            return self._compute_sid_loss(pooled, batch)
        return self._compute_embedding_loss(pooled, batch)
