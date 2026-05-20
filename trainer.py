import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn.functional as F
from pigmento import pnt
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from models import build_backbone
from utils.artifact import ArtifactStore
from utils.compile import CompileConfig, normalize_model_name
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logging import setup_logging
from utils.pipeline import ensure_compiled


def _to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _coerce_bool(value: str, default: bool):
    if value == 'auto':
        return default
    return value == 'true'


@dataclass
class TrainConfig:
    data: str
    model: str
    repr_type: str
    repr_model: Optional[str]
    repr_best: Optional[str]
    repr_combine: str
    task_type: str
    maxitems: int
    model_max_length: Optional[int]
    item_text_max_tokens: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    valid_ratio: float
    seed: int
    device: Optional[str]
    freeze_backbone: str
    use_lora: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float

    @classmethod
    def from_refconfig(cls, configurations):
        trainer = configurations.config.trainer
        return cls(
            data=configurations.data.lower(),
            model=configurations.model.lower(),
            repr_type=configurations.repr_type.lower(),
            repr_model=normalize_model_name(getattr(configurations, 'repr_model', None)),
            repr_best=configurations.repr_best.lower() if getattr(configurations, 'repr_best', None) else None,
            repr_combine=getattr(configurations, 'repr_combine', 'concat').lower(),
            task_type=configurations.task_type.lower(),
            maxitems=int(trainer.maxitems),
            model_max_length=int(trainer.model_max_length) or None,
            item_text_max_tokens=int(trainer.item_text_max_tokens),
            batch_size=int(trainer.batch_size),
            epochs=int(trainer.epochs),
            learning_rate=float(trainer.learning_rate),
            weight_decay=float(trainer.weight_decay),
            valid_ratio=float(trainer.valid_ratio),
            seed=int(trainer.seed),
            device=trainer.device,
            freeze_backbone=str(trainer.freeze_backbone).lower(),
            use_lora=str(trainer.use_lora).lower(),
            lora_rank=int(trainer.lora_rank),
            lora_alpha=int(trainer.lora_alpha),
            lora_dropout=float(trainer.lora_dropout),
            lora_target_modules=str(trainer.lora_target_modules),
            hidden_size=int(trainer.hidden_size),
            num_layers=int(trainer.num_layers),
            num_heads=int(trainer.num_heads),
            dropout=float(trainer.dropout),
        )

    @property
    def compile_config(self):
        return CompileConfig(
            data=self.data,
            model=self.model,
            repr_type=self.repr_type,
            repr_model=self.repr_model,
            repr_best=self.repr_best,
            task_type=self.task_type,
            maxitems=self.maxitems,
            model_max_length=self.model_max_length,
            item_text_max_tokens=self.item_text_max_tokens,
            repr_combine=self.repr_combine,
        )

    @property
    def run_id(self):
        parts = [
            self.compile_config.prepare_id,
            f'bs-{self.batch_size}',
            f'lr-{self.learning_rate:g}',
            f'wd-{self.weight_decay:g}',
            f'seed-{self.seed}',
            f'freeze-{self.freeze_backbone}',
            f'lora-{self.use_lora}',
        ]
        if self.model != 'transformer':
            parts.extend(
                [
                    f'r-{self.lora_rank}',
                    f'a-{self.lora_alpha}',
                    f'drop-{self.lora_dropout:g}',
                    f'target-{self.lora_target_modules.replace(",", "+")}',
                ]
            )
        return '__'.join(parts)


class CompiledSampleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.rows = []
        for row in dataframe.to_dict('records'):
            self.rows.append(
                {
                    'uid': row['uid'],
                    'history_uids': [int(value) for value in _to_list(row['history_uids'])],
                    'target_uid': int(row['target_uid']),
                    'history_item_count': int(row['history_item_count']),
                    'total_input_length': int(row['total_input_length']),
                    'target_pos': int(row['target_pos']),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CompiledArtifacts:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.store = ArtifactStore(config.data)
        self.compile_dir = self.store.compiled_dir(config.compile_config.prepare_id)
        self.meta = None
        self.prompt_main = None
        self.prompt_align = None
        self.vocab_meta = None
        self.uid_raw_items = None
        self.item_views = {}
        self.finetune = None
        self.test = None
        self.embedding_matrix = None
        self.sid_num_quantizers = None
        self.sid_codebook_size = None

    def _read_json(self, path: Path):
        return json.loads(path.read_text())

    def _load_view(self, name: str):
        path = self.compile_dir / 'item_views' / f'{name}.parquet'
        if not path.exists():
            return None
        values = pd.read_parquet(path)['value'].tolist()
        return [_to_list(value) for value in values]

    def _load_embedding_matrix(self):
        if 'embedding' not in self.item_views:
            return None
        if not self.config.repr_model:
            raise ValueError('repr.model is required when compiled data uses embedding views')
        embedding_dir = self.store.embedded_dir(self.config.repr_model)
        embedding_path = embedding_dir / 'embeddings.npy'
        if not embedding_path.exists():
            raise FileNotFoundError(f'Embedding matrix not found: {embedding_path}')
        matrix = np.load(embedding_path).astype(np.float32)
        return torch.tensor(matrix, dtype=torch.float32)

    def load(self):
        required_paths = [
            self.compile_dir / 'meta.json',
            self.compile_dir / 'samples' / 'finetune.parquet',
            self.compile_dir / 'samples' / 'test.parquet',
            self.compile_dir / 'vocab' / 'uid.json',
            self.compile_dir / 'vocab' / 'meta.json',
            self.compile_dir / 'prompts' / 'main.json',
            self.compile_dir / 'prompts' / 'alignment.json',
            self.compile_dir / 'item_views' / 'uid.parquet',
        ]
        if not all(path.exists() for path in required_paths):
            ensure_compiled(self.config.compile_config)

        self.meta = self._read_json(self.compile_dir / 'meta.json')
        self.vocab_meta = self._read_json(self.compile_dir / 'vocab' / 'meta.json')
        self.prompt_main = self._read_json(self.compile_dir / 'prompts' / 'main.json')
        self.prompt_align = self._read_json(self.compile_dir / 'prompts' / 'alignment.json')
        self.uid_raw_items = self._read_json(self.compile_dir / 'vocab' / 'uid.json')['raw_item_ids']
        self.finetune = pd.read_parquet(self.compile_dir / 'samples' / 'finetune.parquet')
        self.test = pd.read_parquet(self.compile_dir / 'samples' / 'test.parquet')

        for view_name in ['uid', 'text', 'sid', 'embedding']:
            values = self._load_view(view_name)
            if values is not None:
                self.item_views[view_name] = values

        sid_vocab_path = self.compile_dir / 'vocab' / 'sid.json'
        if sid_vocab_path.exists():
            sid_vocab = self._read_json(sid_vocab_path)
            self.sid_num_quantizers = int(sid_vocab['num_quantizers'])
            self.sid_codebook_size = int(sid_vocab['codebook_size'])

        self.embedding_matrix = self._load_embedding_matrix()
        return self

    @property
    def num_items(self):
        return len(self.uid_raw_items)

    @property
    def model_vocab_size(self):
        namespace = next(entry for entry in self.vocab_meta['namespaces'] if entry['kind'] == 'model')
        return int(namespace['size'])

    @property
    def sid_vocab_size(self):
        sid_entries = [entry for entry in self.vocab_meta['namespaces'] if entry['kind'] == 'sid']
        return int(sid_entries[0]['size']) if sid_entries else 0

    @property
    def model_kind(self):
        return self.meta['model_kind']

    @property
    def model_key(self):
        model_vocab = self._read_json(self.compile_dir / 'vocab' / 'model.json')
        return model_vocab.get('model_key')


class LLMSequenceEncoder(nn.Module):
    def __init__(
            self,
            model_key: str,
            freeze_backbone: bool,
            use_lora: bool,
            lora_rank: int,
            lora_alpha: int,
            lora_dropout: float,
            lora_target_modules: str,
    ):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_key, trust_remote_code=True)
        hidden_size = getattr(base_model.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, 'd_model', None)
        if hidden_size is None:
            raise ValueError(f'Cannot resolve hidden size from model config for {model_key}')
        self.hidden_size = int(hidden_size)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            target_modules = 'all-linear' if lora_target_modules == 'all-linear' else [
                value.strip() for value in lora_target_modules.split(',') if value.strip()
            ]
            self.model = get_peft_model(
                base_model,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                ),
            )
            trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
            total_params = sum(param.numel() for param in self.model.parameters())
            pnt(
                f'initialized LoRA for {model_key} '
                f'trainable={trainable_params:,}/{total_params:,} '
                f'r={lora_rank} alpha={lora_alpha} dropout={lora_dropout:g} '
                f'targets={lora_target_modules}'
            )
        else:
            self.model = base_model
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and not self.use_lora:
            self.model.eval()
        return self

    def embed_model_tokens(self, token_ids: torch.Tensor):
        return self.model.get_input_embeddings()(token_ids)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        use_no_grad = self.freeze_backbone and not self.use_lora
        with torch.set_grad_enabled(not use_no_grad):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.last_hidden_state


class ScratchSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float, max_length: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def embed_model_tokens(self, token_ids: torch.Tensor):
        return self.token_embedding(token_ids)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, _ = inputs_embeds.shape
        positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        hidden = inputs_embeds + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=attention_mask == 0,
        )
        return hidden


class SequentialRecModel(nn.Module):
    def __init__(self, compiled: CompiledArtifacts, config: TrainConfig):
        super().__init__()
        self.compiled = compiled
        self.config = config
        self.backbone_def = build_backbone(
            config.model,
            [''],
            max_length_override=config.model_max_length,
        )
        freeze_default = compiled.model_kind == 'llm'
        self.freeze_backbone = _coerce_bool(config.freeze_backbone, default=freeze_default)
        self.use_lora = _coerce_bool(config.use_lora, default=compiled.model_kind == 'llm')

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
                token_ids = [int(token_id) for token_id in _to_list(self.compiled.item_views['text'][uid])]
                specs.append(('model_tokens', token_ids))
            elif repr_type == 'sid':
                sid_ids = [int(token_id) for token_id in _to_list(self.compiled.item_views['sid'][uid])]
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
            [int(token_id) for token_id in _to_list(self.compiled.item_views['sid'][sample['target_uid']])]
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


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.compiled = CompiledArtifacts(config).load()
        self.run_dir = ArtifactStore(config.data).trained_dir(config.run_id)
        self.device = self._resolve_device()
        self.model = SequentialRecModel(self.compiled, config).to(self.device)
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def _resolve_device(self):
        if self.config.device:
            return torch.device(self.config.device)
        return torch.device(GPU.auto_choose(torch_format=True))

    def build_dataloaders(self):
        finetune = CompiledSampleDataset(self.compiled.finetune)
        test = CompiledSampleDataset(self.compiled.test)

        indices = np.arange(len(finetune))
        rng = np.random.default_rng(self.config.seed)
        rng.shuffle(indices)
        valid_size = max(1, int(len(indices) * self.config.valid_ratio))
        valid_indices = indices[:valid_size]
        train_indices = indices[valid_size:]
        if len(train_indices) == 0:
            train_indices = valid_indices[:1]

        train_rows = [finetune[index] for index in train_indices]
        valid_rows = [finetune[index] for index in valid_indices]
        test_rows = [test[index] for index in range(len(test))]

        self.train_loader = DataLoader(train_rows, batch_size=self.config.batch_size, shuffle=True, collate_fn=lambda batch: batch)
        self.valid_loader = DataLoader(valid_rows, batch_size=self.config.batch_size, shuffle=False, collate_fn=lambda batch: batch)
        self.test_loader = DataLoader(test_rows, batch_size=self.config.batch_size, shuffle=False, collate_fn=lambda batch: batch)
        pnt(
            f'built dataloaders train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} '
            f'batch_size={self.config.batch_size}'
        )

    def _metric_name(self):
        if self.config.task_type == 'uid':
            return 'uid_acc'
        if self.config.task_type == 'sid':
            return 'sid_seq_acc'
        return 'embedding_cosine'

    def build_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in params)
        pnt(f'build optimizer with trainable params {trainable:,}/{total:,}')
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _run_loader(self, loader, optimizer=None, desc='train'):
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss = 0.0
        total_batches = 0
        metric_sums = {}

        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            if is_train:
                optimizer.zero_grad()
            loss, metrics = self.model.forward_batch(batch)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            postfix = {'loss': f'{loss.item():.4f}'}
            for key, value in metrics.items():
                postfix[key] = f'{value:.4f}'
            iterator.set_postfix(postfix)

        summary = {'loss': total_loss / max(total_batches, 1)}
        for key, value in metric_sums.items():
            summary[key] = value / max(total_batches, 1)
        return summary

    def _save_checkpoint(self, epoch: int, best_metric: float, valid_metrics: dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.trainable_state_dict(),
            'checkpoint_kind': 'trainable_only',
            'config': asdict(self.config),
            'valid_metrics': valid_metrics,
            'best_metric': best_metric,
        }
        path = self.run_dir / 'best.pt'
        torch.save(checkpoint, path)
        pnt(f'saved best checkpoint to {path} with trainable-only state')

    def _save_meta(self, best_epoch: int, best_metric: float, test_metrics: dict):
        meta = {
            'config': asdict(self.config),
            'compiled_dir': str(self.compiled.compile_dir),
            'run_dir': str(self.run_dir),
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'metric_name': self._metric_name(),
            'test_metrics': test_metrics,
        }
        path = self.run_dir / 'meta.json'
        path.write_text(json.dumps(meta, indent=2) + '\n')
        pnt(f'wrote trainer meta to {path}')

    def train(self):
        self.build_dataloaders()
        optimizer = self.build_optimizer()
        metric_name = self._metric_name()
        best_metric = -float('inf')
        best_epoch = 0

        pnt(
            f'start training on {self.config.data} with {self.config.model} '
            f'repr={self.config.repr_type} task={self.config.task_type} device={self.device}'
        )

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_loader(self.train_loader, optimizer=optimizer, desc=f'train@{epoch}')
            valid_metrics = self._run_loader(self.valid_loader, optimizer=None, desc=f'valid@{epoch}')
            pnt(
                f'epoch {epoch:03d} train_loss={train_metrics["loss"]:.4f} '
                f'valid_loss={valid_metrics["loss"]:.4f} '
                f'{metric_name}={valid_metrics.get(metric_name, 0.0):.4f}'
            )

            current_metric = valid_metrics.get(metric_name, -valid_metrics['loss'])
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                self._save_checkpoint(epoch, best_metric, valid_metrics)

        checkpoint = torch.load(self.run_dir / 'best.pt', map_location=self.device)
        load_info = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        missing = getattr(load_info, 'missing_keys', [])
        unexpected = getattr(load_info, 'unexpected_keys', [])
        if unexpected:
            raise RuntimeError(f'unexpected checkpoint keys: {unexpected}')
        pnt(f'loaded best checkpoint with {len(missing)} missing frozen/base keys')
        test_metrics = self._run_loader(self.test_loader, optimizer=None, desc='test')
        pnt(
            f'best_epoch={best_epoch} test_loss={test_metrics["loss"]:.4f} '
            f'{metric_name}={test_metrics.get(metric_name, 0.0):.4f}'
        )
        self._save_meta(best_epoch, best_metric, test_metrics)


if __name__ == '__main__':
    setup_logging()

    configurations = ConfigInit(
        required_args=['data', 'model', 'repr_type', 'task_type'],
        default_args=dict(
            config='config/trainer.yaml',
            repr_model=None,
            repr_best=None,
            repr_combine='concat',
        ),
        makedirs=[],
    ).parse()

    config = TrainConfig.from_refconfig(configurations)
    trainer = Trainer(config)
    trainer.train()
