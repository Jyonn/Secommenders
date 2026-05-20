import argparse
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from processors.base_processor import Processor
from utils import model as model_utils
from utils.artifact import ArtifactStore
from utils.function import load_processor


def _normalize_model_name(name: Optional[str]):
    if not name:
        return None
    return name.replace('.', '').lower()


@dataclass
class CompileConfig:
    data: str
    model: str
    repr_type: str
    repr_model: Optional[str]
    repr_best: Optional[str]
    task_type: str
    maxitems: int
    item_text_max_tokens: int = 50
    alignment: bool = True
    repr_combine: str = 'concat'

    @property
    def repr_types(self):
        return [part.strip().lower() for part in self.repr_type.split('+') if part.strip()]

    @property
    def prepare_id(self):
        parts = [
            f'model-{self.model}',
            f'repr-{self.repr_type}',
            f'combine-{self.repr_combine}',
            f'task-{self.task_type}',
            f'maxitems-{"auto" if self.maxitems == 0 else self.maxitems}',
            f'textlen-{self.item_text_max_tokens}',
        ]
        if self.repr_model:
            parts.append(f'reprmodel-{self.repr_model}')
        if self.repr_best:
            parts.append(f'reprbest-{self.repr_best}')
        return '__'.join(parts)

    @property
    def config_dict(self):
        return asdict(self)


class VocabularyRegistry:
    def __init__(self):
        self.entries = []

    def register(self, name, kind, size, path, **kwargs):
        entry = {
            'name': name,
            'kind': kind,
            'size': int(size),
            'path': str(path),
            'namespace_id': len(self.entries),
        }
        entry.update(kwargs)
        self.entries.append(entry)
        return entry

    def to_dict(self):
        return {'namespaces': self.entries}


class ModelAdapter:
    DEFAULT_MAX_LENGTH = 512

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_spec = None

    @property
    def namespace_name(self):
        return self.model_name

    @property
    def max_length(self):
        raise NotImplementedError

    @property
    def kind(self):
        raise NotImplementedError

    def build_vocab_artifact(self):
        raise NotImplementedError

    def build_prompt_spec(self):
        raise NotImplementedError

    def tokenize_texts(self, texts: list[str], max_tokens: int):
        raise NotImplementedError

    def estimate_main_length(self, history_values: list, target_value, task_type: str):
        raise NotImplementedError

    def build_alignment_spec(self):
        raise NotImplementedError


class LLMAdapter(ModelAdapter):
    HISTORY_PREFIX = 'A user has browsed the following items:'
    ITEM_SEPARATOR = ','
    QUERY_PREFIX = 'Which item would the user probably interact with:'

    ALIGN_PREFIX = 'An item featured'
    ALIGN_BRIDGE = 'can be mapped to'

    def __init__(self, model_name: str, model_key: str):
        super().__init__(model_name)
        self.model_key = model_key
        self.tokenizer = AutoTokenizer.from_pretrained(model_key, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = AutoConfig.from_pretrained(model_key, trust_remote_code=True)
        self._max_length = self._resolve_max_length()

    @property
    def kind(self):
        return 'llm'

    @property
    def max_length(self):
        return self._max_length

    def _resolve_max_length(self):
        tokenizer_max = getattr(self.tokenizer, 'model_max_length', None)
        if tokenizer_max and tokenizer_max < 1_000_000:
            return int(tokenizer_max)

        for attr in ['max_position_embeddings', 'n_positions', 'seq_length', 'max_seq_len', 'model_max_length']:
            value = getattr(self.config, attr, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                return int(value)
        return self.DEFAULT_MAX_LENGTH

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_texts(self, texts: list[str], max_tokens: int):
        return [
            self.tokenizer.encode(text or '[Empty Content]', add_special_tokens=False, truncation=True, max_length=max_tokens)
            for text in texts
        ]

    def build_vocab_artifact(self):
        vocab = self.tokenizer.get_vocab()
        size = max(vocab.values()) + 1 if vocab else 0
        tokens = [''] * size
        for token, index in vocab.items():
            tokens[index] = token
        return {
            'tokens': tokens,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'eos_token_id': getattr(self.tokenizer, 'eos_token_id', None),
            'pad_token_id': getattr(self.tokenizer, 'pad_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
            'model_key': self.model_key,
        }

    def build_prompt_spec(self):
        if self.prompt_spec is not None:
            return self.prompt_spec
        self.prompt_spec = {
            'history_prefix_ids': self._encode(self.HISTORY_PREFIX),
            'item_separator_ids': self._encode(self.ITEM_SEPARATOR),
            'query_prefix_ids': self._encode(self.QUERY_PREFIX),
            'max_length': self.max_length,
            'kind': self.kind,
        }
        return self.prompt_spec

    def build_alignment_spec(self):
        return {
            'align_prefix_ids': self._encode(self.ALIGN_PREFIX),
            'align_bridge_ids': self._encode(self.ALIGN_BRIDGE),
            'kind': self.kind,
        }

    @staticmethod
    def _value_length(value, task_type):
        if task_type == 'embedding':
            return 0
        if isinstance(value, list):
            return len(value)
        return 1

    def estimate_main_length(self, history_values: list, target_value, task_type: str):
        prompt = self.build_prompt_spec()
        history_len = sum(len(value) if isinstance(value, list) else 1 for value in history_values)
        separator_len = len(prompt['item_separator_ids']) * max(0, len(history_values) - 1)
        target_len = self._value_length(target_value, task_type)
        return (
            len(prompt['history_prefix_ids'])
            + history_len
            + separator_len
            + len(prompt['query_prefix_ids'])
            + target_len
        )


class ScratchTokenizer:
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    HISTORY_TOKEN = '<history>'
    SEP_TOKEN = '<sep>'
    NEXT_TOKEN = '<next>'
    ALIGN_TOKEN = '<align>'
    TO_TOKEN = '<to>'

    def __init__(self, texts: list[str]):
        base_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.HISTORY_TOKEN,
            self.SEP_TOKEN,
            self.NEXT_TOKEN,
            self.ALIGN_TOKEN,
            self.TO_TOKEN,
        ]
        vocab = {}
        tokens = []
        for token in base_tokens:
            vocab[token] = len(tokens)
            tokens.append(token)

        for text in texts:
            for token in self.tokenize(text):
                if token not in vocab:
                    vocab[token] = len(tokens)
                    tokens.append(token)

        self.vocab = vocab
        self.tokens = tokens

    @classmethod
    def tokenize(cls, text: str):
        text = (text or '').lower().strip()
        if not text:
            return []
        return cls.TOKEN_PATTERN.findall(text)

    def encode(self, text: str, max_tokens: Optional[int] = None):
        tokens = [self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in self.tokenize(text)]
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        return tokens


class ScratchAdapter(ModelAdapter):
    def __init__(self, model_name: str, texts: list[str]):
        super().__init__(model_name)
        self.tokenizer = ScratchTokenizer(texts)

    @property
    def kind(self):
        return 'scratch'

    @property
    def max_length(self):
        return self.DEFAULT_MAX_LENGTH

    def tokenize_texts(self, texts: list[str], max_tokens: int):
        return [self.tokenizer.encode(text or '[Empty Content]', max_tokens=max_tokens) for text in texts]

    def build_vocab_artifact(self):
        return {
            'tokens': self.tokenizer.tokens,
            'pad_token_id': self.tokenizer.vocab[ScratchTokenizer.PAD_TOKEN],
            'unk_token_id': self.tokenizer.vocab[ScratchTokenizer.UNK_TOKEN],
        }

    def build_prompt_spec(self):
        if self.prompt_spec is not None:
            return self.prompt_spec
        self.prompt_spec = {
            'history_prefix_ids': [self.tokenizer.vocab[ScratchTokenizer.HISTORY_TOKEN]],
            'item_separator_ids': [self.tokenizer.vocab[ScratchTokenizer.SEP_TOKEN]],
            'query_prefix_ids': [self.tokenizer.vocab[ScratchTokenizer.NEXT_TOKEN]],
            'max_length': self.max_length,
            'kind': self.kind,
        }
        return self.prompt_spec

    def build_alignment_spec(self):
        return {
            'align_prefix_ids': [self.tokenizer.vocab[ScratchTokenizer.ALIGN_TOKEN]],
            'align_bridge_ids': [self.tokenizer.vocab[ScratchTokenizer.TO_TOKEN]],
            'kind': self.kind,
        }

    def estimate_main_length(self, history_values: list, target_value, task_type: str):
        prompt = self.build_prompt_spec()
        history_len = sum(len(value) if isinstance(value, list) else 1 for value in history_values)
        separator_len = len(prompt['item_separator_ids']) * max(0, len(history_values) - 1)
        if task_type == 'embedding':
            target_len = 0
        elif isinstance(target_value, list):
            target_len = len(target_value)
        else:
            target_len = 1
        return len(prompt['history_prefix_ids']) + history_len + separator_len + len(prompt['query_prefix_ids']) + target_len


class Compiler:
    VER = 'v1.2'
    SUPPORTED_REPR_TYPES = {'uid', 'sid', 'text', 'embedding'}
    SUPPORTED_TASK_TYPES = {'uid', 'sid', 'embedding'}
    SUPPORTED_REPR_COMBINES = {'concat', 'add'}

    def __init__(self, config: CompileConfig):
        self.config = config
        self.store = ArtifactStore(config.data)
        self.output_dir = self.store.compiled_dir(config.prepare_id)
        self.processor: Optional[Processor] = None
        self.model_adapter: Optional[ModelAdapter] = None
        self.registry = VocabularyRegistry()
        self.uid_raw_items: list = []
        self.uid_item_map = {}
        self.item_texts: list[str] = []
        self.item_views = {}
        self.samples_stats = {}
        self.sample_visuals = {}

        self._init_dirs()

    def _init_dirs(self):
        self.vocab_dir = self.output_dir / 'vocab'
        self.prompts_dir = self.output_dir / 'prompts'
        self.item_views_dir = self.output_dir / 'item_views'
        self.samples_dir = self.output_dir / 'samples'
        self.alignment_dir = self.output_dir / 'alignment'
        for path in [self.vocab_dir, self.prompts_dir, self.item_views_dir, self.samples_dir, self.alignment_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        print(f'|Compiler| {message}')

    @staticmethod
    def _preview_list(values, limit=3):
        values = list(values)
        if len(values) <= limit:
            return values
        return values[:limit] + ['...']

    @staticmethod
    def _ansi(text: str, fg: Optional[int] = None, bg: Optional[int] = None, bold: bool = False, dim: bool = False):
        codes = []
        if bold:
            codes.append('1')
        if dim:
            codes.append('2')
        if fg is not None:
            codes.append(str(fg))
        if bg is not None:
            codes.append(str(bg))
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def _history_repr_label(self):
        if self.config.repr_combine == 'add':
            return '<uid+emb>'
        return '<' + '+'.join(self.config.repr_types) + '>'

    def _task_repr_label(self):
        return f'<{self.config.task_type}>'

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 48):
        text = (text or '').replace('\n', ' ').strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + '...'

    @staticmethod
    def _truncate_repr(text: str, max_chars: int = 72):
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + '...'

    def _short_item_id(self, uid: int):
        raw_id = str(self.uid_raw_items[uid])
        if len(raw_id) <= 10:
            return raw_id
        return raw_id[:4] + '...' + raw_id[-3:]

    def _summarize_view_value(self, view_name: str, uid: int):
        if view_name == 'uid':
            return f'uid={uid}'
        if view_name == 'text':
            token_ids = self.item_views['text'][uid]
            preview = self._truncate_text(self.item_texts[uid], max_chars=36)
            return f'text[{len(token_ids)}]="{preview}"'
        if view_name == 'sid':
            codes = self._as_token_list(self.item_views['sid'][uid])
            preview = ','.join(str(code) for code in codes[:6])
            if len(codes) > 6:
                preview += ',...'
            return f'sid[{len(codes)}]=[{preview}]'
        if view_name == 'embedding':
            return f'emb#{self.item_views["embedding"][uid]}'
        return str(self.item_views[view_name][uid])

    def _history_repr_summary(self, uid: int):
        if self.config.repr_combine == 'add':
            uid_part = self._summarize_view_value('uid', uid)
            emb_part = self._summarize_view_value('embedding', uid)
            return f'add({uid_part} + linear({emb_part})) -> 1 slot'

        parts = [self._summarize_view_value(repr_type, uid) for repr_type in self.config.repr_types]
        return ' + '.join(parts)

    def _task_repr_summary(self, uid: int):
        return self._summarize_view_value(self.config.task_type, uid)

    @staticmethod
    def _role_name(index: int, history_start: int, target_pos: int):
        if index < history_start:
            return 'skip'
        if index < target_pos:
            return 'hist'
        if index == target_pos:
            return 'tgt'
        return 'tail'

    def _role_style(self, role: str):
        if role == 'skip':
            return 30, 47, False, True
        if role == 'hist':
            return 30, 46, True, False
        if role == 'tgt':
            return 37, 41, True, False
        return 37, 100, False, True

    def _render_sequence_strip(self, sequence: list[int], history_start: int, target_pos: int):
        cells = []
        for index, uid in enumerate(sequence):
            role = self._role_name(index, history_start, target_pos)
            fg, bg, bold, dim = self._role_style(role)
            label = f'i{index + 1}:{self._short_item_id(uid)}'
            cells.append(self._ansi(label, fg=fg, bg=bg, bold=bold, dim=dim))
        return ' '.join(cells)

    def _render_sample_visual(self, split_name: str, user_id, sequence: list[int], history_uids: list[int], target_uid: int,
                              target_pos: int, total_input_length: int):
        history_start = target_pos - len(history_uids)
        target_raw = self.uid_raw_items[target_uid]
        lines = []
        title = self._ansi(f' {split_name.upper()} SAMPLE PREVIEW ', fg=37, bg=45, bold=True)
        lines.append(f'  {title}')
        lines.append(
            f'  user_id   : {user_id} | sequence_len={len(sequence)} | target_pos={target_pos + 1} '
            f'| history_span=[{history_start + 1}, {target_pos}] | target_raw={target_raw} '
            f'| input_len={total_input_length}/{self.model_adapter.max_length}'
        )
        lines.append('  legend    : '
                     + self._ansi(' skip ', fg=30, bg=47, dim=True)
                     + ' '
                     + self._ansi(' history ', fg=30, bg=46, bold=True)
                     + ' '
                     + self._ansi(' target ', fg=37, bg=41, bold=True)
                     + ' '
                     + self._ansi(' tail ', fg=37, bg=100, dim=True))
        lines.append(f'  sequence  : {self._render_sequence_strip(sequence, history_start, target_pos)}')
        lines.append(f'  repr rule : history={self._history_repr_label()} | target={self._task_repr_label()}')
        lines.append('  details   :')
        for index, uid in enumerate(sequence):
            role = self._role_name(index, history_start, target_pos)
            fg, bg, bold, dim = self._role_style(role)
            role_tag = self._ansi(role.upper().ljust(4), fg=fg, bg=bg, bold=bold, dim=dim)
            raw_id = self.uid_raw_items[uid]
            if role == 'hist':
                repr_text = self._truncate_repr(self._history_repr_summary(uid))
            elif role == 'tgt':
                repr_text = self._truncate_repr(self._task_repr_summary(uid))
            else:
                repr_text = 'not used in this sample'
            lines.append(f'    {role_tag} pos={index + 1:>2} uid={uid:<5} raw={raw_id} -> {repr_text}')
        if split_name == 'finetune':
            lines.append('  policy    : for each target item, keep the longest suffix history that still fits model max length')
        else:
            lines.append('  policy    : only the final item is evaluated; its history is the longest suffix that still fits model max length')
        return '\n'.join(lines)

    @property
    def meta_path(self):
        return self.output_dir / 'meta.json'

    def is_cached(self):
        if not self.meta_path.exists():
            return False
        meta = json.loads(self.meta_path.read_text())
        if meta.get('config') != self.config.config_dict:
            return False
        required_item_view_paths = [self.item_views_dir / 'uid.parquet']
        if self.requires_view('text'):
            required_item_view_paths.append(self.item_views_dir / 'text.parquet')
        if self.requires_view('sid'):
            required_item_view_paths.append(self.item_views_dir / 'sid.parquet')
        if self.requires_view('embedding'):
            required_item_view_paths.append(self.item_views_dir / 'embedding.parquet')
        required_paths = [
            self.output_dir / 'stats.json',
            self.vocab_dir / 'meta.json',
            self.prompts_dir / 'main.json',
            self.prompts_dir / 'alignment.json',
            self.item_views_dir / 'meta.json',
            self.samples_dir / 'finetune.parquet',
            self.samples_dir / 'test.parquet',
            self.alignment_dir / 'meta.json',
        ] + required_item_view_paths
        return all(path.exists() for path in required_paths)

    def validate(self):
        repr_types = self.config.repr_types
        if not repr_types:
            raise ValueError('repr.type must contain at least one representation')
        if len(set(repr_types)) != len(repr_types):
            raise ValueError(f'repr.type contains duplicates: {self.config.repr_type}')
        unsupported_repr_types = [repr_type for repr_type in repr_types if repr_type not in self.SUPPORTED_REPR_TYPES]
        if unsupported_repr_types:
            raise ValueError(f'Unsupported repr.type entries: {unsupported_repr_types}')
        if self.config.task_type not in self.SUPPORTED_TASK_TYPES:
            raise ValueError(f'Unsupported task.type: {self.config.task_type}')
        if self.config.repr_combine not in self.SUPPORTED_REPR_COMBINES:
            raise ValueError(f'Unsupported repr.combine: {self.config.repr_combine}')
        if self.config.repr_combine == 'add':
            if set(repr_types) != {'uid', 'embedding'} or len(repr_types) != 2:
                raise ValueError('repr.combine=add is only supported for repr.type=uid+embedding')

        external_view_required = any(view in {'sid', 'embedding'} for view in repr_types + [self.config.task_type])
        if external_view_required and not self.config.repr_model:
            raise ValueError('repr.model is required when repr.type or task.type uses sid/embedding')
        if 'sid' in set(repr_types + [self.config.task_type]) and not self.config.repr_best:
            raise ValueError('repr.best is required when repr.type or task.type uses sid')

    def run(self):
        self.validate()
        self._log(
            f'start compile data={self.config.data} model={self.config.model} '
            f'repr={self.config.repr_type} combine={self.config.repr_combine} '
            f'task={self.config.task_type} maxitems={self.config.maxitems} '
            f'textlen={self.config.item_text_max_tokens} alignment={self.config.alignment}'
        )
        if self.is_cached():
            self._log(f'compiled dataset cached at {self.output_dir}')
            return

        self._log(f'cache miss, compiling into {self.output_dir}')
        self.load_processor()
        self.init_model_adapter()
        self.build_vocab_and_prompts()
        self.build_item_views()
        self.build_samples('finetune', self.processor.finetune_set)
        self.build_samples('test', self.processor.test_set)
        self.build_alignment_meta()
        self.save_meta()
        self.save_stats()
        self.log_sample_visuals()
        self._log(f'compile finished: outputs written to {self.output_dir}')

    def load_processor(self):
        self._log(f'loading processed dataset {self.config.data}')
        self.processor = load_processor(self.config.data)
        self.processor.load()
        self.uid_raw_items = self.processor.items[self.processor.IID_COL].tolist()
        self.uid_item_map = {item_id: index for index, item_id in enumerate(self.uid_raw_items)}
        self.item_texts = [
            self.processor.organize_item(item_id, item_attrs=self.processor.default_attrs) or '[Empty Content]'
            for item_id in self.uid_raw_items
        ]
        self._log(
            f'loaded processed assets: items={len(self.uid_raw_items)} '
            f'finetune_users={len(self.processor.finetune_set)} test_users={len(self.processor.test_set)}'
        )

    def init_model_adapter(self):
        model_key = model_utils.match(self.config.model)
        if model_key:
            self.model_adapter = LLMAdapter(self.config.model, model_key)
            self._log(
                f'initialized llm adapter model={self.config.model} '
                f'hf_key={model_key} max_length={self.model_adapter.max_length}'
            )
            return
        self.model_adapter = ScratchAdapter(self.config.model, self.item_texts)
        self._log(
            f'initialized scratch adapter model={self.config.model} '
            f'max_length={self.model_adapter.max_length} '
            f'vocab_size={len(self.model_adapter.tokenizer.tokens)}'
        )

    def _save_json(self, path: Path, data):
        path.write_text(json.dumps(data, indent=2) + '\n')

    def _write_view(self, name: str, values: list):
        path = self.item_views_dir / f'{name}.parquet'
        pd.DataFrame({'value': values}).to_parquet(path, index=False)
        self.item_views[name] = values
        return path

    def build_vocab_and_prompts(self):
        self._log('building vocab registry and prompt assets')
        model_vocab_path = self.vocab_dir / 'model.json'
        model_vocab_artifact = self.model_adapter.build_vocab_artifact()
        self._save_json(model_vocab_path, model_vocab_artifact)
        self.registry.register(
            self.model_adapter.namespace_name,
            kind='model',
            size=len(model_vocab_artifact.get('tokens', [])),
            path=model_vocab_path,
            model_kind=self.model_adapter.kind,
        )

        uid_vocab_path = self.vocab_dir / 'uid.json'
        self._save_json(uid_vocab_path, {'raw_item_ids': self.uid_raw_items})
        self.registry.register(
            'uid',
            kind='uid',
            size=len(self.uid_raw_items),
            path=uid_vocab_path,
        )

        if self.requires_view('sid'):
            sid_meta = self.load_sid_view(build_only_meta=True)
            sid_vocab_path = self.vocab_dir / 'sid.json'
            tokens = [
                f'q{quantizer}_c{code}'
                for quantizer in range(sid_meta['num_quantizers'])
                for code in range(sid_meta['codebook_size'])
            ]
            self._save_json(
                sid_vocab_path,
                {
                    'tokens': tokens,
                    'num_quantizers': sid_meta['num_quantizers'],
                    'codebook_size': sid_meta['codebook_size'],
                },
            )
            self.registry.register(
                'sid',
                kind='sid',
                size=len(tokens),
                path=sid_vocab_path,
                num_quantizers=sid_meta['num_quantizers'],
                codebook_size=sid_meta['codebook_size'],
            )
            self._log(
                f"registered sid vocab size={len(tokens)} "
                f"num_quantizers={sid_meta['num_quantizers']} codebook_size={sid_meta['codebook_size']}"
            )

        self._save_json(self.vocab_dir / 'meta.json', self.registry.to_dict())
        main_prompt = self.model_adapter.build_prompt_spec()
        alignment_prompt = self.model_adapter.build_alignment_spec()
        self._save_json(self.prompts_dir / 'main.json', main_prompt)
        self._save_json(self.prompts_dir / 'alignment.json', alignment_prompt)
        self._log(
            f'vocab ready namespaces={len(self.registry.entries)} '
            f'main_prompt=(history_prefix={len(main_prompt["history_prefix_ids"])}, '
            f'separator={len(main_prompt["item_separator_ids"])}, '
            f'query_prefix={len(main_prompt["query_prefix_ids"])})'
        )

    def requires_view(self, view_name: str):
        views = {'uid', *self.config.repr_types, self.config.task_type}
        if self.config.alignment and ('text' not in views or len(views) > 1):
            views.add('text')
        return view_name in views

    def build_item_views(self):
        required_views = [view for view in ['uid', 'text', 'sid', 'embedding'] if self.requires_view(view)]
        self._log(f'building item views {required_views} for {len(self.uid_raw_items)} items')
        self._write_view('uid', list(range(len(self.uid_raw_items))))
        self._log('uid view ready')

        if self.requires_view('text'):
            self._log(
                f'tokenizing text view with model={self.config.model} '
                f'max_tokens={self.config.item_text_max_tokens}'
            )
            chunk_size = 256
            text_values = []
            text_progress = tqdm(
                range(0, len(self.item_texts), chunk_size),
                desc='text view',
                leave=False,
            )
            for start in text_progress:
                batch_texts = self.item_texts[start:start + chunk_size]
                text_values.extend(
                    self.model_adapter.tokenize_texts(
                        batch_texts,
                        max_tokens=self.config.item_text_max_tokens,
                    )
                )
            self._write_view('text', text_values)
            text_lengths = [len(value) for value in text_values]
            self._log(
                f'text view ready avg_len={np.mean(text_lengths):.2f} '
                f'max_len={max(text_lengths) if text_lengths else 0}'
            )

        if self.requires_view('sid'):
            self._log(
                f'loading sid view from model={self.config.repr_model} '
                f'checkpoint={self.config.repr_best}'
            )
            sid_values = self.load_sid_view()
            self._write_view('sid', sid_values)
            sid_lengths = [len(value) for value in sid_values]
            self._log(
                f'sid view ready avg_codes={np.mean(sid_lengths):.2f} '
                f'max_codes={max(sid_lengths) if sid_lengths else 0}'
            )

        if self.requires_view('embedding'):
            self._log(f'loading embedding view from model={self.config.repr_model}')
            embedding_values = self.load_embedding_view()
            self._write_view('embedding', embedding_values)
            self._log(
                f'embedding view ready indices={len(embedding_values)} '
                f'preview={self._preview_list(embedding_values)}'
            )

        self._save_json(
            self.item_views_dir / 'meta.json',
            {
                'row_order': 'uid_vocab',
                'views': sorted(self.item_views),
            },
        )
        self._log(f'item view manifest saved: {sorted(self.item_views)}')

    def _load_quantized_export(self):
        model_name = _normalize_model_name(self.config.repr_model)
        export_dir = self.store.quantized_dir(model_name) / 'exports' / self.config.repr_best
        meta_path = export_dir / 'meta.json'
        codes_path = export_dir / 'codebook_indices.npy'
        item_ids_path = export_dir / 'item_ids.parquet'
        if not meta_path.exists() or not codes_path.exists() or not item_ids_path.exists():
            raise FileNotFoundError(
                f'Quantized export not found under {export_dir}. '
                f'Run quantizer first.'
            )
        self._log(f'loading quantized export from {export_dir}')
        meta = json.loads(meta_path.read_text())
        codes = np.load(codes_path)
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
        self._log(f'loaded quantized export rows={len(item_ids)} shape={list(codes.shape)}')
        return export_dir, meta, item_ids, codes

    def load_sid_view(self, build_only_meta=False):
        _, meta, item_ids, codes = self._load_quantized_export()
        num_quantizers = int(codes.shape[1]) if codes.ndim > 1 else 1
        codebook_size = int(meta['quantizer_config']['codebook_size'])

        if build_only_meta:
            return {
                'num_quantizers': num_quantizers,
                'codebook_size': codebook_size,
            }

        sid_map = {}
        for item_id, row in tqdm(
                zip(item_ids, codes),
                total=len(item_ids),
                desc='sid map',
                leave=False,
        ):
            row = np.atleast_1d(row).tolist()
            sid_map[item_id] = [index * codebook_size + int(code) for index, code in enumerate(row)]

        missing = []
        ordered_values = []
        for item_id in tqdm(self.uid_raw_items, desc='sid align', leave=False):
            if item_id not in sid_map:
                missing.append(item_id)
                continue
            ordered_values.append(sid_map[item_id])
        if missing:
            raise ValueError(f'{len(missing)} items missing sid codes, first missing item: {missing[0]}')

        return ordered_values

    def load_embedding_view(self):
        model_name = _normalize_model_name(self.config.repr_model)
        embedding_dir = self.store.embedded_dir(model_name)
        item_ids_path = embedding_dir / 'item_ids.parquet'
        if not item_ids_path.exists():
            raise FileNotFoundError(
                f'Embedding item ids not found under {embedding_dir}. '
                f'Run embedder first.'
            )
        self._log(f'loading embedding index mapping from {embedding_dir}')
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
        embedding_index_map = {item_id: index for index, item_id in enumerate(item_ids)}
        missing = []
        ordered_values = []
        for item_id in tqdm(self.uid_raw_items, desc='embedding align', leave=False):
            if item_id not in embedding_index_map:
                missing.append(item_id)
                continue
            ordered_values.append(int(embedding_index_map[item_id]))
        if missing:
            raise ValueError(f'{len(missing)} items missing embeddings, first missing item: {missing[0]}')
        return ordered_values

    @staticmethod
    def _as_token_list(value):
        if isinstance(value, list):
            return value
        return [value]

    def _get_repr_view_value(self, repr_type: str, uid: int):
        return self.item_views[repr_type][uid]

    def _compose_history_item(self, uid: int):
        if self.config.repr_combine == 'add':
            return self.item_views['uid'][uid]

        tokens = []
        for repr_type in self.config.repr_types:
            tokens.extend(self._as_token_list(self._get_repr_view_value(repr_type, uid)))
        return tokens

    def _history_values(self, history_uids: list[int]):
        return [self._compose_history_item(uid) for uid in history_uids]

    def _target_value(self, target_uid: int):
        if self.config.task_type == 'uid':
            return self.item_views['uid'][target_uid]
        return self.item_views[self.config.task_type][target_uid]

    def _build_usable_history(self, prefix_uids: list[int], target_uid: int):
        if self.config.maxitems > 0:
            prefix_uids = prefix_uids[-self.config.maxitems:]

        target_value = self._target_value(target_uid)
        best_history = []
        best_total_length = None

        for start_index in range(len(prefix_uids) - 1, -1, -1):
            candidate_history = prefix_uids[start_index:]
            candidate_values = self._history_values(candidate_history)
            total_input_length = self.model_adapter.estimate_main_length(
                candidate_values,
                target_value,
                self.config.task_type,
            )
            if total_input_length <= self.model_adapter.max_length:
                best_history = candidate_history
                best_total_length = total_input_length
            else:
                break

        if best_total_length is None:
            return None, None
        return best_history, int(best_total_length)

    def build_samples(self, split_name: str, dataframe: pd.DataFrame):
        self._log(
            f'building {split_name} samples from {len(dataframe)} user sequences '
            f'(task={self.config.task_type}, repr={self.config.repr_type})'
        )
        columns = [
            self.processor.UID_COL,
            'history_uids',
            'target_uid',
            'history_item_count',
            'total_input_length',
            'target_pos',
        ]
        rows = []
        resolved_maxitems = 0
        invalid_target_count = 0
        dropped_short_sequence_count = 0
        total_candidate_targets = 0

        iterator = tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f'{split_name} users',
            leave=False,
        )
        for _, row in iterator:
            sequence = [self.uid_item_map[item_id] for item_id in row[self.processor.HIS_COL] if item_id in self.uid_item_map]
            if len(sequence) < 2:
                dropped_short_sequence_count += 1
                iterator.set_postfix(samples=len(rows), invalid=invalid_target_count, dropped=dropped_short_sequence_count)
                continue

            target_positions = range(1, len(sequence)) if split_name == 'finetune' else [len(sequence) - 1]
            total_candidate_targets += len(target_positions)
            for target_pos in target_positions:
                prefix_uids = sequence[:target_pos]
                target_uid = sequence[target_pos]
                history_uids, total_input_length = self._build_usable_history(prefix_uids, target_uid)
                if not history_uids:
                    invalid_target_count += 1
                    continue

                rows.append(
                    {
                        self.processor.UID_COL: row[self.processor.UID_COL],
                        'history_uids': history_uids,
                        'target_uid': int(target_uid),
                        'history_item_count': int(len(history_uids)),
                        'total_input_length': int(total_input_length),
                        'target_pos': int(target_pos),
                    }
                )
                resolved_maxitems = max(resolved_maxitems, len(history_uids))
                if split_name not in self.sample_visuals:
                    self.sample_visuals[split_name] = self._render_sample_visual(
                        split_name=split_name,
                        user_id=row[self.processor.UID_COL],
                        sequence=sequence,
                        history_uids=history_uids,
                        target_uid=target_uid,
                        target_pos=target_pos,
                        total_input_length=total_input_length,
                    )
            iterator.set_postfix(
                samples=len(rows),
                invalid=invalid_target_count,
                maxhist=resolved_maxitems,
            )

        output_path = self.samples_dir / f'{split_name}.parquet'
        pd.DataFrame(rows, columns=columns).to_parquet(output_path, index=False)
        self.samples_stats[split_name] = {
            'sample_count': len(rows),
            'resolved_maxitems': int(resolved_maxitems),
            'invalid_target_count': int(invalid_target_count),
            'dropped_short_sequence_count': int(dropped_short_sequence_count),
            'candidate_target_count': int(total_candidate_targets),
        }
        avg_history_items = float(np.mean([row['history_item_count'] for row in rows])) if rows else 0.0
        avg_input_length = float(np.mean([row['total_input_length'] for row in rows])) if rows else 0.0
        self._log(
            f'{split_name} samples ready count={len(rows)}/{total_candidate_targets} '
            f'invalid={invalid_target_count} dropped_short_seq={dropped_short_sequence_count} '
            f'avg_history_items={avg_history_items:.2f} avg_input_length={avg_input_length:.2f} '
            f'resolved_maxitems={resolved_maxitems} saved={output_path}'
        )

    def build_alignment_meta(self):
        views = set()
        if self.config.alignment:
            views.update({self.config.repr_type, self.config.task_type, 'text'})
        available_views = sorted(view for view in views if view in self.item_views)
        pairs = [list(pair) for pair in combinations(available_views, 2)]
        self._save_json(
            self.alignment_dir / 'meta.json',
            {
                'enabled': bool(self.config.alignment),
                'views': available_views,
                'pairs': pairs,
            },
        )
        self._log(
            f'alignment meta saved enabled={self.config.alignment} '
            f'views={available_views} pairs={pairs}'
        )

    def save_meta(self):
        self._save_json(
            self.meta_path,
            {
                'version': self.VER,
                'stage': 'compiled',
                'data': self.config.data,
                'prepare_id': self.config.prepare_id,
                'config': self.config.config_dict,
                'model_kind': self.model_adapter.kind,
                'model_max_length': int(self.model_adapter.max_length),
                'processed_dir': str(self.store.processed_dir()),
            },
        )
        self._log(f'meta saved to {self.meta_path}')

    def save_stats(self):
        stats = {
            'item_count': len(self.uid_raw_items),
            'finetune_sample_count': self.samples_stats['finetune']['sample_count'],
            'test_sample_count': self.samples_stats['test']['sample_count'],
            'finetune_invalid_target_count': self.samples_stats['finetune']['invalid_target_count'],
            'test_invalid_target_count': self.samples_stats['test']['invalid_target_count'],
            'resolved_maxitems': max(
                self.samples_stats['finetune']['resolved_maxitems'],
                self.samples_stats['test']['resolved_maxitems'],
            ),
        }
        stats_path = self.output_dir / 'stats.json'
        self._save_json(stats_path, stats)
        self._log(
            f"stats saved to {stats_path}: item_count={stats['item_count']} "
            f"finetune_samples={stats['finetune_sample_count']} test_samples={stats['test_sample_count']} "
            f"resolved_maxitems={stats['resolved_maxitems']}"
        )

    def log_sample_visuals(self):
        for split_name in ['finetune', 'test']:
            visual = self.sample_visuals.get(split_name)
            if visual:
                self._log(f'{split_name} sample visualization:\n{visual}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile processed data into trainer-ready dataset assets.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--model', required=True, help='Backbone model name, such as llama3 or transformer.')
    parser.add_argument('--repr.type', dest='repr_type', required=True, help='Representation types, such as uid, text, or uid+text.')
    parser.add_argument('--repr.model', dest='repr_model', default=None, help='External representation model, such as bertbase.')
    parser.add_argument('--repr.best', dest='repr_best', default=None, help='Best checkpoint metric for quantized codes, such as coll.')
    parser.add_argument('--repr.combine', dest='repr_combine', default='concat', help='How to combine multiple repr types: concat or add.')
    parser.add_argument('--task.type', dest='task_type', required=True, choices=['uid', 'sid', 'embedding'])
    parser.add_argument('--maxitems', type=int, default=0, help='Maximum history items, 0 means auto by model max length.')
    parser.add_argument('--item-text-max-tokens', type=int, default=50, help='Maximum tokenized length per item text.')
    parser.add_argument('--task.alignment', dest='alignment', default='true', help='Whether to compile alignment assets.')
    args = parser.parse_args()

    config = CompileConfig(
        data=args.data.lower(),
        model=args.model.lower(),
        repr_type=args.repr_type.lower(),
        repr_model=_normalize_model_name(args.repr_model),
        repr_best=args.repr_best.lower() if args.repr_best else None,
        repr_combine=args.repr_combine.lower(),
        task_type=args.task_type.lower(),
        maxitems=int(args.maxitems),
        item_text_max_tokens=int(args.item_text_max_tokens),
        alignment=str(args.alignment).lower() != 'false',
    )
    compiler = Compiler(config)
    compiler.run()
