import argparse
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pigmento import pnt
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

    @property
    def prepare_id(self):
        parts = [
            f'model-{self.model}',
            f'repr-{self.repr_type}',
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
    VER = 'v1.1'
    SUPPORTED_REPR_TYPES = {'uid', 'sid', 'text', 'embedding'}
    SUPPORTED_TASK_TYPES = {'uid', 'sid', 'embedding'}

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

        self._init_dirs()

    def _init_dirs(self):
        self.vocab_dir = self.output_dir / 'vocab'
        self.prompts_dir = self.output_dir / 'prompts'
        self.item_views_dir = self.output_dir / 'item_views'
        self.samples_dir = self.output_dir / 'samples'
        self.alignment_dir = self.output_dir / 'alignment'
        for path in [self.vocab_dir, self.prompts_dir, self.item_views_dir, self.samples_dir, self.alignment_dir]:
            path.mkdir(parents=True, exist_ok=True)

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
        if self.config.repr_type not in self.SUPPORTED_REPR_TYPES:
            raise ValueError(f'Unsupported repr.type: {self.config.repr_type}')
        if self.config.task_type not in self.SUPPORTED_TASK_TYPES:
            raise ValueError(f'Unsupported task.type: {self.config.task_type}')
        external_view_required = any(view in {'sid', 'embedding'} for view in [self.config.repr_type, self.config.task_type])
        if external_view_required and not self.config.repr_model:
            raise ValueError('repr.model is required when repr.type or task.type uses sid/embedding')
        if 'sid' in {self.config.repr_type, self.config.task_type} and not self.config.repr_best:
            raise ValueError('repr.best is required when repr.type or task.type uses sid')

    def run(self):
        self.validate()
        if self.is_cached():
            pnt(f'compiled dataset cached at {self.output_dir}')
            return

        self.load_processor()
        self.init_model_adapter()
        self.build_vocab_and_prompts()
        self.build_item_views()
        self.build_samples('finetune', self.processor.finetune_set)
        self.build_samples('test', self.processor.test_set)
        self.build_alignment_meta()
        self.save_meta()
        self.save_stats()

    def load_processor(self):
        self.processor = load_processor(self.config.data)
        self.processor.load()
        self.uid_raw_items = self.processor.items[self.processor.IID_COL].tolist()
        self.uid_item_map = {item_id: index for index, item_id in enumerate(self.uid_raw_items)}
        self.item_texts = [
            self.processor.organize_item(item_id, item_attrs=self.processor.default_attrs) or '[Empty Content]'
            for item_id in self.uid_raw_items
        ]

    def init_model_adapter(self):
        model_key = model_utils.match(self.config.model)
        if model_key:
            self.model_adapter = LLMAdapter(self.config.model, model_key)
            return
        self.model_adapter = ScratchAdapter(self.config.model, self.item_texts)

    def _save_json(self, path: Path, data):
        path.write_text(json.dumps(data, indent=2) + '\n')

    def _write_view(self, name: str, values: list):
        path = self.item_views_dir / f'{name}.parquet'
        pd.DataFrame({'value': values}).to_parquet(path, index=False)
        self.item_views[name] = values
        return path

    def build_vocab_and_prompts(self):
        model_vocab_path = self.vocab_dir / 'model.json'
        self._save_json(model_vocab_path, self.model_adapter.build_vocab_artifact())
        self.registry.register(
            self.model_adapter.namespace_name,
            kind='model',
            size=len(self.model_adapter.build_vocab_artifact().get('tokens', [])),
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

        self._save_json(self.vocab_dir / 'meta.json', self.registry.to_dict())
        self._save_json(self.prompts_dir / 'main.json', self.model_adapter.build_prompt_spec())
        self._save_json(self.prompts_dir / 'alignment.json', self.model_adapter.build_alignment_spec())

    def requires_view(self, view_name: str):
        views = {'uid', self.config.repr_type, self.config.task_type}
        if self.config.alignment and ('text' not in views or len(views) > 1):
            views.add('text')
        return view_name in views

    def build_item_views(self):
        self._write_view('uid', list(range(len(self.uid_raw_items))))

        if self.requires_view('text'):
            text_values = self.model_adapter.tokenize_texts(
                self.item_texts,
                max_tokens=self.config.item_text_max_tokens,
            )
            self._write_view('text', text_values)

        if self.requires_view('sid'):
            sid_values = self.load_sid_view()
            self._write_view('sid', sid_values)

        if self.requires_view('embedding'):
            embedding_values = self.load_embedding_view()
            self._write_view('embedding', embedding_values)

        self._save_json(
            self.item_views_dir / 'meta.json',
            {
                'row_order': 'uid_vocab',
                'views': sorted(self.item_views),
            },
        )

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
        meta = json.loads(meta_path.read_text())
        codes = np.load(codes_path)
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
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
        for item_id, row in zip(item_ids, codes):
            row = np.atleast_1d(row).tolist()
            sid_map[item_id] = [index * codebook_size + int(code) for index, code in enumerate(row)]

        missing = [item_id for item_id in self.uid_raw_items if item_id not in sid_map]
        if missing:
            raise ValueError(f'{len(missing)} items missing sid codes, first missing item: {missing[0]}')

        return [sid_map[item_id] for item_id in self.uid_raw_items]

    def load_embedding_view(self):
        model_name = _normalize_model_name(self.config.repr_model)
        embedding_dir = self.store.embedded_dir(model_name)
        item_ids_path = embedding_dir / 'item_ids.parquet'
        if not item_ids_path.exists():
            raise FileNotFoundError(
                f'Embedding item ids not found under {embedding_dir}. '
                f'Run embedder first.'
            )
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
        embedding_index_map = {item_id: index for index, item_id in enumerate(item_ids)}
        missing = [item_id for item_id in self.uid_raw_items if item_id not in embedding_index_map]
        if missing:
            raise ValueError(f'{len(missing)} items missing embeddings, first missing item: {missing[0]}')
        return [int(embedding_index_map[item_id]) for item_id in self.uid_raw_items]

    def _history_values(self, history_uids: list[int]):
        if self.config.repr_type == 'uid':
            return [self.item_views['uid'][uid] for uid in history_uids]
        return [self.item_views[self.config.repr_type][uid] for uid in history_uids]

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

        for _, row in dataframe.iterrows():
            sequence = [self.uid_item_map[item_id] for item_id in row[self.processor.HIS_COL] if item_id in self.uid_item_map]
            if len(sequence) < 2:
                dropped_short_sequence_count += 1
                continue

            target_positions = range(1, len(sequence)) if split_name == 'finetune' else [len(sequence) - 1]
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

        output_path = self.samples_dir / f'{split_name}.parquet'
        pd.DataFrame(rows, columns=columns).to_parquet(output_path, index=False)
        self.samples_stats[split_name] = {
            'sample_count': len(rows),
            'resolved_maxitems': int(resolved_maxitems),
            'invalid_target_count': int(invalid_target_count),
            'dropped_short_sequence_count': int(dropped_short_sequence_count),
        }
        pnt(f'compiled {len(rows)} {split_name} samples')

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

    def save_stats(self):
        self._save_json(
            self.output_dir / 'stats.json',
            {
                'item_count': len(self.uid_raw_items),
                'finetune_sample_count': self.samples_stats['finetune']['sample_count'],
                'test_sample_count': self.samples_stats['test']['sample_count'],
                'finetune_invalid_target_count': self.samples_stats['finetune']['invalid_target_count'],
                'test_invalid_target_count': self.samples_stats['test']['invalid_target_count'],
                'resolved_maxitems': max(
                    self.samples_stats['finetune']['resolved_maxitems'],
                    self.samples_stats['test']['resolved_maxitems'],
                ),
            },
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile processed data into trainer-ready dataset assets.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--model', required=True, help='Backbone model name, such as llama3 or transformer.')
    parser.add_argument('--repr.type', dest='repr_type', required=True, choices=['uid', 'sid', 'text', 'embedding'])
    parser.add_argument('--repr.model', dest='repr_model', default=None, help='External representation model, such as bertbase.')
    parser.add_argument('--repr.best', dest='repr_best', default=None, help='Best checkpoint metric for quantized codes, such as coll.')
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
        task_type=args.task_type.lower(),
        maxitems=int(args.maxitems),
        item_text_max_tokens=int(args.item_text_max_tokens),
        alignment=str(args.alignment).lower() != 'false',
    )
    compiler = Compiler(config)
    compiler.run()
