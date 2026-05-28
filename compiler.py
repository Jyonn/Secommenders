import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from models import BaseBackbone, build_backbone
from models.base import TYPE_MARKER_ORDER, TYPE_MARKER_TOKENS
from processors.base_processor import Processor
from utils.artifact import ArtifactStore
from utils.compile import CompileConfig, normalize_model_name
from utils.function import load_processor
from utils.logging import setup_logging
from utils.pipeline import ensure_embedded, ensure_quantized
from utils import model as model_utils

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


class Compiler:
    VER = 'v2.4'
    SUPPORTED_REPR_TYPES = {'uid', 'sid', 'text', 'embedding'}
    SUPPORTED_TASK_TYPES = {'uid', 'sid', 'embedding'}
    SUPPORTED_REPR_COMBINES = {'concat', 'add'}

    def __init__(self, config: CompileConfig):
        self.config = config
        self.store = ArtifactStore(config.data)
        self.output_dir = self.store.compiled_dir(config.prepare_id)
        self.processor: Optional[Processor] = None
        self.backbone: Optional[BaseBackbone] = None
        self.registry = VocabularyRegistry()
        self.uid_raw_items: list = []
        self.uid_item_map = {}
        self.item_texts: list[str] = []
        self.item_views = {}
        self.samples_stats = {}
        self.sample_visuals = {}
        self.sid_stats = {}

        self._init_dirs()

    def _init_dirs(self):
        self.vocab_dir = self.output_dir / 'vocab'
        self.prompts_dir = self.output_dir / 'prompts'
        self.item_views_dir = self.output_dir / 'item_views'
        self.samples_dir = self.output_dir / 'samples'
        self.alignment_dir = self.output_dir / 'alignment'
        for path in [self.vocab_dir, self.prompts_dir, self.item_views_dir, self.samples_dir, self.alignment_dir]:
            path.mkdir(parents=True, exist_ok=True)

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
            f'| input_len={total_input_length}/{self.backbone.max_length}'
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
            lines.append(
                '  policy    : finetune uses one packed suffix per user; each item block starts with task repr, '
                'next-item supervision is standard causal prediction, and later reprs in the same block serve as alignment targets'
            )
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
        if meta.get('version') != self.VER:
            return False
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
            self.vocab_dir / 'special.json',
            self.vocab_dir / 'meta.json',
            self.prompts_dir / 'main.json',
            self.prompts_dir / 'alignment.json',
            self.item_views_dir / 'meta.json',
            self.samples_dir / 'finetune.parquet',
            self.samples_dir / 'valid.parquet',
            self.samples_dir / 'test.parquet',
            self.alignment_dir / 'meta.json',
        ] + required_item_view_paths
        return all(path.exists() for path in required_paths)

    def validate(self):
        repr_types = self.config.repr_types
        is_scratch_model = model_utils.match(self.config.model) is None
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
        if self.config.task_type not in repr_types:
            raise ValueError('repr.type must contain task.type so each item block starts with task representation')
        if repr_types[0] != self.config.task_type:
            raise ValueError('task.type must be the first entry in repr.type for causal mixed-view training')
        if self.config.repr_combine == 'add':
            raise ValueError('repr.combine=add is not supported by the mixed-view causal training protocol')
        if is_scratch_model and 'text' in repr_types:
            raise ValueError('scratch backbone currently does not support repr.type containing text')

        external_view_required = any(view in {'sid', 'embedding'} for view in repr_types + [self.config.task_type])
        if external_view_required and not self.config.repr_model:
            raise ValueError('repr.model is required when repr.type or task.type uses sid/embedding')
        if 'sid' in set(repr_types + [self.config.task_type]) and not self.config.repr_best:
            raise ValueError('repr.best is required when repr.type or task.type uses sid')

    def run(self):
        self.validate()
        pnt(
            f'start compile data={self.config.data} model={self.config.model} '
            f'repr={self.config.repr_type} combine={self.config.repr_combine} '
            f'task={self.config.task_type} maxitems={self.config.maxitems} '
            f'textlen={self.config.item_text_max_tokens} alignment=always '
            f'model.maxlen={self.config.model_max_length or "native"}'
        )
        if self.is_cached():
            pnt(f'compiled dataset cached at {self.output_dir}')
            return

        pnt(f'cache miss, compiling into {self.output_dir}')
        self.load_processor()
        self.init_backbone()
        self.build_vocab_and_prompts()
        self.build_item_views()
        self.build_samples('finetune', self.processor.finetune_set)
        self.build_samples('valid', self.processor.valid_set)
        self.build_samples('test', self.processor.test_set)
        self.build_alignment_meta()
        self.save_meta()
        self.save_stats()
        self.log_sample_visuals()
        pnt(f'compile finished: outputs written to {self.output_dir}')

    def load_processor(self):
        pnt(f'loading processed dataset {self.config.data}')
        self.processor = load_processor(self.config.data)
        self.processor.load()
        self.uid_raw_items = self.processor.items[self.processor.IID_COL].tolist()
        self.uid_item_map = {item_id: index for index, item_id in enumerate(self.uid_raw_items)}
        self.item_texts = [
            self.processor.organize_item(item_id, item_attrs=self.processor.default_attrs) or '[Empty Content]'
            for item_id in self.uid_raw_items
        ]
        pnt(
            f'loaded processed assets: items={len(self.uid_raw_items)} '
            f'finetune_users={len(self.processor.finetune_set)} test_users={len(self.processor.test_set)}'
        )

    def init_backbone(self):
        self.backbone = build_backbone(
            self.config.model,
            self.item_texts,
            max_length_override=self.config.model_max_length,
        )
        pnt(
            f'initialized backbone model={self.config.model} kind={self.backbone.kind} '
            f'native_maxlen={self.backbone.native_max_length} effective_maxlen={self.backbone.max_length}'
        )
        if getattr(self.backbone.tokenizer, 'tokens', None) is not None:
            pnt(f'scratch backbone vocab_size={len(self.backbone.tokenizer.tokens)}')
        if getattr(self.backbone, 'model_key', None) is not None:
            pnt(f'llm backbone hf_key={self.backbone.model_key}')

    def _save_json(self, path: Path, data):
        path.write_text(json.dumps(data, indent=2) + '\n')

    def _write_view(self, name: str, values: list):
        path = self.item_views_dir / f'{name}.parquet'
        pd.DataFrame({'value': values}).to_parquet(path, index=False)
        self.item_views[name] = values
        return path

    def build_vocab_and_prompts(self):
        pnt('building vocab registry and prompt assets')
        model_vocab_path = self.vocab_dir / 'model.json'
        model_vocab_artifact = self.backbone.build_vocab_artifact()
        self._save_json(model_vocab_path, model_vocab_artifact)
        self.registry.register(
            self.backbone.namespace_name,
            kind='model',
            size=len(model_vocab_artifact.get('tokens', [])),
            path=model_vocab_path,
            model_kind=self.backbone.kind,
        )

        uid_vocab_path = self.vocab_dir / 'uid.json'
        self._save_json(uid_vocab_path, {'raw_item_ids': self.uid_raw_items})
        self.registry.register(
            'uid',
            kind='uid',
            size=len(self.uid_raw_items),
            path=uid_vocab_path,
        )
        special_vocab_path = self.vocab_dir / 'special.json'
        self._save_json(
            special_vocab_path,
            {
                'tokens': [TYPE_MARKER_TOKENS[name] for name in TYPE_MARKER_ORDER],
                'marker_to_index': {name: index for index, name in enumerate(TYPE_MARKER_ORDER)},
                'external_ids': {name: -1 for name in TYPE_MARKER_ORDER},
            },
        )
        self.registry.register(
            'special',
            kind='special',
            size=len(TYPE_MARKER_ORDER),
            path=special_vocab_path,
        )

        if self.requires_view('sid'):
            sid_meta = self.load_sid_view(build_only_meta=True)
            sid_vocab_path = self.vocab_dir / 'sid.json'
            tokens = [
                f'q{quantizer}_c{code}'
                for quantizer in range(sid_meta['num_quantizers'])
                for code in range(sid_meta['codebook_size'])
            ] + [
                f'collision_{index}'
                for index in range(sid_meta['collision_vocab_size'])
            ]
            self._save_json(
                sid_vocab_path,
                {
                    'tokens': tokens,
                    'num_quantizers': sid_meta['final_num_quantizers'],
                    'base_num_quantizers': sid_meta['num_quantizers'],
                    'codebook_size': sid_meta['codebook_size'],
                    'collision_vocab_size': sid_meta['collision_vocab_size'],
                    'collision_token_offset': sid_meta['collision_token_offset'],
                    'collision_group_count': sid_meta['collision_group_count'],
                    'collided_item_count': sid_meta['collided_item_count'],
                    'max_collision_size': sid_meta['max_collision_size'],
                },
            )
            self.registry.register(
                'sid',
                kind='sid',
                size=len(tokens),
                path=sid_vocab_path,
                num_quantizers=sid_meta['final_num_quantizers'],
                base_num_quantizers=sid_meta['num_quantizers'],
                codebook_size=sid_meta['codebook_size'],
                collision_vocab_size=sid_meta['collision_vocab_size'],
            )
            pnt(
                f"registered sid vocab size={len(tokens)} "
                f"base_num_quantizers={sid_meta['num_quantizers']} "
                f"final_num_quantizers={sid_meta['final_num_quantizers']} "
                f"codebook_size={sid_meta['codebook_size']} "
                f"collision_vocab_size={sid_meta['collision_vocab_size']} "
                f"max_collision_size={sid_meta['max_collision_size']}"
            )

        self._save_json(self.vocab_dir / 'meta.json', self.registry.to_dict())
        main_prompt = self.backbone.build_prompt_spec()
        alignment_prompt = self.backbone.build_alignment_spec()
        self._save_json(self.prompts_dir / 'main.json', main_prompt)
        self._save_json(self.prompts_dir / 'alignment.json', alignment_prompt)
        pnt(
            f'vocab ready namespaces={len(self.registry.entries)} '
            f'main_prompt=(history_prefix={len(main_prompt["history_prefix_ids"])}, '
            f'separator={len(main_prompt["item_separator_ids"])}, '
            f'query_prefix={len(main_prompt["query_prefix_ids"])})'
        )

    def requires_view(self, view_name: str):
        views = {'uid', *self.config.repr_types, self.config.task_type}
        return view_name in views

    def build_item_views(self):
        required_views = [view for view in ['uid', 'text', 'sid', 'embedding'] if self.requires_view(view)]
        pnt(f'building item views {required_views} for {len(self.uid_raw_items)} items')
        self._write_view('uid', list(range(len(self.uid_raw_items))))
        pnt('uid view ready')

        if self.requires_view('text'):
            pnt(
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
                    self.backbone.tokenize_texts(
                        batch_texts,
                        max_tokens=self.config.item_text_max_tokens,
                    )
                )
            self._write_view('text', text_values)
            text_lengths = [len(value) for value in text_values]
            pnt(
                f'text view ready avg_len={np.mean(text_lengths):.2f} '
                f'max_len={max(text_lengths) if text_lengths else 0}'
            )

        if self.requires_view('sid'):
            pnt(
                f'loading sid view from model={self.config.repr_model} '
                f'checkpoint={self.config.repr_best}'
            )
            sid_values = self.load_sid_view()
            self._write_view('sid', sid_values)
            sid_lengths = [len(value) for value in sid_values]
            pnt(
                f'sid view ready avg_codes={np.mean(sid_lengths):.2f} '
                f'max_codes={max(sid_lengths) if sid_lengths else 0}'
            )

        if self.requires_view('embedding'):
            pnt(f'loading embedding view from model={self.config.repr_model}')
            embedding_values = self.load_embedding_view()
            self._write_view('embedding', embedding_values)
            pnt(
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
        pnt(f'item view manifest saved: {sorted(self.item_views)}')

    def _load_quantized_export(self):
        model_name = normalize_model_name(self.config.repr_model)
        export_dir = self.store.quantized_dir(model_name) / 'exports' / self.config.repr_best
        meta_path = export_dir / 'meta.json'
        codes_path = export_dir / 'codebook_indices.npy'
        item_ids_path = export_dir / 'item_ids.parquet'
        if not meta_path.exists() or not codes_path.exists() or not item_ids_path.exists():
            ensure_quantized(self.config.data, model_name)
        if not meta_path.exists() or not codes_path.exists() or not item_ids_path.exists():
            raise FileNotFoundError(f'Quantized export not found after auto preparation: {export_dir}')
        pnt(f'loading quantized export from {export_dir}')
        meta = json.loads(meta_path.read_text())
        codes = np.load(codes_path)
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
        pnt(f'loaded quantized export rows={len(item_ids)} shape={list(codes.shape)}')
        return export_dir, meta, item_ids, codes

    def load_sid_view(self, build_only_meta=False):
        _, meta, item_ids, codes = self._load_quantized_export()
        num_quantizers = int(codes.shape[1]) if codes.ndim > 1 else 1
        codebook_size = int(meta['quantizer_config']['codebook_size'])
        collision_token_offset = num_quantizers * codebook_size

        base_sid_groups = {}
        for item_id, row in zip(item_ids, codes):
            row = np.atleast_1d(row).tolist()
            base_sid = tuple(index * codebook_size + int(code) for index, code in enumerate(row))
            base_sid_groups.setdefault(base_sid, []).append(item_id)

        max_collision_size = max((len(group) for group in base_sid_groups.values()), default=1)
        collision_vocab_size = max_collision_size
        final_num_quantizers = num_quantizers + 1
        collision_group_count = sum(1 for group in base_sid_groups.values() if len(group) > 1)
        collided_item_count = sum(len(group) for group in base_sid_groups.values() if len(group) > 1)

        self.sid_stats = {
            'base_num_quantizers': num_quantizers,
            'final_num_quantizers': final_num_quantizers,
            'codebook_size': codebook_size,
            'collision_vocab_size': collision_vocab_size,
            'collision_token_offset': collision_token_offset,
            'collision_group_count': collision_group_count,
            'collided_item_count': collided_item_count,
            'max_collision_size': max_collision_size,
        }

        if build_only_meta:
            return {
                'num_quantizers': num_quantizers,
                'codebook_size': codebook_size,
                'final_num_quantizers': final_num_quantizers,
                'collision_vocab_size': collision_vocab_size,
                'collision_token_offset': collision_token_offset,
                'collision_group_count': collision_group_count,
                'collided_item_count': collided_item_count,
                'max_collision_size': max_collision_size,
            }

        sid_map = {}
        for base_sid, grouped_item_ids in tqdm(base_sid_groups.items(), total=len(base_sid_groups), desc='sid map', leave=False):
            ordered_item_ids = sorted(grouped_item_ids, key=lambda value: str(value))
            for collision_index, item_id in enumerate(ordered_item_ids):
                sid_map[item_id] = list(base_sid) + [collision_token_offset + collision_index]

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
        model_name = normalize_model_name(self.config.repr_model)
        embedding_dir = self.store.embedded_dir(model_name)
        item_ids_path = embedding_dir / 'item_ids.parquet'
        if not item_ids_path.exists():
            ensure_embedded(self.config.data, model_name)
        if not item_ids_path.exists():
            raise FileNotFoundError(f'Embedding item ids not found after auto preparation under {embedding_dir}.')
        pnt(f'loading embedding index mapping from {embedding_dir}')
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

    def _type_marker_key(self, repr_type: str | None = None):
        if repr_type is not None:
            return repr_type
        if self.config.repr_combine == 'add':
            return 'uid+embedding'
        return None

    def _history_item_length(self, uid: int):
        if self.config.repr_combine == 'add':
            return 2

        total = 0
        for repr_type in self.config.repr_types:
            total += 1
            value = self._get_repr_view_value(repr_type, uid)
            total += len(value) if isinstance(value, list) else 1
        return total

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

    def _repr_segment_length(self, repr_type: str, uid: int):
        value = self._get_repr_view_value(repr_type, uid)
        payload_length = len(value) if isinstance(value, list) else 1
        return 1 + payload_length

    def _finetune_item_length(self, uid: int, include_non_task: bool):
        total = self._repr_segment_length(self.config.task_type, uid)
        if include_non_task:
            for repr_type in self.config.repr_types[1:]:
                total += self._repr_segment_length(repr_type, uid)
        return total

    def _estimate_packed_length(self, sequence_uids: list[int]):
        if len(sequence_uids) < 2:
            return None
        prompt = self.backbone.build_prompt_spec()
        separator_len = len(prompt['item_separator_ids']) * max(0, len(sequence_uids) - 1)
        total = separator_len
        for index, uid in enumerate(sequence_uids):
            include_non_task = index < len(sequence_uids) - 1
            total += self._finetune_item_length(uid, include_non_task=include_non_task)
        return total

    def _build_usable_sequence(self, sequence_uids: list[int]):
        if self.config.maxitems > 0:
            sequence_uids = sequence_uids[-self.config.maxitems:]
        if len(sequence_uids) < 2:
            return None, None

        best_sequence = None
        best_total_length = None
        for start_index in range(len(sequence_uids) - 2, -1, -1):
            candidate_sequence = sequence_uids[start_index:]
            total_input_length = self._estimate_packed_length(candidate_sequence)
            if total_input_length is None:
                continue
            if total_input_length <= self.backbone.max_length:
                best_sequence = candidate_sequence
                best_total_length = total_input_length
            else:
                break

        return best_sequence, best_total_length

    def _build_usable_history(self, prefix_uids: list[int], target_uid: int):
        if self.config.maxitems > 0:
            prefix_uids = prefix_uids[-self.config.maxitems:]

        prompt = self.backbone.build_prompt_spec()
        best_history = []
        best_total_length: int = None

        for start_index in range(len(prefix_uids) - 1, -1, -1):
            candidate_history = prefix_uids[start_index:]
            history_len = sum(self._history_item_length(uid) for uid in candidate_history)
            separator_len = len(prompt['item_separator_ids']) * max(0, len(candidate_history) - 1)
            target_value = self._target_value(target_uid)
            if self.config.task_type == 'embedding':
                target_len = 0
            elif isinstance(target_value, list):
                target_len = len(target_value)
            else:
                target_len = 1
            total_input_length = (
                len(prompt['history_prefix_ids'])
                + history_len
                + separator_len
                + len(prompt['query_prefix_ids'])
                + 1
                + target_len
            )
            if total_input_length <= self.backbone.max_length:
                best_history = candidate_history
                best_total_length = total_input_length
            else:
                break

        if best_total_length is None:
            return None, None
        return best_history, best_total_length

    def build_samples(self, split_name: str, dataframe: pd.DataFrame):
        pnt(
            f'building {split_name} samples from {len(dataframe)} user sequences '
            f'(task={self.config.task_type}, repr={self.config.repr_type})'
        )
        if split_name == 'finetune':
            columns = [
                self.processor.UID_COL,
                'sequence_uids',
                'sequence_item_count',
                'prediction_count',
                'total_input_length',
            ]
        else:
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

            total_candidate_targets += 1
            if split_name == 'finetune':
                usable_sequence, total_input_length = self._build_usable_sequence(sequence)
                if not usable_sequence:
                    invalid_target_count += 1
                    iterator.set_postfix(samples=len(rows), invalid=invalid_target_count, maxhist=resolved_maxitems)
                    continue

                rows.append(
                    {
                        self.processor.UID_COL: row[self.processor.UID_COL],
                        'sequence_uids': usable_sequence,
                        'sequence_item_count': int(len(usable_sequence)),
                        'prediction_count': int(len(usable_sequence) - 1),
                        'total_input_length': int(total_input_length),
                    }
                )
                resolved_maxitems = max(resolved_maxitems, len(usable_sequence))
                if split_name not in self.sample_visuals:
                    target_pos = len(usable_sequence) - 1
                    self.sample_visuals[split_name] = self._render_sample_visual(
                        split_name=split_name,
                        user_id=row[self.processor.UID_COL],
                        sequence=usable_sequence,
                        history_uids=usable_sequence[:-1],
                        target_uid=usable_sequence[-1],
                        target_pos=target_pos,
                        total_input_length=total_input_length,
                    )
            else:
                target_pos = len(sequence) - 1
                prefix_uids = sequence[:target_pos]
                target_uid = sequence[target_pos]
                history_uids, total_input_length = self._build_usable_history(prefix_uids, target_uid)
                if not history_uids:
                    invalid_target_count += 1
                    iterator.set_postfix(samples=len(rows), invalid=invalid_target_count, maxhist=resolved_maxitems)
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
        if split_name == 'finetune':
            avg_item_count = float(np.mean([row['sequence_item_count'] for row in rows])) if rows else 0.0
            avg_item_label = 'avg_sequence_items'
        else:
            avg_item_count = float(np.mean([row['history_item_count'] for row in rows])) if rows else 0.0
            avg_item_label = 'avg_history_items'
        avg_input_length = float(np.mean([row['total_input_length'] for row in rows])) if rows else 0.0
        pnt(
            f'{split_name} samples ready count={len(rows)}/{total_candidate_targets} '
            f'invalid={invalid_target_count} dropped_short_seq={dropped_short_sequence_count} '
            f'{avg_item_label}={avg_item_count:.2f} avg_input_length={avg_input_length:.2f} '
            f'resolved_maxitems={resolved_maxitems} saved={output_path}'
        )

    def build_alignment_meta(self):
        views = set(self.config.repr_types + [self.config.task_type])
        available_views = sorted(view for view in views if view in self.item_views)
        pairs = [list(pair) for pair in combinations(available_views, 2)]
        self._save_json(
            self.alignment_dir / 'meta.json',
            {
                'enabled': True,
                'views': available_views,
                'pairs': pairs,
            },
        )
        pnt(
            f'alignment meta saved enabled=True '
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
                'model_kind': self.backbone.kind,
                'model_max_length': int(self.backbone.max_length),
                'processed_dir': str(self.store.processed_dir()),
            },
        )
        pnt(f'meta saved to {self.meta_path}')

    def save_stats(self):
        stats = {
            'item_count': len(self.uid_raw_items),
            'finetune_sample_count': self.samples_stats['finetune']['sample_count'],
            'valid_sample_count': self.samples_stats['valid']['sample_count'],
            'test_sample_count': self.samples_stats['test']['sample_count'],
            'finetune_invalid_target_count': self.samples_stats['finetune']['invalid_target_count'],
            'valid_invalid_target_count': self.samples_stats['valid']['invalid_target_count'],
            'test_invalid_target_count': self.samples_stats['test']['invalid_target_count'],
            'resolved_maxitems': max(
                self.samples_stats['finetune']['resolved_maxitems'],
                self.samples_stats['valid']['resolved_maxitems'],
                self.samples_stats['test']['resolved_maxitems'],
            ),
            'sid_base_num_quantizers': self.sid_stats.get('base_num_quantizers', 0),
            'sid_final_num_quantizers': self.sid_stats.get('final_num_quantizers', 0),
            'sid_collision_vocab_size': self.sid_stats.get('collision_vocab_size', 0),
            'sid_collision_group_count': self.sid_stats.get('collision_group_count', 0),
            'sid_collided_item_count': self.sid_stats.get('collided_item_count', 0),
            'sid_max_collision_size': self.sid_stats.get('max_collision_size', 0),
        }
        stats_path = self.output_dir / 'stats.json'
        self._save_json(stats_path, stats)
        pnt(
            f"stats saved to {stats_path}: item_count={stats['item_count']} "
            f"finetune_samples={stats['finetune_sample_count']} test_samples={stats['test_sample_count']} "
            f"resolved_maxitems={stats['resolved_maxitems']} "
            f"sid_final_num_quantizers={stats['sid_final_num_quantizers']} "
            f"sid_max_collision_size={stats['sid_max_collision_size']}"
        )

    def log_sample_visuals(self):
        for split_name in ['finetune', 'test']:
            visual = self.sample_visuals.get(split_name)
            if visual:
                pnt(f'{split_name} sample visualization:\n{visual}')


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description='Compile processed data into trainer-ready dataset assets.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--model', required=True, help='Backbone model name, such as llama3 or transformer.')
    parser.add_argument('--repr.type', dest='repr_type', required=True, help='Representation types, such as uid, text, or uid+text.')
    parser.add_argument('--repr.model', dest='repr_model', default=None, help='External representation model, such as bertbase.')
    parser.add_argument('--repr.best', dest='repr_best', default=None, help='Best checkpoint metric for quantized codes, such as coll.')
    parser.add_argument('--repr.combine', dest='repr_combine', default='concat', help='How to combine multiple repr types: concat or add.')
    parser.add_argument('--model.maxlen', dest='model_max_length', type=int, default=0, help='Optional override for backbone max length, e.g. 2048.')
    parser.add_argument('--task.type', dest='task_type', required=True, choices=['uid', 'sid', 'embedding'])
    parser.add_argument('--maxitems', type=int, default=0, help='Maximum history items, 0 means auto by model max length.')
    parser.add_argument('--item-text-max-tokens', type=int, default=50, help='Maximum tokenized length per item text.')
    args = parser.parse_args()

    config = CompileConfig(
        data=args.data.lower(),
        model=args.model.lower(),
        repr_type=args.repr_type.lower(),
        repr_model=normalize_model_name(args.repr_model),
        repr_best=args.repr_best.lower() if args.repr_best else None,
        repr_combine=args.repr_combine.lower(),
        task_type=args.task_type.lower(),
        maxitems=int(args.maxitems),
        model_max_length=int(args.model_max_length) or None,
        item_text_max_tokens=int(args.item_text_max_tokens),
    )
    compiler = Compiler(config)
    compiler.run()
