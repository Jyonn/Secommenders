import argparse
import json
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
from utils.artifact_identity import (
    compiled_artifact_identity,
    register_compiled_artifact,
    resolve_compiled_dir,
    resolve_quantized_dir_from_upstream,
)
from utils.compile import CompileConfig, normalize_model_name
from utils.experiment_template import build_default_upstreams
from utils import function
from utils.logging import setup_logging
from utils.pipeline import ensure_embedded, ensure_quantized
from utils.embedding_fusion import load_fused_embeddings, fusion_model_ref, normalize_embedding_fusion
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
    VER = 'v3.2'
    SUPPORTED_REPR_TYPES = {'uid', 'sid', 'hash', 'text', 'embedding'}
    SUPPORTED_TASK_TYPES = {'uid', 'sid', 'hash', 'embedding'}
    SUPPORTED_REPR_COMBINES = {'concat', 'add'}
    SUPPORTED_QUANTIZERS = ('rqvae', 'pqvae', 'opqvae', 'basic-rqvae', 'lsh', 'simhash', 'pcahash', 'itq')

    def __init__(self, config: CompileConfig):
        self.config = config
        self.store = ArtifactStore(config.data)
        self.output_dir = resolve_compiled_dir(config)
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
        self.hash_stats = {}

        self._init_dirs()

    def _init_dirs(self):
        self.vocab_dir = self.output_dir / 'vocab'
        self.prompts_dir = self.output_dir / 'prompts'
        self.item_views_dir = self.output_dir / 'item_views'
        self.samples_dir = self.output_dir / 'samples'
        for path in [self.vocab_dir, self.prompts_dir, self.item_views_dir, self.samples_dir]:
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
        return '<' + '+'.join(self.config.representation_names) + '>'

    def _task_repr_label(self):
        return '<' + '+'.join(self.config.target_names) + '>'

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
        kind = self.config.representation_kind(view_name)
        if kind == 'uid':
            return f'uid={uid}'
        if kind == 'text':
            token_ids = self.item_views[view_name][uid]
            preview = self._truncate_text(self.item_texts[uid], max_chars=36)
            return f'text[{len(token_ids)}]="{preview}"'
        if kind == 'sid':
            codes = self._as_token_list(self.item_views[view_name][uid])
            preview = ','.join(str(code) for code in codes[:6])
            if len(codes) > 6:
                preview += ',...'
            return f'sid[{len(codes)}]=[{preview}]'
        if kind == 'hash':
            codes = self._as_token_list(self.item_views[view_name][uid])
            preview = ','.join(str(code) for code in codes[:6])
            if len(codes) > 6:
                preview += ',...'
            return f'hash[{len(codes)}]=[{preview}]'
        if kind == 'embedding':
            return f'{view_name}#{self.item_views[view_name][uid]}'
        return str(self.item_views[view_name][uid])

    def _history_repr_summary(self, uid: int):
        if self.config.repr_combine == 'add':
            uid_part = self._summarize_view_value('uid', uid)
            emb_part = self._summarize_view_value('embedding', uid)
            return f'add({uid_part} + linear({emb_part})) -> 1 slot'

        parts = [self._summarize_view_value(name, uid) for name in self.config.representation_names]
        return ' + '.join(parts)

    def _task_repr_summary(self, uid: int):
        return ' + '.join(self._summarize_view_value(name, uid) for name in self.config.target_names)

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
        processed_build_id = self._processed_build_id()
        if not processed_build_id or meta.get('processed_build_id') != processed_build_id:
            pnt(
                f'compiled cache processed identity mismatch at {self.output_dir}; '
                'rebuilding samples from current processed artifacts'
            )
            return False
        required_item_view_paths = [self.item_views_dir / 'uid.parquet']
        if self.requires_view('text'):
            required_item_view_paths.append(self.item_views_dir / 'text.parquet')
        if self.requires_view('sid'):
            required_item_view_paths.append(self.item_views_dir / 'sid.parquet')
        if self.requires_view('hash'):
            required_item_view_paths.append(self.item_views_dir / 'hash.parquet')
        if self.config.representation_graph:
            required_item_view_paths = [
                self.item_views_dir / f'{name}.parquet'
                for name in self.config.representation_names
            ]
            required_item_view_paths.extend(
                self.vocab_dir / f'{name}.json'
                for name in self.config.names_for_kind('sid')
            )
            for name in self.config.names_for_kind('embedding'):
                required_item_view_paths.append(self.output_dir / 'embeddings' / f'{name}.npy')
        elif self.requires_view('embedding'):
            required_item_view_paths.append(self.item_views_dir / 'embedding.parquet')
            if self.config.embedding:
                required_item_view_paths.append(self.output_dir / 'embeddings.npy')
        required_paths = [
            self.output_dir / 'stats.json',
            self.vocab_dir / 'special.json',
            self.vocab_dir / 'meta.json',
            self.prompts_dir / 'main.json',
            self.item_views_dir / 'meta.json',
            self.samples_dir / 'finetune.parquet',
            self.samples_dir / 'valid.parquet',
            self.samples_dir / 'test.parquet',
        ] + required_item_view_paths
        return all(path.exists() for path in required_paths)

    def _processed_build_id(self):
        path = self.store.processed_dir() / 'meta.json'
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text()).get('build_id')
        except (json.JSONDecodeError, OSError):
            return None

    def validate(self):
        repr_types = [self.config.representation_kind(name) for name in self.config.representation_names]
        model_name = str(self.config.model).strip().lower()
        is_scratch_model = model_name == 'scratch'
        if not is_scratch_model and model_utils.match(model_name) is None:
            raise ValueError(
                f'Unknown model "{self.config.model}". '
                f'Use "scratch" for the scratch backbone or add the model alias to .model'
            )
        if not repr_types:
            raise ValueError('repr.type must contain at least one representation')
        if not self.config.representation_graph and len(set(repr_types)) != len(repr_types):
            raise ValueError(f'repr.type contains duplicates: {self.config.repr_type}')
        unsupported_repr_types = [repr_type for repr_type in repr_types if repr_type not in self.SUPPORTED_REPR_TYPES]
        if unsupported_repr_types:
            raise ValueError(f'Unsupported repr.type entries: {unsupported_repr_types}')
        task_types = [self.config.representation_kind(name) for name in self.config.target_names]
        unsupported_task_types = [task for task in task_types if task not in self.SUPPORTED_TASK_TYPES]
        if unsupported_task_types:
            raise ValueError(f'Unsupported task.type entries: {unsupported_task_types}')
        if len(task_types) > 1:
            if (
                any(kind not in {'sid', 'uid'} for kind in task_types)
                or task_types.count('uid') > 1
                or ('uid' in task_types and task_types[-1] != 'uid')
            ):
                raise ValueError(
                    'multi task decoding supports one or more sid targets, '
                    'optionally followed by one uid target'
                )
        if self.config.repr_combine not in self.SUPPORTED_REPR_COMBINES:
            raise ValueError(f'Unsupported repr.combine: {self.config.repr_combine}')
        if not set(self.config.target_names).issubset(self.config.representation_names):
            raise ValueError('repr.type must contain every task.type representation')
        if self.config.representation_names[:len(self.config.target_names)] != self.config.target_names:
            raise ValueError('task.type entries must lead repr.type for causal mixed-view training')
        if self.config.repr_combine == 'add':
            if not (self.config.task_type == 'uid' and repr_types == ['uid', 'embedding']):
                raise ValueError(
                    'repr.combine=add is currently only supported for uid+embedding history with uid targets'
                )
        if is_scratch_model and 'text' in repr_types:
            raise ValueError('scratch backbone currently does not support repr.type containing text')

        used_views = set(repr_types + task_types)
        hash_upstream = self.config.upstreams.get('hash') or {}
        hash_has_source = bool((hash_upstream.get('embedding') or {}).get('sources') or hash_upstream.get('embedding_model'))
        named_embeddings = self.config.names_for_kind('embedding') if self.config.representation_graph else []
        if 'embedding' in used_views and not (named_embeddings or self.config.embedding or self.config.repr_source_model):
            raise ValueError(
                'embedding representation requires repr_source_model or representation.embedding.models'
            )
        sid_names = self.config.names_for_kind('sid')
        for sid_name in sid_names:
            sid_upstream = self.config.upstream_for(sid_name)
            sid_has_source = bool(
                (sid_upstream.get('embedding') or {}).get('sources')
                or sid_upstream.get('embedding_model')
            )
            if not (sid_has_source or self.config.repr_source_model):
                raise ValueError(
                    f'SID representation {sid_name} requires embedding sources'
                )
        if 'hash' in used_views and not (hash_has_source or self.config.repr_source_model):
            raise ValueError('hash requires repr_source_model or upstreams.hash.embedding.sources')
        if not self.config.representation_graph and 'sid' in set(repr_types + self.config.task_types):
            if not self.config.sid_export:
                raise ValueError('data.sid_export is required when repr.type or task.type uses sid')
            if not self.config.sid_coder:
                raise ValueError('data.sid_coder is required when repr.type or task.type uses sid')
        if 'hash' in set(repr_types + self.config.task_types) and not self.config.hash_coder:
            raise ValueError('data.hash_coder is required when repr.type or task.type uses hash')

    def run(self):
        self.validate()
        pnt(
            f'start compile data={self.config.data} model={self.config.model} '
            f'repr={self.config.repr_type} combine={self.config.repr_combine} '
            f'task={self.config.task_type} maxitems={self.config.maxitems} '
            f'textlen={self.config.item_text_max_tokens} alignment=always '
            f'model.maxlen={self.config.model_max_length or "native"}'
        )
        # Validate/rebuild processed artifacts before trusting compiled samples.
        self.load_processor()
        if self.is_cached():
            pnt(f'compiled dataset cached at {self.output_dir}')
            return

        pnt(f'cache miss, compiling into {self.output_dir}')
        self.init_backbone()
        self.build_vocab_and_prompts()
        self.build_item_views()
        self.build_samples('finetune', self.processor.finetune_set)
        self.build_samples('valid', self.processor.valid_set)
        self.build_samples('test', self.processor.test_set)
        self.save_meta()
        self.save_stats()
        self.log_sample_visuals()
        pnt(f'compile finished: outputs written to {self.output_dir}')

    def load_processor(self):
        pnt(f'loading processed dataset {self.config.data}')
        self.processor = function.load_processor(self.config.data)
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
        marker_names = (
            list(dict.fromkeys(self.config.representation_names + self.config.target_names))
            if self.config.representation_graph
            else TYPE_MARKER_ORDER
        )
        if self.config.representation_graph and len(self.config.target_names) > 1:
            marker_names.append('decoder_' + '_'.join(self.config.target_names))
        marker_tokens = {
            name: f'<repr:{name}>' if self.config.representation_graph else TYPE_MARKER_TOKENS[name]
            for name in marker_names
        }
        self._save_json(
            special_vocab_path,
            {
                'tokens': [marker_tokens[name] for name in marker_names],
                'marker_to_index': {name: index for index, name in enumerate(marker_names)},
                'external_ids': {name: -1 for name in marker_names},
            },
        )
        self.registry.register(
            'special',
            kind='special',
            size=len(marker_names),
            path=special_vocab_path,
        )

        for sid_name in self.config.names_for_kind('sid'):
            sid_meta = self.load_sid_view(name=sid_name, build_only_meta=True)
            sid_vocab_path = self.vocab_dir / f'{sid_name}.json'
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
                    'quantizer_name': sid_meta.get('quantizer_name'),
                    'quantizer_scheme': sid_meta.get('quantizer_scheme'),
                    'recommended_decoding': sid_meta.get('recommended_decoding'),
                    'quantized_export_dir': sid_meta.get('quantized_export_dir'),
                },
            )
            self.registry.register(
                sid_name,
                kind='sid',
                size=len(tokens),
                path=sid_vocab_path,
                num_quantizers=sid_meta['final_num_quantizers'],
                base_num_quantizers=sid_meta['num_quantizers'],
                codebook_size=sid_meta['codebook_size'],
                collision_vocab_size=sid_meta['collision_vocab_size'],
                quantizer_name=sid_meta.get('quantizer_name'),
                quantizer_scheme=sid_meta.get('quantizer_scheme'),
                recommended_decoding=sid_meta.get('recommended_decoding'),
            )
            pnt(
                f"registered sid vocab name={sid_name} size={len(tokens)} "
                f"base_num_quantizers={sid_meta['num_quantizers']} "
                f"final_num_quantizers={sid_meta['final_num_quantizers']} "
                f"codebook_size={sid_meta['codebook_size']} "
                f"collision_vocab_size={sid_meta['collision_vocab_size']} "
                f"max_collision_size={sid_meta['max_collision_size']}"
            )
        if self.requires_view('hash'):
            hash_meta = self.load_hash_view(build_only_meta=True)
            hash_vocab_path = self.vocab_dir / 'hash.json'
            tokens = [
                f'h{token_index}_c{code}'
                for token_index, slot_size in enumerate(hash_meta['slot_sizes'])
                for code in range(slot_size)
            ] + [
                f'collision_{index}'
                for index in range(hash_meta['collision_vocab_size'])
            ]
            self._save_json(
                hash_vocab_path,
                {
                    'tokens': tokens,
                    'num_tokens': hash_meta['final_num_tokens'],
                    'base_num_tokens': hash_meta['num_tokens'],
                    'slot_sizes': hash_meta['slot_sizes'],
                    'slot_offsets': hash_meta['slot_offsets'],
                    'codebook_size': max(hash_meta['slot_sizes']) if hash_meta['slot_sizes'] else 0,
                    'collision_vocab_size': hash_meta['collision_vocab_size'],
                    'collision_token_offset': hash_meta['collision_token_offset'],
                    'collision_group_count': hash_meta['collision_group_count'],
                    'collided_item_count': hash_meta['collided_item_count'],
                    'max_collision_size': hash_meta['max_collision_size'],
                    'num_bits_total': hash_meta['num_bits_total'],
                    'bits_per_token': hash_meta['bits_per_token'],
                    'quantizer_name': hash_meta.get('quantizer_name'),
                    'quantizer_scheme': hash_meta.get('quantizer_scheme'),
                    'recommended_decoding': hash_meta.get('recommended_decoding'),
                    'quantized_export_dir': hash_meta.get('quantized_export_dir'),
                },
            )
            hash_name = self.config.primary_name('hash') or 'hash'
            self.registry.register(
                hash_name,
                kind='hash',
                size=len(tokens),
                path=hash_vocab_path,
                num_tokens=hash_meta['final_num_tokens'],
                base_num_tokens=hash_meta['num_tokens'],
                slot_sizes=hash_meta['slot_sizes'],
                slot_offsets=hash_meta['slot_offsets'],
                codebook_size=max(hash_meta['slot_sizes']) if hash_meta['slot_sizes'] else 0,
                collision_vocab_size=hash_meta['collision_vocab_size'],
                quantizer_name=hash_meta.get('quantizer_name'),
                quantizer_scheme=hash_meta.get('quantizer_scheme'),
                recommended_decoding=hash_meta.get('recommended_decoding'),
            )
            pnt(
                f"registered hash vocab size={len(tokens)} "
                f"base_num_tokens={hash_meta['num_tokens']} "
                f"final_num_tokens={hash_meta['final_num_tokens']} "
                f"slot_sizes={hash_meta['slot_sizes']} "
                f"collision_vocab_size={hash_meta['collision_vocab_size']} "
                f"max_collision_size={hash_meta['max_collision_size']}"
            )

        self._save_json(self.vocab_dir / 'meta.json', self.registry.to_dict())
        main_prompt = self.backbone.build_prompt_spec()
        self._save_json(self.prompts_dir / 'main.json', main_prompt)
        pnt(
            f'vocab ready namespaces={len(self.registry.entries)} '
            f'main_prompt=(history_prefix={len(main_prompt["history_prefix_ids"])}, '
            f'separator={len(main_prompt["item_separator_ids"])}, '
            f'query_prefix={len(main_prompt["query_prefix_ids"])})'
        )

    def requires_view(self, view_name: str):
        if self.config.representation_graph:
            return view_name in self.config.representation_names or any(
                self.config.representation_kind(name) == view_name
                for name in self.config.representation_names
            )
        views = {'uid', *self.config.repr_types, *self.config.task_types}
        return view_name in views

    def _upstream(self, name: str):
        return self.config.upstream_for(name)

    def build_item_views(self):
        if self.config.representation_graph:
            return self._build_named_item_views()
        required_views = [view for view in ['uid', 'text', 'sid', 'hash', 'embedding'] if self.requires_view(view)]
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
                f'loading sid view from model={self.config.repr_source_model} '
                f'export={self.config.sid_export}'
            )
            sid_values = self.load_sid_view()
            self._write_view('sid', sid_values)
            sid_lengths = [len(value) for value in sid_values]
            pnt(
                f'sid view ready avg_codes={np.mean(sid_lengths):.2f} '
                f'max_codes={max(sid_lengths) if sid_lengths else 0}'
            )

        if self.requires_view('hash'):
            pnt(
                f'loading hash view from model={self.config.repr_source_model} '
                f'coder={self.config.hash_coder}'
            )
            hash_values = self.load_hash_view()
            self._write_view('hash', hash_values)
            hash_lengths = [len(value) for value in hash_values]
            pnt(
                f'hash view ready avg_codes={np.mean(hash_lengths):.2f} '
                f'max_codes={max(hash_lengths) if hash_lengths else 0}'
            )

        if self.requires_view('embedding'):
            source = fusion_model_ref(self.config.embedding) if self.config.embedding else self.config.repr_source_model
            pnt(f'loading embedding view from model={source}')
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

    def _build_named_item_views(self):
        names = self.config.representation_names
        pnt(f'building named item views {names} for {len(self.uid_raw_items)} items')
        text_values = None
        for name in names:
            spec = self.config.representation_spec(name)
            kind = spec['type']
            if kind == 'uid':
                values = list(range(len(self.uid_raw_items)))
            elif kind == 'text':
                if text_values is None:
                    text_values = []
                    for start in tqdm(range(0, len(self.item_texts), 256), desc=f'{name} view', leave=False):
                        text_values.extend(self.backbone.tokenize_texts(
                            self.item_texts[start:start + 256],
                            max_tokens=self.config.item_text_max_tokens,
                        ))
                values = text_values
            elif kind == 'sid':
                values = self.load_sid_view(name=name)
            elif kind == 'hash':
                values = self.load_hash_view()
            elif kind == 'embedding':
                values = self.load_embedding_view(name=name, spec=spec['embedding'])
            else:
                raise ValueError(f'unsupported named representation type: {kind}')
            self._write_view(name, values)
            pnt(f'named view ready name={name} type={kind} rows={len(values)}')
        self._save_json(self.item_views_dir / 'meta.json', {
            'row_order': 'uid_vocab',
            'views': names,
            'types': {name: self.config.representation_kind(name) for name in names},
        })

    def _load_quantized_export(self, name='sid'):
        upstream = self._upstream(name)
        quantizer = upstream.get('quantizer') or {}
        legacy_model = normalize_model_name(upstream.get('embedding_model') or self.config.repr_source_model)
        embedding_spec = normalize_embedding_fusion(upstream.get('embedding') or {}, legacy_model=legacy_model)
        model_name = fusion_model_ref(embedding_spec)
        quantizer_name = (quantizer.get('name') or self.config.sid_coder or '').strip().lower()
        export_name = str(upstream.get('export') or self.config.sid_export or '').strip().lower()
        if not quantizer_name:
            raise ValueError('quantizer_name is required when compile config uses sid views')
        if not export_name:
            raise ValueError('sid export is required when compile config uses sid views')
        if quantizer_name not in self.SUPPORTED_QUANTIZERS:
            supported = ', '.join(self.SUPPORTED_QUANTIZERS)
            raise ValueError(
                f'Unsupported quantizer "{quantizer_name}". '
                f'Only {supported} are supported.'
            )

        export_dir = (
            resolve_quantized_dir_from_upstream(self.config.data, upstream)
            if upstream
            else self.store.quantized_dir(model_name, quantizer_name)
        ) / 'exports' / export_name

        def _export_ready(path: Path):
            meta_path = path / 'meta.json'
            codes_path = path / 'codebook_indices.npy'
            item_ids_path = path / 'item_ids.parquet'
            return meta_path.exists() and codes_path.exists() and item_ids_path.exists()

        if quantizer_name != 'basic-rqvae':
            ensure_quantized(self.config.data, model_name, quantizer_name, upstream)
        export_ready = _export_ready(export_dir)
        if not export_ready:
            if quantizer_name == 'basic-rqvae':
                raise FileNotFoundError(
                    'basic-rqvae export not found. '
                    f'Expected: {export_dir}. '
                    f'Please run `python basic_rqvae_quantizer.py --data {self.config.data} --model {model_name}` first.'
                )
        if not export_ready:
            raise FileNotFoundError(
                'Quantized export not found after auto preparation. '
                f'Searched: {export_dir}'
            )

        meta_path = export_dir / 'meta.json'
        codes_path = export_dir / 'codebook_indices.npy'
        item_ids_path = export_dir / 'item_ids.parquet'
        pnt(f'loading quantized export from {export_dir}')
        meta = json.loads(meta_path.read_text())
        codes = np.load(codes_path)
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
        pnt(
            f'loaded quantized export rows={len(item_ids)} shape={list(codes.shape)} '
            f'quantizer={meta.get("quantizer_model", "unknown")}'
        )
        return export_dir, meta, item_ids, codes

    def _load_hash_export(self):
        upstream = self._upstream('hash')
        quantizer = upstream.get('quantizer') or {}
        legacy_model = normalize_model_name(upstream.get('embedding_model') or self.config.repr_source_model)
        embedding_spec = normalize_embedding_fusion(upstream.get('embedding') or {}, legacy_model=legacy_model)
        model_name = fusion_model_ref(embedding_spec)
        quantizer_name = (quantizer.get('name') or self.config.hash_coder or '').strip().lower()
        export_name = str(upstream.get('export') or 'hash').strip().lower()
        if not quantizer_name:
            raise ValueError('quantizer_name is required when compile config uses hash views')
        if quantizer_name not in self.SUPPORTED_QUANTIZERS:
            supported = ', '.join(self.SUPPORTED_QUANTIZERS)
            raise ValueError(
                f'Unsupported quantizer "{quantizer_name}". '
                f'Only {supported} are supported.'
            )

        export_dir = (
            resolve_quantized_dir_from_upstream(self.config.data, upstream)
            if upstream
            else self.store.quantized_dir(model_name, quantizer_name)
        ) / 'exports' / export_name

        def _export_ready(path: Path):
            meta_path = path / 'meta.json'
            bits_path = path / 'binary_bits.npy'
            item_ids_path = path / 'item_ids.parquet'
            return meta_path.exists() and bits_path.exists() and item_ids_path.exists()

        ensure_quantized(self.config.data, model_name, quantizer_name, upstream)
        export_ready = _export_ready(export_dir)
        if not export_ready:
            raise FileNotFoundError(
                'Hash export not found after auto preparation. '
                f'Searched: {export_dir}'
            )

        meta_path = export_dir / 'meta.json'
        bits_path = export_dir / 'binary_bits.npy'
        item_ids_path = export_dir / 'item_ids.parquet'
        pnt(f'loading hash export from {export_dir}')
        meta = json.loads(meta_path.read_text())
        bits = np.load(bits_path)
        item_ids = pd.read_parquet(item_ids_path)[self.processor.IID_COL].tolist()
        pnt(
            f'loaded hash export rows={len(item_ids)} shape={list(bits.shape)} '
            f'hash_model={meta.get("hash_model", meta.get("quantizer_model", "unknown"))}'
        )
        return export_dir, meta, item_ids, bits

    def load_sid_view(self, name='sid', build_only_meta=False):
        export_dir, meta, item_ids, codes = self._load_quantized_export(name)
        num_quantizers = int(codes.shape[1]) if codes.ndim > 1 else 1
        codebook_size = int(meta['quantizer_config']['codebook_size'])
        collision_token_offset = num_quantizers * codebook_size

        base_sid_groups = {}
        for item_id, row in zip(item_ids, codes):
            row = np.atleast_1d(row).tolist()
            base_sid = tuple(index * codebook_size + int(code) for index, code in enumerate(row))
            base_sid_groups.setdefault(base_sid, []).append(str(item_id))

        max_collision_size = max((len(group) for group in base_sid_groups.values()), default=1)
        collision_vocab_size = max_collision_size
        final_num_quantizers = num_quantizers + 1
        collision_group_count = sum(1 for group in base_sid_groups.values() if len(group) > 1)
        collided_item_count = sum(len(group) for group in base_sid_groups.values() if len(group) > 1)

        self.sid_stats[name] = {
            'base_num_quantizers': num_quantizers,
            'final_num_quantizers': final_num_quantizers,
            'codebook_size': codebook_size,
            'collision_vocab_size': collision_vocab_size,
            'collision_token_offset': collision_token_offset,
            'collision_group_count': collision_group_count,
            'collided_item_count': collided_item_count,
            'max_collision_size': max_collision_size,
            'quantizer_name': meta.get('quantizer_model'),
            'quantizer_scheme': meta.get('quantizer_scheme'),
            'recommended_decoding': meta.get('recommended_decoding'),
            'quantized_export_dir': str(export_dir),
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
                'quantizer_name': meta.get('quantizer_model'),
                'quantizer_scheme': meta.get('quantizer_scheme'),
                'recommended_decoding': meta.get('recommended_decoding'),
                'quantized_export_dir': str(export_dir),
            }

        sid_map = {}
        for base_sid, grouped_item_ids in tqdm(base_sid_groups.items(), total=len(base_sid_groups), desc='sid map', leave=False):
            ordered_item_ids = sorted(grouped_item_ids, key=lambda value: str(value))
            for collision_index, item_id in enumerate(ordered_item_ids):
                sid_map[item_id] = list(base_sid) + [collision_token_offset + collision_index]

        missing = []
        ordered_values = []
        for item_id in tqdm(self.uid_raw_items, desc='sid align', leave=False):
            item_key = str(item_id)
            if item_key not in sid_map:
                missing.append(item_id)
                continue
            ordered_values.append(sid_map[item_key])
        if missing:
            raise ValueError(f'{len(missing)} items missing sid codes, first missing item: {missing[0]}')

        return ordered_values

    @staticmethod
    def _pack_bit_groups(bit_values: list[int], bits_per_token: int, num_tokens: int):
        padded = list(bit_values)
        total_required = bits_per_token * num_tokens
        if len(padded) < total_required:
            padded.extend([0] * (total_required - len(padded)))

        packed = []
        for token_index in range(num_tokens):
            start = token_index * bits_per_token
            end = start + bits_per_token
            bucket_id = 0
            for bit in padded[start:end]:
                bucket_id = (bucket_id << 1) | int(bit)
            packed.append(bucket_id)
        return packed

    def load_hash_view(self, build_only_meta=False):
        export_dir, meta, item_ids, binary_bits = self._load_hash_export()
        if binary_bits.ndim != 2:
            raise ValueError(f'Expected 2D binary hash array, got shape {binary_bits.shape}')

        num_bits_total = int(binary_bits.shape[1])
        num_tokens = 3
        bits_per_token = int(np.ceil(num_bits_total / max(num_tokens, 1)))
        raw_slot_values = [[] for _ in range(num_tokens)]
        packed_rows = []
        for row in tqdm(binary_bits, total=len(binary_bits), desc='hash pack', leave=False):
            packed = self._pack_bit_groups(np.asarray(row).astype(np.uint8).tolist(), bits_per_token, num_tokens)
            packed_rows.append(packed)
            for token_index, packed_value in enumerate(packed):
                raw_slot_values[token_index].append(int(packed_value))

        slot_value_maps = []
        slot_sizes = []
        slot_offsets = []
        offset = 0
        for token_index in range(num_tokens):
            unique_values = sorted(set(raw_slot_values[token_index]))
            value_map = {value: index for index, value in enumerate(unique_values)}
            slot_value_maps.append(value_map)
            slot_sizes.append(len(unique_values))
            slot_offsets.append(offset)
            offset += len(unique_values)

        collision_token_offset = offset

        base_hash_groups = {}
        for item_id, packed in tqdm(zip(item_ids, packed_rows), total=len(item_ids), desc='hash group', leave=False):
            base_hash = tuple(
                slot_offsets[token_index] + slot_value_maps[token_index][int(code)]
                for token_index, code in enumerate(packed)
            )
            base_hash_groups.setdefault(base_hash, []).append(str(item_id))

        max_collision_size = max((len(group) for group in base_hash_groups.values()), default=1)
        collision_vocab_size = max_collision_size
        final_num_tokens = num_tokens + 1
        collision_group_count = sum(1 for group in base_hash_groups.values() if len(group) > 1)
        collided_item_count = sum(len(group) for group in base_hash_groups.values() if len(group) > 1)

        self.hash_stats = {
            'base_num_tokens': num_tokens,
            'final_num_tokens': final_num_tokens,
            'slot_sizes': slot_sizes,
            'slot_offsets': slot_offsets,
            'collision_vocab_size': collision_vocab_size,
            'collision_token_offset': collision_token_offset,
            'collision_group_count': collision_group_count,
            'collided_item_count': collided_item_count,
            'max_collision_size': max_collision_size,
            'num_bits_total': num_bits_total,
            'bits_per_token': bits_per_token,
            'quantizer_name': meta.get('hash_model', meta.get('quantizer_model')),
            'quantizer_scheme': meta.get('quantizer_scheme'),
            'recommended_decoding': meta.get('recommended_decoding', 'parallel'),
            'quantized_export_dir': str(export_dir),
        }

        if build_only_meta:
            return {
                'num_tokens': num_tokens,
                'final_num_tokens': final_num_tokens,
                'slot_sizes': slot_sizes,
                'slot_offsets': slot_offsets,
                'collision_vocab_size': collision_vocab_size,
                'collision_token_offset': collision_token_offset,
                'collision_group_count': collision_group_count,
                'collided_item_count': collided_item_count,
                'max_collision_size': max_collision_size,
                'num_bits_total': num_bits_total,
                'bits_per_token': bits_per_token,
                'quantizer_name': meta.get('hash_model', meta.get('quantizer_model')),
                'quantizer_scheme': meta.get('quantizer_scheme'),
                'recommended_decoding': meta.get('recommended_decoding', 'parallel'),
                'quantized_export_dir': str(export_dir),
            }

        hash_map = {}
        for base_hash, grouped_item_ids in tqdm(base_hash_groups.items(), total=len(base_hash_groups), desc='hash map', leave=False):
            ordered_item_ids = sorted(grouped_item_ids, key=lambda value: str(value))
            for collision_index, item_id in enumerate(ordered_item_ids):
                hash_map[item_id] = list(base_hash) + [collision_token_offset + collision_index]

        missing = []
        ordered_values = []
        for item_id in tqdm(self.uid_raw_items, desc='hash align', leave=False):
            item_key = str(item_id)
            if item_key not in hash_map:
                missing.append(item_id)
                continue
            ordered_values.append(hash_map[item_key])
        if missing:
            raise ValueError(f'{len(missing)} items missing hash codes, first missing item: {missing[0]}')

        return ordered_values

    def load_embedding_view(self, name=None, spec=None):
        if spec:
            matrix, item_ids, summaries = load_fused_embeddings(
                self.config.data,
                self.processor,
                spec,
                ensure_embedded,
            )
            expected_ids = [str(item) for item in self.uid_raw_items]
            if item_ids != expected_ids:
                raise ValueError('fused embedding item order does not match compiler uid vocabulary')
            embedding_dir = self.output_dir / 'embeddings'
            embedding_dir.mkdir(parents=True, exist_ok=True)
            np.save(embedding_dir / f'{name}.npy', matrix)
            self._save_json(embedding_dir / f'{name}.json', {
                'representation': name,
                'model': fusion_model_ref(spec),
                'shape': list(matrix.shape),
                'sources': summaries,
                'normalize_output': spec['normalize_output'],
            })
            return list(range(len(item_ids)))
        if self.config.embedding:
            matrix, item_ids, summaries = load_fused_embeddings(
                self.config.data,
                self.processor,
                self.config.embedding,
                ensure_embedded,
            )
            expected_ids = [str(item) for item in self.uid_raw_items]
            if item_ids != expected_ids:
                raise ValueError('fused embedding item order does not match compiler uid vocabulary')
            np.save(self.output_dir / 'embeddings.npy', matrix)
            self._save_json(
                self.output_dir / 'embedding_meta.json',
                {
                    'model': fusion_model_ref(self.config.embedding),
                    'shape': list(matrix.shape),
                    'sources': summaries,
                    'normalize_output': self.config.embedding['normalize_output'],
                },
            )
            return list(range(len(item_ids)))

        model_name = normalize_model_name(self.config.repr_source_model)
        embedding_dir = self.store.embedded_dir(model_name)
        item_ids_path = embedding_dir / 'item_ids.parquet'
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
        for repr_type in self.config.representation_names:
            total += 1
            value = self._get_repr_view_value(repr_type, uid)
            total += len(value) if isinstance(value, list) else 1
        return total

    def _compose_history_item(self, uid: int):
        if self.config.repr_combine == 'add':
            return self.item_views['uid'][uid]

        tokens = []
        for repr_type in self.config.representation_names:
            tokens.extend(self._as_token_list(self._get_repr_view_value(repr_type, uid)))
        return tokens

    def _history_values(self, history_uids: list[int]):
        return [self._compose_history_item(uid) for uid in history_uids]

    def _target_value(self, target_uid: int):
        if len(self.config.target_names) > 1:
            return [
                token
                for task_type in self.config.target_names
                for token in self._as_token_list(self.item_views[task_type][target_uid])
            ]
        target_name = self.config.target_names[0]
        if self.config.representation_kind(target_name) == 'uid':
            return self.item_views[target_name][target_uid]
        return self.item_views[target_name][target_uid]

    def _repr_segment_length(self, repr_type: str, uid: int):
        value = self._get_repr_view_value(repr_type, uid)
        payload_length = len(value) if isinstance(value, list) else 1
        return 1 + payload_length

    def _finetune_item_length(self, uid: int, include_non_task: bool):
        total = sum(self._repr_segment_length(name, uid) for name in self.config.target_names)
        if include_non_task:
            for repr_type in self.config.representation_names[len(self.config.target_names):]:
                total += self._repr_segment_length(repr_type, uid)
        return total

    def _estimate_packed_length(self, sequence_uids: list[int]):
        if len(sequence_uids) < 2:
            return None
        if self.config.repr_combine == 'add':
            prompt = self.backbone.build_prompt_spec()
            separator_len = len(prompt['item_separator_ids']) * max(0, len(sequence_uids) - 1)
            fused_len = 2
            uid_target_len = 2
            total = separator_len + fused_len + uid_target_len
            if len(sequence_uids) > 2:
                total += (len(sequence_uids) - 2) * (uid_target_len + fused_len)
            return total
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
                'uid',
                'sequence_uids',
                'sequence_item_count',
                'prediction_count',
                'total_input_length',
            ]
        else:
            columns = [
                'uid',
                'history_uids',
                'target_uid',
                'ground_truth_uids',
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
                        'uid': row[self.processor.UID_COL],
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
                ground_truth_uids = None
                if (
                    split_name == 'test'
                    and self.processor.multi_item_col
                    and self.processor.multi_item_col in row
                ):
                    ground_truth_uids = [
                        self.uid_item_map[item_id]
                        for item_id in function.to_list(row[self.processor.multi_item_col])
                        if item_id in self.uid_item_map
                    ]
                    ground_truth_uids = list(dict.fromkeys(ground_truth_uids))
                    if not ground_truth_uids:
                        invalid_target_count += 1
                        iterator.set_postfix(samples=len(rows), invalid=invalid_target_count, maxhist=resolved_maxitems)
                        continue
                    prefix_uids = sequence
                    target_uid = int(ground_truth_uids[0])
                    target_pos = len(sequence)
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
                        'uid': row[self.processor.UID_COL],
                        'history_uids': history_uids,
                        'target_uid': int(target_uid),
                        'ground_truth_uids': ground_truth_uids if ground_truth_uids is not None else [int(target_uid)],
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
                        sequence=sequence + [int(target_uid)] if ground_truth_uids is not None else sequence,
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

    def save_meta(self):
        identity = compiled_artifact_identity(self.config, self.output_dir)
        primary_sid_name = self.config.primary_name('sid', targets=True) or self.config.primary_name('sid')
        primary_sid_stats = self.sid_stats.get(primary_sid_name, {})
        self._save_json(
            self.meta_path,
            {
                'version': self.VER,
                'stage': 'compiled',
                'data': self.config.data,
                'prepare_id': self.config.prepare_id,
                'config': self.config.config_dict,
                'artifact_identity': identity,
                'model_kind': self.backbone.kind,
                'model_max_length': int(self.backbone.max_length),
                'processed_dir': str(self.store.processed_dir()),
                'processed_build_id': self._processed_build_id(),
                'sid_quantizer_name': primary_sid_stats.get('quantizer_name'),
                'sid_quantizer_scheme': primary_sid_stats.get('quantizer_scheme'),
                'sid_recommended_decoding': primary_sid_stats.get('recommended_decoding'),
                'sid_representations': self.sid_stats,
                'hash_quantizer_name': self.hash_stats.get('quantizer_name'),
                'hash_quantizer_scheme': self.hash_stats.get('quantizer_scheme'),
                'hash_recommended_decoding': self.hash_stats.get('recommended_decoding'),
            },
        )
        register_compiled_artifact(self.config, self.output_dir, aliases=identity.get('aliases'))
        pnt(f'meta saved to {self.meta_path}')

    def save_stats(self):
        primary_sid_name = self.config.primary_name('sid', targets=True) or self.config.primary_name('sid')
        primary_sid_stats = self.sid_stats.get(primary_sid_name, {})
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
            'sid_base_num_quantizers': primary_sid_stats.get('base_num_quantizers', 0),
            'sid_final_num_quantizers': primary_sid_stats.get('final_num_quantizers', 0),
            'sid_collision_vocab_size': primary_sid_stats.get('collision_vocab_size', 0),
            'sid_collision_group_count': primary_sid_stats.get('collision_group_count', 0),
            'sid_collided_item_count': primary_sid_stats.get('collided_item_count', 0),
            'sid_max_collision_size': primary_sid_stats.get('max_collision_size', 0),
            'sid_representations': self.sid_stats,
            'hash_base_num_tokens': self.hash_stats.get('base_num_tokens', 0),
            'hash_final_num_tokens': self.hash_stats.get('final_num_tokens', 0),
            'hash_collision_vocab_size': self.hash_stats.get('collision_vocab_size', 0),
            'hash_collision_group_count': self.hash_stats.get('collision_group_count', 0),
            'hash_collided_item_count': self.hash_stats.get('collided_item_count', 0),
            'hash_max_collision_size': self.hash_stats.get('max_collision_size', 0),
        }
        stats_path = self.output_dir / 'stats.json'
        self._save_json(stats_path, stats)
        pnt(
            f"stats saved to {stats_path}: item_count={stats['item_count']} "
            f"finetune_samples={stats['finetune_sample_count']} test_samples={stats['test_sample_count']} "
            f"resolved_maxitems={stats['resolved_maxitems']} "
            f"sid_final_num_quantizers={stats['sid_final_num_quantizers']} "
            f"sid_max_collision_size={stats['sid_max_collision_size']} "
            f"hash_final_num_tokens={stats['hash_final_num_tokens']} "
            f"hash_max_collision_size={stats['hash_max_collision_size']}"
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
    parser.add_argument('--repr.type', dest='repr_type', default=None, help='Representation types, such as uid, text, or uid+text. Defaults to task.type.')
    parser.add_argument('--repr.source-model', dest='repr_source_model', default=None, help='External representation source model, such as bertbase.')
    parser.add_argument('--sid.export', dest='sid_export', default=None, help='Export tag for SID codes, such as coll.')
    parser.add_argument('--sid.coder', dest='sid_coder', default=None, help='Coder name for SID exports, such as rqvae, pqvae, opqvae, or basic-rqvae.')
    parser.add_argument('--hash.coder', dest='hash_coder', default=None, help='Coder/indexer name for hash exports, such as lsh, simhash, pcahash, or itq.')
    parser.add_argument('--repr.combine', dest='repr_combine', default='concat', help='How to combine multiple repr types: concat or add.')
    parser.add_argument('--model.maxlen', dest='model_max_length', type=int, default=0, help='Optional override for backbone max length, e.g. 2048.')
    parser.add_argument(
        '--task.type',
        dest='task_type',
        required=True,
        choices=['uid', 'sid', 'hash', 'embedding', 'sid+uid', 'uid+sid'],
    )
    parser.add_argument('--maxitems', type=int, default=0, help='Maximum history items, 0 means auto by model max length.')
    parser.add_argument('--item-text-max-tokens', type=int, default=50, help='Maximum tokenized length per item text.')
    args = parser.parse_args()

    config = CompileConfig(
        data=args.data.lower(),
        model=args.model.lower(),
        repr_type=args.repr_type.lower() if args.repr_type else None,
        repr_source_model=normalize_model_name(args.repr_source_model),
        sid_export=args.sid_export.lower() if args.sid_export else None,
        sid_coder=args.sid_coder.lower() if args.sid_coder else None,
        hash_coder=args.hash_coder.lower() if args.hash_coder else None,
        repr_combine=args.repr_combine.lower(),
        task_type=args.task_type.lower(),
        maxitems=int(args.maxitems),
        model_max_length=int(args.model_max_length) or None,
        item_text_max_tokens=int(args.item_text_max_tokens),
        upstreams=build_default_upstreams(vars(args)),
    )
    compiler = Compiler(config)
    compiler.run()
