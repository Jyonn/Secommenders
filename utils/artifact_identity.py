import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.compile import CompileConfig, normalize_model_name, short_config_hash
from utils.experiment_template import (
    CLUSTERER_CONFIG_DEFAULTS,
    CLUSTERER_WORD2VEC_DEFAULTS,
    HASH_QUANTIZER_CONFIG_DEFAULTS,
    QUANTIZER_TRAINER_DEFAULTS,
    SID_ENCODER_CONFIG_DEFAULTS,
    SID_QUANTIZER_CONFIG_DEFAULTS,
    build_default_upstreams,
    merge_defaults,
    normalize_list,
    used_upstreams_for_config,
)


TRAINED_SPEC_VERSION = 'trained.v2'
CLUSTERED_SPEC_VERSION = 'clustered.v2'
COMPILED_SPEC_VERSION = 'compiled.v2'
QUANTIZED_SPEC_VERSION = 'quantized.v2'
GENERIC_SPEC_VERSIONS = {
    'clustered': CLUSTERED_SPEC_VERSION,
    'compiled': COMPILED_SPEC_VERSION,
    'quantized': QUANTIZED_SPEC_VERSION,
}
TRAINED_INDEX_NAME = '.index.json'
GENERIC_INDEX_NAME = TRAINED_INDEX_NAME
LEGACY_RUN_HASH_RE = re.compile(r'__h([0-9a-fA-F]{6,64})$')
TRAINED_PHASES = {'precheck', 'train', 'test'}
HASH_INDEXER_NAMES = {'lsh', 'simhash', 'pcahash', 'itq'}
LEGACY_CLUSTERED_FOLDER_RE = re.compile(
    r'(?:pt)?w2v__lv(?P<levels>[0-9x,]+)__d(?P<vector_size>\d+)__w(?P<window>\d+)__(?:p|e)(?P<patience>\d+)'
)


TRAIN_CONFIG_DEFAULTS = {
    'repr_source_model': None,
    'sid_export': 'coll',
    'sid_coder': None,
    'hash_coder': None,
    'repr_combine': 'concat',
    'maxitems': 20,
    'model_max_length': None,
    'item_text_max_tokens': 20,
    'batch_size': 64,
    'accumulate_batch': 1,
    'valid_only': 0,
    'test_only': False,
    'load_ckpt': None,
    'overwrite': 'auto',
    'epochs': 0,
    'learning_rate': 0.0001,
    'weight_decay': 0.01,
    'seed': 42,
    'device': None,
    'num_gpus': 1,
    'freeze_backbone': 'auto',
    'uid_decoding': 'flat',
    'uid_cluster_levels': None,
    'uid_cluster_topk': None,
    'code_decoding': 'auto',
    'main_metric': 'loss',
    'metrics': ['ndcg@5', 'ndcg@10', 'ndcg@20', 'hr@5', 'hr@10', 'hr@20', 'mrr'],
    'patience': 3,
    'alignment_weight': 0.0,
    'code_beam_width': 20,
    'code_beam_chunk_size': None,
    'code_collision_loss_weight': 0.1,
    'model_dtype': 'auto',
    'use_lora': 'auto',
    'lora_rank': 8,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_layers': None,
    'lora_target_modules': 'all-linear',
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dropout': 0.1,
    'upstreams': None,
}

TRAIN_CONFIG_REQUIRED_FIELDS = {
    'data',
    'model',
    'repr_type',
    'task_type',
}

TRAIN_CONFIG_FIELD_NAMES = TRAIN_CONFIG_REQUIRED_FIELDS | set(TRAIN_CONFIG_DEFAULTS)


class TrainedArtifactRegistryConflict(ValueError):
    def __init__(
        self,
        message: str,
        *,
        signature: str | None = None,
        folder: str | None = None,
        alias: str | None = None,
        existing_folder: str | None = None,
    ):
        super().__init__(message)
        self.signature = signature
        self.folder = folder
        self.alias = alias
        self.existing_folder = existing_folder


class ArtifactRegistryVersionError(RuntimeError):
    pass


def _artifact_root(root: Path | str | None = None):
    return Path(root or '.')


def trained_dataset_dir(data: str, root: Path | str | None = None):
    path = _artifact_root(root) / 'artifacts' / 'trained' / str(data).lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def trained_index_path(data: str, root: Path | str | None = None):
    return trained_dataset_dir(data, root=root) / TRAINED_INDEX_NAME


def generic_dataset_dir(stage: str, data: str, root: Path | str | None = None):
    if stage not in GENERIC_SPEC_VERSIONS:
        raise ValueError(f'unsupported artifact stage: {stage}')
    path = _artifact_root(root) / 'artifacts' / stage / str(data).lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def generic_index_path(stage: str, data: str, root: Path | str | None = None):
    return generic_dataset_dir(stage, data, root=root) / GENERIC_INDEX_NAME


def trained_seed(config: Any):
    return int(_config_get(config, 'seed', 42))


def trained_seed_dir_name(config: Any):
    return str(trained_seed(config))


def trained_mode(config: Any):
    if bool(_config_get(config, 'test_only', False)):
        return 'test'
    valid_only = _config_get(config, 'valid_only', 0)
    if valid_only:
        return 'precheck'
    return 'train'


def trained_phase_dir_name(config: Any):
    return trained_mode(config)


def load_trained_index(data: str, root: Path | str | None = None):
    path = trained_index_path(data, root=root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_trained_index(data: str, index: dict, root: Path | str | None = None):
    path = trained_index_path(data, root=root)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')


def load_generic_index(stage: str, data: str, root: Path | str | None = None):
    path = generic_index_path(stage, data, root=root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_generic_index(stage: str, data: str, index: dict, root: Path | str | None = None):
    path = generic_index_path(stage, data, root=root)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')


def legacy_signature_from_folder(folder: str):
    match = LEGACY_RUN_HASH_RE.search(str(folder))
    return match.group(1).lower() if match else None


def _config_get(config: Any, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_data(config: Any):
    data = _config_get(config, 'data')
    if not data:
        raise ValueError('train config data is required')
    return str(data).lower()


def _strip_path_values(value):
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                key_text in {'path', 'output_dir'}
                or key_text.endswith('_path')
                or key_text.endswith('_dir')
                or key_text in {'device', 'seed'}
            ):
                continue
            clean[key_text] = _strip_path_values(item)
        return clean
    if isinstance(value, list):
        return [_strip_path_values(item) for item in value]
    return value


def _section_dict(section: Any):
    if section is None:
        return {}
    if isinstance(section, dict):
        return deepcopy(section)
    if callable(section):
        try:
            value = section()
            return deepcopy(value) if isinstance(value, dict) else {}
        except TypeError:
            pass
    config_method = getattr(section, 'config', None)
    if callable(config_method):
        value = config_method()
        return deepcopy(value) if isinstance(value, dict) else {}
    return {}


def _section_get(section: Any, key: str, default=None):
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _normalized_hidden_dims(value):
    dims = normalize_list(value, cast=int)
    return dims if dims is not None else SID_ENCODER_CONFIG_DEFAULTS['hidden_dims']


def _quantizer_trainer_payload(value: dict | None):
    payload = merge_defaults(QUANTIZER_TRAINER_DEFAULTS, value or {})
    payload = _strip_path_values(payload)
    payload.pop('advice', None)
    payload.pop('show_only_best_epochs', None)
    payload.pop('display_metrics', None)
    payload.pop('log_memory', None)
    save_best_by = normalize_list(payload.get('save_best_by'))
    if save_best_by is not None:
        payload['save_best_by'] = save_best_by
    return payload


def quantized_spec_from_config(data: str, embedding_model: str, config: Any):
    quantizer_section = _section_get(config, 'quantizer')
    hash_section = _section_get(config, 'hash')
    encoder_section = _section_get(config, 'encoder')
    trainer_section = _section_get(config, 'trainer')
    quantizer_name = str(_section_get(quantizer_section, 'name', '') or '').strip().lower()
    if not quantizer_name:
        raise ValueError('quantizer.name is required')

    trainer_payload = _quantizer_trainer_payload(_section_dict(trainer_section))
    payload = {
        'embedding_model': normalize_model_name(embedding_model),
        'quantizer': {
            'name': quantizer_name,
        },
        'trainer': trainer_payload,
    }
    if quantizer_name in HASH_INDEXER_NAMES:
        payload['family'] = 'hash'
        payload['quantizer']['config'] = merge_defaults(
            HASH_QUANTIZER_CONFIG_DEFAULTS,
            _section_dict(_section_get(hash_section, 'config')),
        )
        payload['quantizer']['config'].pop('seed', None)
    else:
        encoder_config = merge_defaults(
            SID_ENCODER_CONFIG_DEFAULTS,
            _section_dict(_section_get(encoder_section, 'config')),
        )
        encoder_config['hidden_dims'] = _normalized_hidden_dims(encoder_config.get('hidden_dims'))
        payload['family'] = 'sid'
        payload['quantizer']['config'] = merge_defaults(
            SID_QUANTIZER_CONFIG_DEFAULTS,
            _section_dict(_section_get(quantizer_section, 'config')),
        )
        payload['encoder'] = {
            'name': str(_section_get(encoder_section, 'name', 'mlp') or 'mlp').strip().lower(),
            'config': encoder_config,
        }
    return {
        'stage': 'quantized',
        'schema_version': QUANTIZED_SPEC_VERSION,
        'data': str(data).lower(),
        'config': payload,
    }


def quantized_spec_from_upstream(data: str, upstream: dict):
    quantizer = upstream.get('quantizer') or {}
    quantizer_name = str(quantizer.get('name') or '').strip().lower()
    if not quantizer_name:
        raise ValueError('upstream quantizer.name is required')
    trainer_payload = _quantizer_trainer_payload(upstream.get('trainer') or {})
    payload = {
        'embedding_model': normalize_model_name(upstream.get('embedding_model')),
        'quantizer': {
            'name': quantizer_name,
        },
        'trainer': trainer_payload,
    }
    if quantizer_name in HASH_INDEXER_NAMES:
        payload['family'] = 'hash'
        payload['quantizer']['config'] = merge_defaults(
            HASH_QUANTIZER_CONFIG_DEFAULTS,
            (quantizer.get('config') or {}),
        )
        payload['quantizer']['config'].pop('seed', None)
    else:
        encoder = upstream.get('encoder') or {}
        encoder_config = merge_defaults(SID_ENCODER_CONFIG_DEFAULTS, encoder.get('config') or {})
        encoder_config['hidden_dims'] = _normalized_hidden_dims(encoder_config.get('hidden_dims'))
        payload['family'] = 'sid'
        payload['quantizer']['config'] = merge_defaults(
            SID_QUANTIZER_CONFIG_DEFAULTS,
            (quantizer.get('config') or {}),
        )
        payload['encoder'] = {
            'name': str(encoder.get('name') or 'mlp').strip().lower(),
            'config': encoder_config,
        }
    return {
        'stage': 'quantized',
        'schema_version': QUANTIZED_SPEC_VERSION,
        'data': str(data).lower(),
        'config': payload,
    }


def quantized_spec_from_meta(meta: dict):
    quantizer_name = str(meta.get('quantizer_model') or meta.get('hash_model') or '').strip().lower()
    if not quantizer_name:
        raise ValueError('quantized meta missing quantizer_model/hash_model')
    trainer_args = meta.get('trainer_args') or {}
    payload = {
        'embedding_model': normalize_model_name(meta.get('embedding_model')),
        'quantizer': {
            'name': quantizer_name,
        },
        'trainer': _quantizer_trainer_payload(trainer_args),
    }
    if quantizer_name in HASH_INDEXER_NAMES or meta.get('representation_family') == 'hash':
        payload['family'] = 'hash'
        payload['quantizer']['config'] = merge_defaults(
            HASH_QUANTIZER_CONFIG_DEFAULTS,
            meta.get('hash_config') or meta.get('quantizer_config') or {},
        )
        payload['quantizer']['config'].pop('seed', None)
    else:
        encoder_config = merge_defaults(SID_ENCODER_CONFIG_DEFAULTS, meta.get('encoder_config') or {})
        encoder_config['hidden_dims'] = _normalized_hidden_dims(encoder_config.get('hidden_dims'))
        payload['family'] = 'sid'
        payload['quantizer']['config'] = merge_defaults(
            SID_QUANTIZER_CONFIG_DEFAULTS,
            meta.get('quantizer_config') or {},
        )
        payload['encoder'] = {
            'name': str(meta.get('encoder_name') or 'mlp').strip().lower(),
            'config': encoder_config,
        }
    return {
        'stage': 'quantized',
        'schema_version': QUANTIZED_SPEC_VERSION,
        'data': str(meta.get('data') or meta.get('dataset')).lower(),
        'config': payload,
    }


def clustered_spec_from_config(config: Any, resolved_levels: list[int] | None = None):
    levels = list(resolved_levels or [])
    payload = {
        'levels_spec': str(_config_get(config, 'levels_spec')).strip().lower(),
        'resolved_levels': levels,
        'word2vec': {
            'vector_size': int(_config_get(config, 'vector_size')),
            'window': int(_config_get(config, 'window')),
            'patience': int(_config_get(config, 'patience')),
            'sg': int(_config_get(config, 'sg')),
            'negative': int(_config_get(config, 'negative')),
            'min_count': int(_config_get(config, 'min_count')),
        },
        'cluster': {
            'batch_size': int(_config_get(config, 'cluster_batch_size')),
            'max_iter': int(_config_get(config, 'cluster_max_iter')),
            'n_init': int(_config_get(config, 'cluster_n_init')),
        },
    }
    return {
        'stage': 'clustered',
        'schema_version': CLUSTERED_SPEC_VERSION,
        'data': str(_config_get(config, 'data')).lower(),
        'config': payload,
    }


def _parse_clustered_levels(value: Any):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    text = str(value).strip().lower().replace('x', ',')
    return [int(part.strip()) for part in text.split(',') if part.strip().isdigit()]


def _legacy_clustered_folder_config(folder: str | None):
    match = LEGACY_CLUSTERED_FOLDER_RE.search(str(folder or ''))
    if not match:
        return {}
    levels = _parse_clustered_levels(match.group('levels'))
    return {
        'levels_spec': ','.join(str(level) for level in levels),
        'resolved_levels': levels,
        'word2vec': {
            'vector_size': int(match.group('vector_size')),
            'window': int(match.group('window')),
            'patience': int(match.group('patience')),
        },
    }


def _int_with_default(value, default):
    return int(default if value is None else value)


def clustered_spec_from_meta(meta: dict):
    word2vec = meta.get('word2vec') or {}
    cluster = meta.get('cluster') or {}
    legacy = _legacy_clustered_folder_config(meta.get('_legacy_folder') or meta.get('legacy_folder'))
    legacy_word2vec = legacy.get('word2vec') or {}
    resolved_levels = meta.get('resolved_levels') or legacy.get('resolved_levels') or []
    levels_spec = meta.get('levels_spec') or legacy.get('levels_spec')
    payload = {
        'levels_spec': str(levels_spec).strip().lower(),
        'resolved_levels': _parse_clustered_levels(resolved_levels),
        'word2vec': {
            'vector_size': _int_with_default(
                word2vec.get('vector_size', legacy_word2vec.get('vector_size')),
                CLUSTERER_WORD2VEC_DEFAULTS['vector_size'],
            ),
            'window': _int_with_default(
                word2vec.get('window', legacy_word2vec.get('window')),
                CLUSTERER_WORD2VEC_DEFAULTS['window'],
            ),
            'patience': _int_with_default(
                word2vec.get('patience', legacy_word2vec.get('patience')),
                CLUSTERER_WORD2VEC_DEFAULTS['patience'],
            ),
            'sg': _int_with_default(word2vec.get('sg'), CLUSTERER_WORD2VEC_DEFAULTS['sg']),
            'negative': _int_with_default(word2vec.get('negative'), CLUSTERER_WORD2VEC_DEFAULTS['negative']),
            'min_count': _int_with_default(word2vec.get('min_count'), CLUSTERER_WORD2VEC_DEFAULTS['min_count']),
        },
        'cluster': {
            'batch_size': _int_with_default(cluster.get('batch_size'), CLUSTERER_CONFIG_DEFAULTS['batch_size']),
            'max_iter': _int_with_default(cluster.get('max_iter'), CLUSTERER_CONFIG_DEFAULTS['max_iter']),
            'n_init': _int_with_default(cluster.get('n_init'), CLUSTERER_CONFIG_DEFAULTS['n_init']),
        },
    }
    return {
        'stage': 'clustered',
        'schema_version': CLUSTERED_SPEC_VERSION,
        'data': str(meta.get('data') or meta.get('dataset')).lower(),
        'config': payload,
    }


def compiled_spec_from_config(config: CompileConfig):
    return {
        'stage': 'compiled',
        'schema_version': COMPILED_SPEC_VERSION,
        'data': config.data,
        'config': config.config_dict,
    }


def compiled_spec_from_meta(meta: dict):
    if not isinstance(meta.get('config'), dict):
        raise ValueError('compiled meta missing config')
    config = deepcopy(meta.get('config'))
    if config.get('repr_source_model') is None and config.get('repr_model') is not None:
        config['repr_source_model'] = config.get('repr_model')
    if config.get('sid_export') is None and config.get('repr_best') is not None:
        config['sid_export'] = config.get('repr_best')
    quantizer_name = config.get('quantizer_name')
    view_text = '+'.join(
        str(value or '').strip().lower()
        for value in (config.get('repr_type'), config.get('task_type'))
    )
    if quantizer_name and 'sid' in view_text and config.get('sid_coder') is None:
        config['sid_coder'] = quantizer_name
    if quantizer_name and 'hash' in view_text and config.get('hash_coder') is None:
        config['hash_coder'] = quantizer_name
    if not config.get('upstreams'):
        config['upstreams'] = build_default_upstreams(config)
    compile_config = CompileConfig(
        data=config.get('data') or meta.get('data') or meta.get('dataset'),
        model=config.get('model'),
        repr_type=config.get('repr_type'),
        repr_source_model=config.get('repr_source_model'),
        sid_export=config.get('sid_export'),
        sid_coder=config.get('sid_coder'),
        hash_coder=config.get('hash_coder'),
        task_type=config.get('task_type'),
        maxitems=int(config.get('maxitems', 20)),
        model_max_length=config.get('model_max_length'),
        item_text_max_tokens=int(config.get('item_text_max_tokens', 20)),
        repr_combine=config.get('repr_combine', 'concat'),
        upstreams=config.get('upstreams'),
    )
    return {
        'stage': 'compiled',
        'schema_version': COMPILED_SPEC_VERSION,
        'data': str(meta.get('data') or meta.get('dataset')).lower(),
        'config': compile_config.config_dict,
    }


def generic_stage_spec(stage: str, source: Any, **kwargs):
    if stage == 'compiled':
        return compiled_spec_from_config(source) if isinstance(source, CompileConfig) else compiled_spec_from_meta(source)
    if stage == 'clustered':
        if isinstance(source, dict) and ('dataset' in source or 'data' in source) and 'word2vec' in source:
            return clustered_spec_from_meta(source)
        return clustered_spec_from_config(source, resolved_levels=kwargs.get('resolved_levels'))
    if stage == 'quantized':
        if isinstance(source, dict) and ('dataset' in source or 'data' in source) and (
            'quantizer_model' in source or 'hash_model' in source
        ):
            return quantized_spec_from_meta(source)
        return quantized_spec_from_config(kwargs.get('data'), kwargs.get('embedding_model'), source)
    raise ValueError(f'unsupported artifact stage: {stage}')


def generic_signature_from_spec(spec: dict):
    return short_config_hash(spec, length=16)


def compiled_signature_from_config(config: CompileConfig):
    return generic_signature_from_spec(compiled_spec_from_config(config))


def clustered_signature_from_config(config: Any, resolved_levels: list[int]):
    return generic_signature_from_spec(clustered_spec_from_config(config, resolved_levels))


def quantized_signature_from_config(data: str, embedding_model: str, config: Any):
    return generic_signature_from_spec(quantized_spec_from_config(data, embedding_model, config))


def quantized_signature_from_upstream(data: str, upstream: dict):
    return generic_signature_from_spec(quantized_spec_from_upstream(data, upstream))


def generic_signature_from_meta(stage: str, meta: dict):
    return generic_signature_from_spec(generic_stage_spec(stage, meta))


def _generic_schema_version(stage: str):
    return GENERIC_SPEC_VERSIONS[stage]


def _version_error(stage: str, signature: str, found: str | None, expected: str):
    found_label = found or 'missing'
    return ArtifactRegistryVersionError(
        f'{stage} artifact index version mismatch for {signature}: '
        f'index has {found_label}, current code expects {expected}. '
        f'Please run `python scripts/init_artifact_registry.py --stage {stage} --apply` to update SIGN aliases.'
    )


def _resolve_generic_index_entry(stage: str, data: str, signature: str, root: Path | str | None = None):
    expected = _generic_schema_version(stage)
    index = load_generic_index(stage, data, root=root)
    seen = set()
    cursor = signature
    while cursor and cursor not in seen:
        seen.add(cursor)
        entry = index.get(cursor)
        if not isinstance(entry, dict):
            return None
        entry_version = entry.get('schema_version')
        if entry_version != expected:
            raise _version_error(stage, cursor, entry_version, expected)
        alias_of = entry.get('alias_of')
        if alias_of:
            cursor = str(alias_of)
            continue
        return entry
    return None


def canonical_generic_artifact_dir(stage: str, data: str, signature: str, root: Path | str | None = None):
    return generic_dataset_dir(stage, data, root=root) / signature


def resolve_generic_artifact_dir(
    stage: str,
    data: str,
    signature: str,
    root: Path | str | None = None,
):
    dataset_dir = generic_dataset_dir(stage, data, root=root)
    entry = _resolve_generic_index_entry(stage, data, signature, root=root)
    if isinstance(entry, dict) and entry.get('folder'):
        candidate = dataset_dir / str(entry['folder'])
        if candidate.exists():
            return candidate
    return canonical_generic_artifact_dir(stage, data, signature, root=root)


def generic_folder_from_dir(stage: str, data: str, run_dir: Path, root: Path | str | None = None):
    dataset_dir = generic_dataset_dir(stage, data, root=root)
    try:
        return Path(run_dir).relative_to(dataset_dir).as_posix()
    except ValueError:
        return Path(run_dir).name


def generic_artifact_identity(
    stage: str,
    meta_or_spec: dict,
    folder: str,
    *,
    aliases: list[str] | None = None,
    migration_status: str = 'current',
):
    if meta_or_spec.get('stage') == stage and meta_or_spec.get('schema_version') == _generic_schema_version(stage):
        spec = meta_or_spec
    else:
        spec = generic_stage_spec(stage, meta_or_spec)
    signature = generic_signature_from_spec(spec)
    return {
        'stage': stage,
        'schema_version': _generic_schema_version(stage),
        'signature': signature,
        'folder': folder,
        'aliases': _dedupe(aliases or []),
        'migration_status': migration_status,
        'spec': spec,
    }


def register_generic_artifact(
    stage: str,
    meta_or_spec: dict,
    folder: str,
    *,
    aliases: list[str] | None = None,
    root: Path | str | None = None,
    allow_version_migration: bool = False,
):
    if meta_or_spec.get('stage') == stage and meta_or_spec.get('schema_version') == _generic_schema_version(stage):
        spec = meta_or_spec
    else:
        spec = generic_stage_spec(stage, meta_or_spec)
    signature = generic_signature_from_spec(spec)
    data = str(spec['data']).lower()
    index = load_generic_index(stage, data, root=root)
    expected = _generic_schema_version(stage)
    dataset_dir = generic_dataset_dir(stage, data, root=root)
    alias_values = _dedupe([*(aliases or []), legacy_signature_from_folder(folder)])

    def check_index_version(key: str, found: str | None):
        if found in {None, expected}:
            return
        if allow_version_migration:
            return
        raise _version_error(stage, key, found, expected)

    def folder_has_artifacts(primary_folder: str):
        folder_dir = dataset_dir / str(primary_folder)
        if not folder_dir.exists():
            return False
        for pattern in ('meta.json', '*/meta.json', '*/*/meta.json', 'exports/*/meta.json', '*/exports/*/meta.json'):
            if next(folder_dir.glob(pattern), None) is not None:
                return True
        return False

    def same_or_stale_primary(entry: dict):
        primary_folder = entry.get('folder')
        if not primary_folder:
            return False
        if primary_folder == folder:
            return True
        return not folder_has_artifacts(str(primary_folder))

    def resolve_alias_target(alias_of: str):
        seen = set()
        cursor = str(alias_of)
        chain = []
        while cursor and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            entry = index.get(cursor)
            if not isinstance(entry, dict):
                return 'stale', None, chain
            entry_version = entry.get('schema_version')
            check_index_version(cursor, entry_version)
            if entry.get('folder'):
                if same_or_stale_primary(entry):
                    return 'current', str(entry.get('folder')), chain
                return 'conflict', str(entry.get('folder')), chain
            next_alias = entry.get('alias_of')
            if not next_alias:
                return 'stale', None, chain
            if str(next_alias) == signature:
                return 'current', folder, chain
            cursor = str(next_alias)
        return 'stale', None, chain

    if allow_version_migration:
        folder_primary_keys = set()
        legacy_primary_keys = set()
        for key, entry in list(index.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get('folder') == folder:
                folder_primary_keys.add(str(key))
                if str(key) != signature:
                    legacy_primary_keys.add(str(key))
                alias_values = _dedupe([*alias_values, str(key)])
        changed = True
        while changed:
            changed = False
            for key, entry in list(index.items()):
                if not isinstance(entry, dict):
                    continue
                alias_of = entry.get('alias_of')
                if alias_of and str(alias_of) in legacy_primary_keys and str(key) not in alias_values:
                    alias_values = _dedupe([*alias_values, str(key)])
                    changed = True

    existing = index.get(signature)
    if isinstance(existing, dict):
        entry_version = existing.get('schema_version')
        check_index_version(signature, entry_version)
        if existing.get('folder') and existing.get('folder') != folder:
            existing_path = dataset_dir / str(existing.get('folder'))
            if existing_path.exists():
                raise TrainedArtifactRegistryConflict(
                    f'{stage} artifact signature conflict for {signature}: {existing.get("folder")} vs {folder}',
                    signature=signature,
                    folder=folder,
                    existing_folder=str(existing.get('folder')),
                )
        if existing.get('alias_of') and existing.get('alias_of') != signature:
            status, existing_folder, chain = resolve_alias_target(str(existing.get('alias_of')))
            if status in {'current', 'stale'}:
                alias_values = _dedupe([*alias_values, *chain])
            else:
                raise TrainedArtifactRegistryConflict(
                    f'{stage} artifact signature conflict for {signature}: already aliases {existing.get("alias_of")}',
                    signature=signature,
                    folder=folder,
                    alias=str(existing.get('alias_of')),
                    existing_folder=existing_folder,
                )

    alias_index = 0
    while alias_index < len(alias_values):
        alias = alias_values[alias_index]
        alias_index += 1
        if alias == signature:
            continue
        existing_alias = index.get(alias)
        if isinstance(existing_alias, dict):
            alias_version = existing_alias.get('schema_version')
            check_index_version(alias, alias_version)
            alias_of = existing_alias.get('alias_of')
            if alias_of and alias_of != signature:
                status, existing_folder, chain = resolve_alias_target(str(alias_of))
                if status in {'current', 'stale'}:
                    alias_values = _dedupe([*alias_values, *chain])
                elif allow_version_migration:
                    pass
                else:
                    raise TrainedArtifactRegistryConflict(
                        f'{stage} artifact alias conflict for {alias}: {alias_of} vs {signature}',
                        signature=signature,
                        folder=folder,
                        alias=alias,
                        existing_folder=existing_folder,
                    )
            if existing_alias.get('folder') and not same_or_stale_primary(existing_alias):
                raise TrainedArtifactRegistryConflict(
                    f'{stage} artifact alias conflict for {alias}: already registered as primary',
                    signature=signature,
                    folder=folder,
                    alias=alias,
                    existing_folder=str(existing_alias.get('folder')),
                )
        index[alias] = {'alias_of': signature, 'schema_version': expected}

    alias_values = _dedupe(alias for alias in alias_values if alias != signature)
    index[signature] = {
        'folder': folder,
        'schema_version': expected,
        'aliases': alias_values,
    }
    save_generic_index(stage, data, index, root=root)
    return index[signature]


def resolve_compiled_dir(config: CompileConfig, root: Path | str | None = None):
    signature = compiled_signature_from_config(config)
    return resolve_generic_artifact_dir('compiled', config.data, signature, root=root)


def compiled_artifact_identity(config: CompileConfig, run_dir: Path, *, aliases: list[str] | None = None):
    spec = compiled_spec_from_config(config)
    return generic_artifact_identity(
        'compiled',
        spec,
        generic_folder_from_dir('compiled', config.data, run_dir),
        aliases=aliases,
    )


def register_compiled_artifact(config: CompileConfig, run_dir: Path, *, aliases: list[str] | None = None):
    spec = compiled_spec_from_config(config)
    folder = generic_folder_from_dir('compiled', config.data, run_dir)
    return register_generic_artifact('compiled', spec, folder, aliases=aliases)


def resolve_clustered_dir(config: Any, resolved_levels: list[int], root: Path | str | None = None):
    signature = clustered_signature_from_config(config, resolved_levels)
    return resolve_generic_artifact_dir('clustered', str(_config_get(config, 'data')).lower(), signature, root=root)


def clustered_artifact_identity(config: Any, resolved_levels: list[int], run_dir: Path, *, aliases: list[str] | None = None):
    spec = clustered_spec_from_config(config, resolved_levels)
    data = str(_config_get(config, 'data')).lower()
    return generic_artifact_identity(
        'clustered',
        spec,
        generic_folder_from_dir('clustered', data, run_dir),
        aliases=aliases,
    )


def register_clustered_artifact(config: Any, resolved_levels: list[int], run_dir: Path, *, aliases: list[str] | None = None):
    spec = clustered_spec_from_config(config, resolved_levels)
    data = str(_config_get(config, 'data')).lower()
    folder = generic_folder_from_dir('clustered', data, run_dir)
    return register_generic_artifact('clustered', spec, folder, aliases=aliases)


def resolve_quantized_dir(data: str, embedding_model: str, config: Any, root: Path | str | None = None):
    spec = quantized_spec_from_config(data, embedding_model, config)
    signature = generic_signature_from_spec(spec)
    return resolve_generic_artifact_dir('quantized', str(data).lower(), signature, root=root)


def resolve_quantized_dir_from_upstream(data: str, upstream: dict, root: Path | str | None = None):
    spec = quantized_spec_from_upstream(data, upstream)
    signature = generic_signature_from_spec(spec)
    return resolve_generic_artifact_dir('quantized', str(data).lower(), signature, root=root)


def quantized_artifact_identity(data: str, embedding_model: str, config: Any, run_dir: Path, *, aliases: list[str] | None = None):
    spec = quantized_spec_from_config(data, embedding_model, config)
    return generic_artifact_identity(
        'quantized',
        spec,
        generic_folder_from_dir('quantized', str(data).lower(), run_dir),
        aliases=aliases,
    )


def register_quantized_artifact(data: str, embedding_model: str, config: Any, run_dir: Path, *, aliases: list[str] | None = None):
    spec = quantized_spec_from_config(data, embedding_model, config)
    folder = generic_folder_from_dir('quantized', str(data).lower(), run_dir)
    return register_generic_artifact('quantized', spec, folder, aliases=aliases)


def _compile_config_from_config(config: Any):
    if not isinstance(config, dict) and hasattr(config, 'compile_config'):
        return config.compile_config
    defaults = TRAIN_CONFIG_DEFAULTS
    upstreams = _config_get(config, 'upstreams', None)
    if not upstreams:
        upstreams = build_default_upstreams(config if isinstance(config, dict) else {
            key: _config_get(config, key) for key in TRAIN_CONFIG_FIELD_NAMES
        })
    return CompileConfig(
        data=_config_get(config, 'data'),
        model=_config_get(config, 'model'),
        repr_type=_config_get(config, 'repr_type'),
        repr_source_model=_config_get(config, 'repr_source_model', defaults['repr_source_model']),
        sid_export=_config_get(config, 'sid_export', defaults['sid_export']),
        sid_coder=_config_get(config, 'sid_coder', defaults['sid_coder']),
        hash_coder=_config_get(config, 'hash_coder', defaults['hash_coder']),
        task_type=_config_get(config, 'task_type'),
        maxitems=int(_config_get(config, 'maxitems', defaults['maxitems'])),
        model_max_length=_config_get(config, 'model_max_length', defaults['model_max_length']),
        item_text_max_tokens=int(_config_get(config, 'item_text_max_tokens', defaults['item_text_max_tokens'])),
        repr_combine=_config_get(config, 'repr_combine', defaults['repr_combine']),
        upstreams=upstreams,
    )


def _config_sign_payload(config: Any):
    if not isinstance(config, dict) and hasattr(config, 'sign_payload'):
        return config.sign_payload

    normalized = deepcopy(TRAIN_CONFIG_DEFAULTS)
    if isinstance(config, dict):
        normalized.update({key: value for key, value in config.items() if key in TRAIN_CONFIG_FIELD_NAMES})
    else:
        normalized.update({key: _config_get(config, key) for key in TRAIN_CONFIG_FIELD_NAMES})
    payload = {key: normalized.get(key) for key in TRAIN_CONFIG_FIELD_NAMES if key in normalized}
    payload.pop('device', None)
    batch_size = int(payload.get('batch_size') or TRAIN_CONFIG_DEFAULTS['batch_size'])
    accumulate_batch = int(payload.get('accumulate_batch') or TRAIN_CONFIG_DEFAULTS['accumulate_batch'])
    payload['effective_batch_size'] = batch_size * accumulate_batch
    payload.pop('seed', None)
    payload.pop('batch_size', None)
    payload.pop('accumulate_batch', None)
    payload.pop('valid_only', None)
    payload.pop('test_only', None)
    payload.pop('load_ckpt', None)
    payload.pop('overwrite', None)
    payload.pop('code_beam_chunk_size', None)
    compile_config = _compile_config_from_config(payload)
    used_views = compile_config.used_views
    if not payload.get('upstreams'):
        payload['upstreams'] = build_default_upstreams(payload)
    used_upstreams = used_upstreams_for_config(
        payload.get('task_type'),
        payload.get('repr_type'),
        payload.get('uid_decoding', TRAIN_CONFIG_DEFAULTS['uid_decoding']),
    )
    payload['upstreams'] = {
        key: value for key, value in (payload.get('upstreams') or {}).items()
        if key in used_upstreams
    }
    if not any(view in {'sid', 'hash', 'embedding'} for view in used_views):
        payload.pop('repr_source_model', None)
    if 'sid' not in used_views:
        payload.pop('sid_export', None)
        payload.pop('sid_coder', None)
    if 'hash' not in used_views:
        payload.pop('hash_coder', None)
    if payload.get('task_type') != 'uid':
        payload.pop('uid_decoding', None)
        payload.pop('uid_cluster_levels', None)
        payload.pop('uid_cluster_topk', None)
    elif payload.get('uid_decoding') != 'hierarchical':
        payload.pop('uid_cluster_levels', None)
        payload.pop('uid_cluster_topk', None)
    if payload.get('task_type') != 'sid':
        payload.pop('code_decoding', None)
        payload.pop('code_beam_width', None)
        payload.pop('code_beam_chunk_size', None)
    if not any(view in {'sid', 'hash'} for view in used_views):
        payload.pop('code_collision_loss_weight', None)
    if payload.get('repr_combine') == 'add' or len(compile_config.repr_types) <= 1:
        payload.pop('alignment_weight', None)
    if not payload.get('upstreams'):
        payload.pop('upstreams', None)
    if not payload.get('test_only'):
        payload.pop('load_ckpt', None)
    return payload


def trained_spec_from_config(config: Any):
    compile_config = _compile_config_from_config(config)
    return {
        'stage': 'trained',
        'schema_version': TRAINED_SPEC_VERSION,
        'config': _config_sign_payload(config),
        'compile_prepare_id': compile_config.prepare_id,
    }


def trained_signature_from_config(config: Any):
    return short_config_hash(trained_spec_from_config(config), length=16)


def canonical_trained_run_dir(config: Any, root: Path | str | None = None):
    signature = trained_signature_from_config(config)
    return (
        trained_dataset_dir(_config_data(config), root=root)
        / signature
        / trained_seed_dir_name(config)
        / trained_phase_dir_name(config)
    )


def _resolve_alias(index: dict, signature: str):
    seen = set()
    cursor = signature
    while cursor and cursor not in seen:
        seen.add(cursor)
        entry = index.get(cursor)
        if not isinstance(entry, dict):
            return cursor, None
        entry_version = entry.get('schema_version')
        if entry_version != TRAINED_SPEC_VERSION:
            raise ArtifactRegistryVersionError(
                f'trained artifact index version mismatch for {cursor}: '
                f'index has {entry_version or "missing"}, current code expects {TRAINED_SPEC_VERSION}. '
                'Please run `python scripts/init_artifact_registry.py --stage trained --apply` to update SIGN aliases.'
            )
        alias_of = entry.get('alias_of')
        if not alias_of:
            return cursor, entry
        cursor = str(alias_of)
    return signature, None


def resolve_trained_run_dir(config: Any, root: Path | str | None = None):
    signature = trained_signature_from_config(config)
    data = _config_data(config)
    dataset_dir = trained_dataset_dir(data, root=root)
    index = load_trained_index(data, root=root)
    _, entry = _resolve_alias(index, signature)
    seed_name = trained_seed_dir_name(config)
    phase_name = trained_phase_dir_name(config)
    if isinstance(entry, dict) and entry.get('folder'):
        candidate = dataset_dir / str(entry['folder']) / seed_name / phase_name
        if candidate.exists():
            return candidate
    return canonical_trained_run_dir(config, root=root)


def trained_setting_folder_from_run_dir(config: Any, run_dir: Path):
    run_dir = Path(run_dir)
    seed_name = trained_seed_dir_name(config)
    if run_dir.name in TRAINED_PHASES and run_dir.parent.name == seed_name:
        return run_dir.parent.parent.name
    if run_dir.name == seed_name:
        return run_dir.parent.name
    return run_dir.name


def _dedupe(values):
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def trained_artifact_identity(
    config: Any,
    run_dir: Path,
    *,
    aliases: list[str] | None = None,
    migration_status: str = 'current',
):
    signature = trained_signature_from_config(config)
    folder = trained_setting_folder_from_run_dir(config, Path(run_dir))
    original_signature = legacy_signature_from_folder(folder)
    alias_values = _dedupe([*(aliases or []), original_signature])
    return {
        'stage': 'trained',
        'schema_version': TRAINED_SPEC_VERSION,
        'signature': signature,
        'folder': folder,
        'seed': trained_seed(config),
        'mode': trained_mode(config),
        'phase': trained_phase_dir_name(config),
        'aliases': alias_values,
        'migration_status': migration_status,
        'spec': trained_spec_from_config(config),
    }


def register_trained_artifact(
    config: Any,
    run_dir: Path,
    *,
    aliases: list[str] | None = None,
    root: Path | str | None = None,
):
    signature = trained_signature_from_config(config)
    folder = trained_setting_folder_from_run_dir(config, Path(run_dir))
    data = _config_data(config)
    dataset_dir = trained_dataset_dir(data, root=root)
    index = load_trained_index(data, root=root)
    alias_values = _dedupe([*(aliases or []), legacy_signature_from_folder(folder)])

    def folder_has_run_artifacts(primary_folder: str):
        folder_dir = dataset_dir / str(primary_folder)
        if not folder_dir.exists():
            return False
        for pattern in (
            'meta.json',
            '*/meta.json',
            '*/*/meta.json',
            'best.pt',
            '*/best.pt',
            '*/*/best.pt',
        ):
            if next(folder_dir.glob(pattern), None) is not None:
                return True
        return False

    def same_or_stale_primary(entry: dict):
        primary_folder = entry.get('folder')
        if not primary_folder:
            return False
        if primary_folder == folder:
            return True
        return not folder_has_run_artifacts(str(primary_folder))

    def resolve_alias_target(alias_of: str):
        seen = set()
        cursor = str(alias_of)
        chain = []
        while cursor and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            entry = index.get(cursor)
            if not isinstance(entry, dict):
                return 'stale', None, chain
            if entry.get('folder'):
                if same_or_stale_primary(entry):
                    return 'stale', str(entry.get('folder')), chain
                return 'conflict', str(entry.get('folder')), chain
            next_alias = entry.get('alias_of')
            if not next_alias:
                return 'stale', None, chain
            if str(next_alias) == signature:
                return 'current', folder, chain
            cursor = str(next_alias)
        return 'stale', None, chain

    existing = index.get(signature)
    if isinstance(existing, dict) and existing.get('folder') and not same_or_stale_primary(existing):
        raise TrainedArtifactRegistryConflict(
            f'trained artifact signature conflict for {signature}: '
            f'{existing.get("folder")} vs {folder}',
            signature=signature,
            folder=folder,
            existing_folder=str(existing.get('folder')),
        )
    if isinstance(existing, dict) and existing.get('alias_of') and existing.get('alias_of') != signature:
        existing_alias_of = str(existing.get('alias_of'))
        status, existing_folder, chain = resolve_alias_target(existing_alias_of)
        if status in {'current', 'stale'}:
            alias_values = _dedupe([*alias_values, *chain])
        else:
            raise TrainedArtifactRegistryConflict(
                f'trained artifact signature conflict for {signature}: '
                f'already aliases {existing.get("alias_of")}',
                signature=signature,
                folder=folder,
                alias=existing_alias_of,
                existing_folder=existing_folder,
            )

    alias_index = 0
    while alias_index < len(alias_values):
        alias = alias_values[alias_index]
        alias_index += 1
        if alias == signature:
            continue
        existing_alias = index.get(alias)
        if isinstance(existing_alias, dict):
            alias_of = existing_alias.get('alias_of')
            if alias_of and alias_of != signature:
                status, existing_folder, chain = resolve_alias_target(str(alias_of))
                if status == 'stale':
                    alias_values = _dedupe([*alias_values, *chain])
                elif status == 'current':
                    alias_values = _dedupe([*alias_values, *chain])
                else:
                    raise TrainedArtifactRegistryConflict(
                        f'trained artifact alias conflict for {alias}: {alias_of} vs {signature}',
                        signature=signature,
                        folder=folder,
                        alias=alias,
                        existing_folder=existing_folder,
                    )
            if existing_alias.get('folder') and not same_or_stale_primary(existing_alias):
                raise TrainedArtifactRegistryConflict(
                    f'trained artifact alias conflict for {alias}: already registered as primary',
                    signature=signature,
                    folder=folder,
                    alias=alias,
                    existing_folder=str(existing_alias.get('folder')),
                )
        index[alias] = {'alias_of': signature, 'schema_version': TRAINED_SPEC_VERSION}
    index[signature] = {
        'folder': folder,
        'schema_version': TRAINED_SPEC_VERSION,
        'aliases': alias_values,
    }
    save_trained_index(data, index, root=root)
    return index[signature]


def migrate_train_config_dict(raw_config: dict[str, Any]):
    if not isinstance(raw_config, dict):
        raise ValueError('meta.config must be a dict')
    config = deepcopy(TRAIN_CONFIG_DEFAULTS)
    for key, value in raw_config.items():
        if key in TRAIN_CONFIG_FIELD_NAMES:
            config[key] = value

    missing_required = [key for key in ('data', 'model', 'repr_type', 'task_type') if not config.get(key)]
    if missing_required:
        raise ValueError(f'missing required train config fields: {missing_required}')

    if config.get('model_max_length') in (0, '0', ''):
        config['model_max_length'] = None
    if config.get('code_beam_chunk_size') in (None, 0, '0', ''):
        config['code_beam_chunk_size'] = int(config['batch_size'])
    if isinstance(config.get('metrics'), str):
        config['metrics'] = [part.strip().lower() for part in config['metrics'].split(',') if part.strip()]
    if isinstance(config.get('valid_only'), bool):
        config['valid_only'] = -1 if config['valid_only'] else 0
    if not config.get('upstreams'):
        config['upstreams'] = build_default_upstreams(config)

    return {key: config[key] for key in TRAIN_CONFIG_FIELD_NAMES}
