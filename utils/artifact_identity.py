import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.compile import CompileConfig, short_config_hash


TRAINED_SPEC_VERSION = 'trained.v2'
TRAINED_INDEX_NAME = '.index.json'
LEGACY_RUN_HASH_RE = re.compile(r'__h([0-9a-fA-F]{6,64})$')


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
}

TRAIN_CONFIG_REQUIRED_FIELDS = {
    'data',
    'model',
    'repr_type',
    'task_type',
}

TRAIN_CONFIG_FIELD_NAMES = TRAIN_CONFIG_REQUIRED_FIELDS | set(TRAIN_CONFIG_DEFAULTS)


def _artifact_root(root: Path | str | None = None):
    return Path(root or '.')


def trained_dataset_dir(data: str, root: Path | str | None = None):
    path = _artifact_root(root) / 'artifacts' / 'trained' / str(data).lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def trained_index_path(data: str, root: Path | str | None = None):
    return trained_dataset_dir(data, root=root) / TRAINED_INDEX_NAME


def trained_seed(config: Any):
    return int(_config_get(config, 'seed', 42))


def trained_seed_dir_name(config: Any):
    return str(trained_seed(config))


def trained_mode(config: Any):
    if bool(_config_get(config, 'test_only', False)):
        return 'test'
    valid_only = _config_get(config, 'valid_only', 0)
    if valid_only:
        return 'valid'
    return 'train'


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


def _compile_config_from_config(config: Any):
    if not isinstance(config, dict) and hasattr(config, 'compile_config'):
        return config.compile_config
    defaults = TRAIN_CONFIG_DEFAULTS
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
    payload.pop('code_beam_chunk_size', None)
    compile_config = _compile_config_from_config(payload)
    used_views = compile_config.used_views
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
    return trained_dataset_dir(_config_data(config), root=root) / signature / trained_seed_dir_name(config)


def _resolve_alias(index: dict, signature: str):
    seen = set()
    cursor = signature
    while cursor and cursor not in seen:
        seen.add(cursor)
        entry = index.get(cursor)
        if not isinstance(entry, dict):
            return cursor, None
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
    if isinstance(entry, dict) and entry.get('folder'):
        candidate = dataset_dir / str(entry['folder']) / seed_name
        if candidate.exists():
            return candidate
    return canonical_trained_run_dir(config, root=root)


def trained_setting_folder_from_run_dir(config: Any, run_dir: Path):
    run_dir = Path(run_dir)
    seed_name = trained_seed_dir_name(config)
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
    index = load_trained_index(data, root=root)
    alias_values = _dedupe([*(aliases or []), legacy_signature_from_folder(folder)])

    existing = index.get(signature)
    if isinstance(existing, dict) and existing.get('folder') and existing.get('folder') != folder:
        raise ValueError(
            f'trained artifact signature conflict for {signature}: '
            f'{existing.get("folder")} vs {folder}'
        )
    if isinstance(existing, dict) and existing.get('alias_of') and existing.get('alias_of') != signature:
        raise ValueError(
            f'trained artifact signature conflict for {signature}: '
            f'already aliases {existing.get("alias_of")}'
        )

    index[signature] = {
        'folder': folder,
        'schema_version': TRAINED_SPEC_VERSION,
        'aliases': alias_values,
    }
    for alias in alias_values:
        if alias == signature:
            continue
        existing_alias = index.get(alias)
        if isinstance(existing_alias, dict):
            alias_of = existing_alias.get('alias_of')
            if alias_of and alias_of != signature:
                raise ValueError(f'trained artifact alias conflict for {alias}: {alias_of} vs {signature}')
            if existing_alias.get('folder') and alias != signature:
                raise ValueError(f'trained artifact alias conflict for {alias}: already registered as primary')
        index[alias] = {'alias_of': signature}
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

    return {key: config[key] for key in TRAIN_CONFIG_FIELD_NAMES}
