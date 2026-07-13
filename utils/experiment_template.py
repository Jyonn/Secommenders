from copy import deepcopy

from utils.compile import normalize_model_name


SID_QUANTIZER_CONFIG_DEFAULTS = {
    'latent_dim': 64,
    'reconstruction_loss': 'mse',
    'codebook_size': 256,
    'commitment_weight': 0.25,
    'codebook_weight': 1.0,
    'use_ema_codebook': True,
    'ema_decay': 0.99,
    'ema_epsilon': 0.00001,
    'dead_code_reset': False,
    'dead_code_threshold': 0,
    'num_quantizers': 3,
    'num_codebooks': 3,
    'assignment_strategy': 'sinkhorn',
    'sinkhorn_epsilon': [0.0, 0.0, 0.003],
    'sinkhorn_iters': 50,
    'kmeans_init': True,
    'kmeans_iters': 100,
}

HASH_QUANTIZER_CONFIG_DEFAULTS = {
    'num_bits': 24,
    'num_tables': 1,
    'projection_distribution': 'gaussian',
    'use_median_thresholds': True,
    'num_iterations': 50,
    'normalize_inputs': True,
}

SID_ENCODER_CONFIG_DEFAULTS = {
    'hidden_dims': [2048, 1024, 512, 256, 128],
    'activation': 'relu',
    'use_bias': True,
}

QUANTIZER_TRAINER_DEFAULTS = {
    'validation_ratio': 0.1,
    'test_ratio': 0.1,
    'full_dataset_as_splits': True,
    'epochs': 0,
    'batch_size': 1000,
    'learning_rate': 0.001,
    'patience': 50,
    'save_best_by': ['loss', 'coll', 'codes', 'recon'],
}

CLUSTERER_WORD2VEC_DEFAULTS = {
    'vector_size': 64,
    'window': 5,
    'patience': 5,
    'sg': 1,
    'negative': 5,
    'min_count': 1,
    'workers': 4,
}

CLUSTERER_CONFIG_DEFAULTS = {
    'batch_size': 4096,
    'max_iter': 100,
    'n_init': 10,
}

CLUSTERER_EMBEDDING_DEFAULTS = {
    'source': 'collaborative',
    'content_model': None,
    'content_reduce_dim': 128,
    'normalize_blocks': True,
    'mix_alpha': 0.5,
}


def normalize_optional_string(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'null':
        return None
    return text


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes', 'y'}:
        return True
    if text in {'false', '0', 'no', 'n'}:
        return False
    return value


def normalize_list(value, *, cast=None):
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    if cast is None:
        return parts
    return [cast(part) for part in parts]


def _coerce_like(value, default):
    if default is None:
        return value
    if isinstance(default, bool):
        return normalize_bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if default and isinstance(default[0], int):
            return normalize_list(value, cast=int)
        if default and isinstance(default[0], float):
            return normalize_list(value, cast=float)
        return normalize_list(value)
    if isinstance(default, str):
        return str(value).strip().lower()
    return value


def merge_defaults(defaults: dict, overrides: dict | None):
    payload = deepcopy(defaults)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        payload[key] = _coerce_like(value, defaults.get(key))
    return payload


def normalize_metrics(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(',') if part.strip()]
    return [str(metric).strip().lower() for metric in list(value) if str(metric).strip()]


def build_default_upstreams(flat: dict):
    repr_source_model = normalize_model_name(flat.get('repr_source_model'))
    sid_quantizer_name = str(flat.get('sid_quantizer_name') or flat.get('sid_coder') or 'rqvae').strip().lower()
    hash_quantizer_name = str(flat.get('hash_quantizer_name') or flat.get('hash_coder') or 'simhash').strip().lower()
    uid_cluster_levels = normalize_optional_string(flat.get('uid_cluster_levels')) or 'auto'
    uid_cluster_embedding = merge_defaults(CLUSTERER_EMBEDDING_DEFAULTS, flat.get('uid_cluster_embedding') or {})
    uid_cluster_embedding['content_model'] = normalize_model_name(uid_cluster_embedding.get('content_model'))
    uid_clusterer = {
        'levels': uid_cluster_levels,
        'embedding': uid_cluster_embedding,
        'word2vec': merge_defaults(CLUSTERER_WORD2VEC_DEFAULTS, flat.get('uid_cluster_word2vec') or {}),
        'cluster': merge_defaults(CLUSTERER_CONFIG_DEFAULTS, flat.get('uid_cluster_config') or {}),
    }
    return {
        'sid': {
            'kind': 'quantized',
            'embedding_model': normalize_model_name(flat.get('sid_embedding_model')) or repr_source_model,
            'export': str(flat.get('sid_export') or 'coll').strip().lower(),
            'quantizer': {
                'name': sid_quantizer_name,
                'config': merge_defaults(SID_QUANTIZER_CONFIG_DEFAULTS, flat.get('sid_quantizer_config') or {}),
            },
            'encoder': {
                'name': str(flat.get('sid_encoder_name') or 'mlp').strip().lower(),
                'config': merge_defaults(SID_ENCODER_CONFIG_DEFAULTS, flat.get('sid_encoder_config') or {}),
            },
            'trainer': merge_defaults(QUANTIZER_TRAINER_DEFAULTS, flat.get('sid_quantizer_trainer') or {}),
        },
        'hash': {
            'kind': 'quantized',
            'embedding_model': normalize_model_name(flat.get('hash_embedding_model')) or repr_source_model,
            'export': 'hash',
            'quantizer': {
                'name': hash_quantizer_name,
                'config': merge_defaults(HASH_QUANTIZER_CONFIG_DEFAULTS, flat.get('hash_quantizer_config') or {}),
            },
        },
        'uid': {
            'kind': 'clustered',
            'clusterer': uid_clusterer,
        },
    }


def used_upstreams_for_config(task_type: str, repr_type: str, uid_decoding: str):
    parts = [part.strip().lower() for part in str(repr_type or '').split('+') if part.strip()]
    used_views = set(parts + [str(task_type).strip().lower()])
    used = set()
    if 'sid' in used_views:
        used.add('sid')
    if 'hash' in used_views:
        used.add('hash')
    if str(task_type).strip().lower() == 'uid' and str(uid_decoding).strip().lower() == 'hierarchical':
        used.add('uid')
    return used
