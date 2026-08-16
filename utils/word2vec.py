from copy import deepcopy

from utils.compile import short_config_hash


WORD2VEC_DEFAULTS = {
    'vector_size': 64,
    'window': 5,
    'patience': 5,
    'sg': 1,
    'negative': 5,
    'min_count': 1,
    'workers': 4,
    'seed': 42,
    'max_epochs': 100,
    'learning_rate': 0.003,
    'batch_size': 8192,
    'valid_batch_size': 16384,
    'min_delta': 0.0001,
}


def normalize_word2vec_config(value=None):
    config = deepcopy(WORD2VEC_DEFAULTS)
    config.update({key: item for key, item in (value or {}).items() if item is not None})
    integer_keys = {
        'vector_size', 'window', 'patience', 'sg', 'negative', 'min_count', 'workers', 'seed',
        'max_epochs', 'batch_size', 'valid_batch_size',
    }
    for key in integer_keys:
        config[key] = int(config[key])
    for key in {'learning_rate', 'min_delta'}:
        config[key] = float(config[key])
    return config


def word2vec_signature(config=None):
    payload = normalize_word2vec_config(config)
    payload.pop('seed', None)
    payload.pop('workers', None)
    return short_config_hash({'model': 'word2vec', 'config': payload}, length=16)


def word2vec_model_ref(config=None):
    return f'word2vec/{word2vec_signature(config)}'
