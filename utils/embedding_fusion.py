import json
from copy import deepcopy

from utils.artifact import ArtifactStore
from utils.compile import normalize_model_name, short_config_hash
from utils.word2vec import normalize_word2vec_config, word2vec_model_ref


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def parse_embedding_sources(value):
    if value is None or value == 'null':
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith('['):
            value = json.loads(text)
        else:
            value = [{'model': model.strip()} for model in text.split(',') if model.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError('embedding.sources must be a list or JSON list')
    return list(value)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def normalize_embedding_fusion(value=None, legacy_model=None):
    if value is not None and not isinstance(value, dict) and callable(value):
        value = value()
    raw = deepcopy(value or {})
    sources = parse_embedding_sources(raw.get('sources'))
    if not sources and legacy_model:
        sources = [{'model': legacy_model, 'normalize': False}]
    normalized = []
    for source in sources:
        if isinstance(source, str):
            source = {'model': source}
        model = normalize_model_name(_get(source, 'model'))
        if not model:
            raise ValueError('each embedding source requires model')
        embedding_config = _get(source, 'config') or None
        if embedding_config is not None and not isinstance(embedding_config, dict) and callable(embedding_config):
            embedding_config = embedding_config()
        if model == 'word2vec' and embedding_config:
            embedding_config = normalize_word2vec_config(embedding_config)
            model = word2vec_model_ref(embedding_config)
        elif model.startswith('word2vec/') and embedding_config:
            expected = word2vec_model_ref(embedding_config)
            if model != expected:
                raise ValueError(f'word2vec source {model} does not match config {expected}')
        if embedding_config:
            embedding_config.pop('seed', None)
            embedding_config.pop('workers', None)
        normalized.append({
            'model': model,
            'normalize': _as_bool(_get(source, 'normalize'), True),
            'reduce_dim': int(_get(source, 'reduce_dim') or 0),
            'weight': float(_get(source, 'weight', 1.0)),
            **({'config': embedding_config} if embedding_config else {}),
        })
    if not normalized:
        raise ValueError('at least one embedding source is required')
    if any(source['reduce_dim'] < 0 for source in normalized):
        raise ValueError('embedding source reduce_dim must be >= 0')
    if any(source['weight'] < 0 for source in normalized) or not any(source['weight'] > 0 for source in normalized):
        raise ValueError('embedding source weights must be non-negative with at least one positive weight')
    fusion = str(raw.get('fusion') or 'concat').strip().lower()
    if fusion != 'concat':
        raise ValueError(f'unsupported embedding fusion: {fusion}; expected concat')
    return {
        'sources': normalized,
        'fusion': fusion,
        'normalize_output': _as_bool(raw.get('normalize_output'), False),
        'seed': 42,
    }


def is_legacy_single_source(spec):
    source = spec['sources'][0] if len(spec['sources']) == 1 else None
    return bool(
        source
        and not source['normalize']
        and source['reduce_dim'] == 0
        and source['weight'] == 1.0
        and not source.get('config')
        and spec['fusion'] == 'concat'
        and not spec['normalize_output']
    )


def fusion_model_ref(spec):
    if is_legacy_single_source(spec):
        return spec['sources'][0]['model']
    payload = deepcopy(spec)
    payload.pop('seed', None)
    return f'fusion/{short_config_hash(payload, length=16)}'


def _l2_normalize(values):
    import numpy as np

    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return (values / norms).astype(np.float32)


def load_fused_embeddings(data, processor, spec, ensure_embedded):
    import numpy as np
    import pandas as pd

    expected_ids = [str(item) for item in processor.items[processor.IID_COL].tolist()]
    blocks = []
    summaries = []
    positive_weight = sum(source['weight'] for source in spec['sources'])
    for source in spec['sources']:
        model = source['model']
        ensure_embedded(data, model, embedding_spec=source.get('config'))
        directory = ArtifactStore(data).embedded_dir(model)
        values = np.load(directory / 'embeddings.npy').astype(np.float32)
        frame = pd.read_parquet(directory / 'item_ids.parquet')
        column = processor.IID_COL if processor.IID_COL in frame.columns else frame.columns[0]
        source_ids = [str(item) for item in frame[column].tolist()]
        if len(source_ids) != len(values):
            raise ValueError(f'embedding rows and item ids mismatch for {data}/{model}')
        positions = {item: index for index, item in enumerate(source_ids)}
        missing = [item for item in expected_ids if item not in positions]
        if missing:
            raise ValueError(
                f'embedding source {data}/{model} misses {len(missing)} processed items; first missing: {missing[:10]}'
            )
        values = values[np.asarray([positions[item] for item in expected_ids], dtype=np.int64)]
        original_dim = int(values.shape[1])
        if source['normalize']:
            values = _l2_normalize(values)
        reduce_dim = source['reduce_dim']
        explained_variance = None
        if reduce_dim and reduce_dim < min(values.shape):
            from sklearn.decomposition import PCA

            pca = PCA(n_components=reduce_dim, svd_solver='randomized', random_state=spec['seed'])
            values = pca.fit_transform(values).astype(np.float32)
            explained_variance = float(np.sum(pca.explained_variance_ratio_))
        scale = np.sqrt(source['weight'] / positive_weight)
        blocks.append(values * scale)
        summaries.append({
            'model': model,
            'original_dim': original_dim,
            'output_dim': int(values.shape[1]),
            'normalize': source['normalize'],
            'reduce_dim': reduce_dim,
            'weight': source['weight'],
            'pca_explained_variance_ratio': explained_variance,
        })
    fused = np.concatenate(blocks, axis=1).astype(np.float32)
    if spec['normalize_output']:
        fused = _l2_normalize(fused)
    if not np.isfinite(fused).all():
        raise ValueError('fused embeddings contain NaN/Inf')
    return fused, expected_ids, summaries
