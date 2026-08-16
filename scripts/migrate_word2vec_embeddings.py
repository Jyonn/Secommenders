#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.word2vec import normalize_word2vec_config, word2vec_model_ref  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def source_config(meta):
    word2vec = meta.get('word2vec') or {}
    return normalize_word2vec_config({
        key: word2vec.get(key)
        for key in (
            'vector_size', 'window', 'patience', 'sg', 'negative', 'min_count', 'workers', 'seed',
            'max_epochs', 'learning_rate', 'batch_size', 'valid_batch_size', 'min_delta',
        )
        if word2vec.get(key) is not None
    })


def compatible_target(target, source_embeddings, source_item_ids, config):
    required = [target / 'embeddings.npy', target / 'item_ids.parquet', target / 'meta.json']
    if not all(path.exists() for path in required):
        return False
    meta = read_json(target / 'meta.json')
    if meta.get('word2vec') != config:
        return False
    embeddings = np.load(target / 'embeddings.npy', mmap_mode='r')
    if embeddings.shape != source_embeddings.shape:
        return False
    target_ids = pd.read_parquet(target / 'item_ids.parquet').iloc[:, 0].astype(str).tolist()
    return target_ids == source_item_ids


def migrate(root: Path, data: str | None, apply: bool):
    clustered_root = root / 'artifacts' / 'clustered'
    pattern = f'{data.lower()}/*/meta.json' if data else '*/*/meta.json'
    migrated = skipped = conflicts = 0
    for meta_path in sorted(clustered_root.glob(pattern)):
        meta = read_json(meta_path)
        embedding = meta.get('embedding') or {}
        if embedding.get('source', 'collaborative') != 'collaborative':
            skipped += 1
            continue
        source_dir = meta_path.parent
        source_embeddings_path = source_dir / 'item_embeddings.npy'
        source_item_ids_path = source_dir / 'item_ids.parquet'
        if not source_embeddings_path.exists() or not source_item_ids_path.exists():
            skipped += 1
            continue
        dataset = str(meta.get('dataset') or meta.get('data')).lower()
        config = source_config(meta)
        model_ref = word2vec_model_ref(config)
        target = root / 'artifacts' / 'embedded' / dataset / model_ref
        source_embeddings = np.load(source_embeddings_path, mmap_mode='r')
        source_item_ids = pd.read_parquet(source_item_ids_path).iloc[:, 0].astype(str).tolist()
        if compatible_target(target, source_embeddings, source_item_ids, config):
            skipped += 1
            print(f'  exists {dataset}/{model_ref}')
            continue
        if target.exists() and any(target.iterdir()):
            conflicts += 1
            print(f'  conflict {dataset}/{model_ref}: target exists but is incompatible')
            continue

        print(f'  migrate {dataset}/{source_dir.name} -> embedded/{dataset}/{model_ref}')
        if not apply:
            migrated += 1
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_embeddings_path, target / 'embeddings.npy')
        shutil.copy2(source_item_ids_path, target / 'item_ids.parquet')
        target_meta = {
            'dataset': dataset,
            'model': model_ref,
            'model_key': 'word2vec',
            'item_count': len(source_item_ids),
            'embedding_dim': int(source_embeddings.shape[1]),
            'content_attrs': [],
            'processed_items_path': meta.get('processed_items_path') or f'artifacts/processed/{dataset}/items.parquet',
            'normalize': False,
            'status': 'completed',
            'source': 'migrated-clustered-word2vec',
            'word2vec': config,
            'word2vec_summary': (meta.get('word2vec') or {}).get('summary'),
            'migration': {
                'source_clustered_dir': str(source_dir),
                'source_clustered_signature': (meta.get('artifact_identity') or {}).get('signature'),
            },
        }
        (target / 'meta.json').write_text(json.dumps(target_meta, indent=2) + '\n')
        migrated += 1
    mode = 'apply' if apply else 'dry-run'
    print(f'word2vec embedding migration mode={mode} migrated={migrated} skipped={skipped} conflicts={conflicts}')
    return 1 if conflicts else 0


def main():
    parser = argparse.ArgumentParser(description='Extract reusable Word2Vec embeddings from clustered artifacts.')
    parser.add_argument('--data', default=None)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    raise SystemExit(migrate(ROOT, args.data, args.apply))


if __name__ == '__main__':
    main()
