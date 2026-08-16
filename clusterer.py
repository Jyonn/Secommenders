import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pigmento import pnt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from processors.base_processor import Processor
from utils.artifact import ArtifactStore
from utils.artifact_identity import (
    clustered_artifact_identity,
    register_clustered_artifact,
    resolve_clustered_dir,
)
from utils.compile import compact_float, normalize_model_name, short_config_hash
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor
from utils.logging import setup_logging
from utils.uid_hierarchy import format_uid_cluster_levels, resolve_uid_cluster_levels
from utils.word2vec import WORD2VEC_DEFAULTS, normalize_word2vec_config, word2vec_model_ref


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes', 'y'}:
        return True
    if text in {'false', '0', 'no', 'n'}:
        return False
    raise ValueError(f'expected boolean value, got {value!r}')


def _normalize_optional_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'none', 'null'}:
        return None
    return text


def _parse_optional_int(value, default=0):
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {'none', 'null'}:
        return default
    return int(value)


def _get_config_value(section, name, default=None):
    return getattr(section, name, default) if section is not None else default


@dataclass
class ClustererConfig:
    data: str
    levels_spec: str
    vector_size: int
    window: int
    patience: int
    sg: int
    negative: int
    min_count: int
    workers: int
    seed: int
    max_epochs: int
    learning_rate: float
    word2vec_batch_size: int
    valid_batch_size: int
    min_delta: float
    embedding_source: str
    content_model: str | None
    content_reduce_dim: int
    normalize_blocks: bool
    mix_alpha: float
    cluster_batch_size: int
    cluster_max_iter: int
    cluster_n_init: int

    @classmethod
    def from_refconfig(cls, configurations):
        embedding = getattr(configurations.config, 'embedding', None)
        return cls(
            data=str(configurations.config.data.name).lower(),
            levels_spec=str(configurations.config.data.levels).strip().lower(),
            vector_size=int(configurations.config.word2vec.vector_size),
            window=int(configurations.config.word2vec.window),
            patience=int(configurations.config.word2vec.patience),
            sg=int(configurations.config.word2vec.sg),
            negative=int(configurations.config.word2vec.negative),
            min_count=int(configurations.config.word2vec.min_count),
            workers=int(configurations.config.word2vec.workers),
            seed=int(configurations.config.word2vec.seed),
            max_epochs=int(configurations.config.word2vec.max_epochs),
            learning_rate=float(configurations.config.word2vec.learning_rate),
            word2vec_batch_size=int(configurations.config.word2vec.batch_size),
            valid_batch_size=int(configurations.config.word2vec.valid_batch_size),
            min_delta=float(configurations.config.word2vec.min_delta),
            embedding_source=str(_get_config_value(embedding, 'source', 'collaborative')).strip().lower(),
            content_model=normalize_model_name(_normalize_optional_text(_get_config_value(embedding, 'content_model'))),
            content_reduce_dim=_parse_optional_int(_get_config_value(embedding, 'content_reduce_dim', 128), default=0),
            normalize_blocks=_parse_bool(_get_config_value(embedding, 'normalize_blocks', True)),
            mix_alpha=float(_get_config_value(embedding, 'mix_alpha', 0.5)),
            cluster_batch_size=int(configurations.config.cluster.batch_size),
            cluster_max_iter=int(configurations.config.cluster.max_iter),
            cluster_n_init=int(configurations.config.cluster.n_init),
        )


class Clusterer:
    VER = 'v2.0'

    def __init__(self, config: ClustererConfig):
        self.config = config
        if self.config.embedding_source not in {'collaborative', 'content', 'concat'}:
            raise ValueError(
                'cluster embedding source must be one of: collaborative, content, concat; '
                f'got {self.config.embedding_source}'
            )
        if self._uses_content and not self.config.content_model:
            raise ValueError('cluster_content_model is required when cluster_embedding_source is content or concat')
        if self.config.content_reduce_dim < 0:
            raise ValueError(f'content_reduce_dim must be >= 0, got {self.config.content_reduce_dim}')
        if self.config.embedding_source == 'concat' and not (0.0 <= self.config.mix_alpha <= 1.0):
            raise ValueError(f'mix_alpha must be in [0, 1], got {self.config.mix_alpha}')
        if self._uses_collaborative:
            if self.config.vector_size <= 0:
                raise ValueError(f'word2vec.vector_size must be positive, got {self.config.vector_size}')
            if self.config.window <= 0:
                raise ValueError(f'word2vec.window must be positive, got {self.config.window}')
            if self.config.sg != 1:
                raise ValueError(f'PyTorch clusterer only supports skip-gram (word2vec.sg=1), got {self.config.sg}')
            if self.config.min_count != 1:
                raise ValueError(
                    f'PyTorch clusterer currently expects word2vec.min_count=1 to preserve the processed item vocabulary, '
                    f'got {self.config.min_count}'
                )
            if self.config.patience <= 0:
                raise ValueError(f'word2vec.patience must be positive, got {self.config.patience}')
            if self.config.negative <= 0:
                raise ValueError(f'word2vec.negative must be positive, got {self.config.negative}')

        np.random.seed(self.config.seed)
        self.processor: Processor = load_processor(self.config.data, data_dir=get_data_dir(self.config.data))
        self.processor.load()

        self.item_ids = [str(item_id) for item_id in self.processor.items[self.processor.IID_COL].tolist()]
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}

        self.train_histories = self._load_histories(self.processor.finetune_set, split_name='finetune')
        self.item_frequency = self._count_item_frequency(self.train_histories)

        self.resolved_levels = resolve_uid_cluster_levels(self.config.levels_spec, len(self.item_ids))
        self.prepare_id = self._build_prepare_id()
        self.output_dir = resolve_clustered_dir(self.config, self.resolved_levels)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.output_dir / 'meta.json'
        self.item_ids_path = self.output_dir / 'item_ids.parquet'
        self.item_embeddings_path = self.output_dir / 'item_embeddings.npy'
        self.item_node_ids_path = self.output_dir / 'item_node_ids.npy'
        self.item_labels_path = self.output_dir / 'item_labels.npy'
        self.node_meta_path = self.output_dir / 'node_meta.parquet'
        self.child_nodes_path = self.output_dir / 'child_nodes.parquet'
        self.leaf_items_path = self.output_dir / 'leaf_items.parquet'
        self.word2vec_summary = None
        self.embedding_summary = None

    @property
    def _uses_collaborative(self):
        return self.config.embedding_source in {'collaborative', 'concat'}

    @property
    def _uses_content(self):
        return self.config.embedding_source in {'content', 'concat'}

    def _load_histories(self, frame: pd.DataFrame | None, split_name: str):
        if frame is None or frame.empty:
            raise ValueError(f'No processed {split_name} set found for {self.config.data}')

        histories = []
        dropped_short = 0
        for history in frame[self.processor.HIS_COL].tolist():
            tokenized = [self.item_index[str(item)] for item in history if str(item) in self.item_index]
            if len(tokenized) < 2:
                dropped_short += 1
                continue
            histories.append(tokenized)

        if not histories:
            raise ValueError(f'No usable {split_name} histories with length >= 2 found for {self.config.data}')
        if dropped_short:
            pnt(f'dropped {dropped_short} {split_name} histories with length < 2')
        return histories

    def _count_item_frequency(self, histories: list[list[int]]):
        counter = Counter()
        for history in histories:
            for item_index in history:
                counter[self.item_ids[item_index]] += 1
        return counter

    @property
    def hierarchy_depth(self):
        return len(self.resolved_levels) + 1

    def _build_prepare_id(self):
        payload = asdict(self.config).copy()
        payload.pop('seed', None)
        added_word2vec_fields = {
            'max_epochs': 'max_epochs',
            'learning_rate': 'learning_rate',
            'word2vec_batch_size': 'batch_size',
            'valid_batch_size': 'valid_batch_size',
            'min_delta': 'min_delta',
        }
        for field, default_key in added_word2vec_fields.items():
            if payload.get(field) == WORD2VEC_DEFAULTS[default_key]:
                payload.pop(field, None)
        payload['resolved_levels'] = self.resolved_levels
        if self.config.embedding_source == 'collaborative':
            prefix = 'ptw2v'
            details = (
                f'__d{self.config.vector_size}__w{self.config.window}__p{self.config.patience}'
            )
        elif self.config.embedding_source == 'content':
            prefix = 'ptcontent'
            details = (
                f'__cm-{self.config.content_model}'
                f'__pca{self.config.content_reduce_dim}'
            )
        else:
            prefix = 'ptmix'
            details = (
                f'__d{self.config.vector_size}__w{self.config.window}__p{self.config.patience}'
                f'__cm-{self.config.content_model}'
                f'__pca{self.config.content_reduce_dim}'
                f'__a{compact_float(self.config.mix_alpha)}'
            )
        payload_hash = short_config_hash(payload)
        return (
            f'{prefix}__lv{format_uid_cluster_levels(self.resolved_levels)}'
            f'{details}'
            f'__h{payload_hash}'
        )

    def is_cached(self):
        required_paths = [
            self.meta_path,
            self.item_ids_path,
            self.item_embeddings_path,
            self.item_node_ids_path,
            self.item_labels_path,
            self.node_meta_path,
            self.child_nodes_path,
            self.leaf_items_path,
        ]
        if not all(path.exists() for path in required_paths):
            return False
        try:
            meta = json.loads(self.meta_path.read_text())
        except json.JSONDecodeError:
            return False
        return (
            meta.get('version') == self.VER
            and meta.get('prepare_id') == self.prepare_id
            and meta.get('resolved_levels') == self.resolved_levels
            and int(meta.get('item_count', -1)) == len(self.item_ids)
        )

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (embeddings / norms).astype(np.float32)

    def _load_embedding_artifact(self, model_ref: str, *, label: str, embedding_spec=None):
        from utils.pipeline import ensure_embedded

        ensure_embedded(self.config.data, model_ref, embedding_spec=embedding_spec)
        embedding_dir = ArtifactStore(self.config.data).embedded_dir(model_ref)
        embeddings_path = embedding_dir / 'embeddings.npy'
        item_ids_path = embedding_dir / 'item_ids.parquet'
        if not embeddings_path.exists() or not item_ids_path.exists():
            raise FileNotFoundError(
                f'{label} embedding artifacts not found for {self.config.data}/{model_ref}: '
                f'{embeddings_path}, {item_ids_path}'
            )

        content_embeddings = np.load(embeddings_path).astype(np.float32)
        item_ids_frame = pd.read_parquet(item_ids_path)
        id_column = self.processor.IID_COL if self.processor.IID_COL in item_ids_frame.columns else item_ids_frame.columns[0]
        content_item_ids = [str(item_id) for item_id in item_ids_frame[id_column].tolist()]
        if len(content_item_ids) != int(content_embeddings.shape[0]):
            raise ValueError(
                f'embedded item_ids length mismatch for {self.config.data}/{model_ref}: '
                f'ids={len(content_item_ids)} embeddings={content_embeddings.shape[0]}'
            )

        content_index = {item_id: index for index, item_id in enumerate(content_item_ids)}
        missing_items = [item_id for item_id in self.item_ids if item_id not in content_index]
        if missing_items:
            preview = ', '.join(missing_items[:5])
            raise ValueError(
                f'{label} embeddings missing {len(missing_items)} processed items for '
                f'{self.config.data}/{model_ref}; first missing: {preview}'
            )

        ordered = np.asarray(
            [content_embeddings[content_index[item_id]] for item_id in self.item_ids],
            dtype=np.float32,
        )
        if not np.isfinite(ordered).all():
            raise ValueError(f'{label} embeddings contain NaN/Inf for {self.config.data}/{model_ref}')
        return ordered

    def _load_content_embeddings(self):
        return self._load_embedding_artifact(self.config.content_model, label='content')

    def _word2vec_config(self):
        return normalize_word2vec_config({
            'vector_size': self.config.vector_size,
            'window': self.config.window,
            'patience': self.config.patience,
            'sg': self.config.sg,
            'negative': self.config.negative,
            'min_count': self.config.min_count,
            'workers': self.config.workers,
            'seed': self.config.seed,
            'max_epochs': self.config.max_epochs,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.word2vec_batch_size,
            'valid_batch_size': self.config.valid_batch_size,
            'min_delta': self.config.min_delta,
        })

    def _load_collaborative_embeddings(self):
        config = self._word2vec_config()
        model_ref = word2vec_model_ref(config)
        embeddings = self._load_embedding_artifact(
            model_ref,
            label='collaborative',
            embedding_spec=config,
        )
        meta_path = ArtifactStore(self.config.data).embedded_dir(model_ref) / 'meta.json'
        meta = json.loads(meta_path.read_text())
        self.word2vec_summary = meta.get('word2vec_summary')
        return embeddings

    def _prepare_content_embeddings(self):
        content_embeddings = self._load_content_embeddings()
        original_dim = int(content_embeddings.shape[1])
        if self.config.normalize_blocks:
            content_embeddings = self._l2_normalize(content_embeddings)

        pca_dim = int(self.config.content_reduce_dim or 0)
        explained_variance = None
        if pca_dim > 0:
            max_components = min(content_embeddings.shape[0], content_embeddings.shape[1])
            if pca_dim < max_components:
                pnt(
                    f'PCA reducing content embeddings for {self.config.data}/{self.config.content_model} '
                    f'{original_dim}->{pca_dim}'
                )
                pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=self.config.seed)
                content_embeddings = pca.fit_transform(content_embeddings).astype(np.float32)
                explained_variance = float(np.sum(pca.explained_variance_ratio_))
            else:
                pnt(
                    f'skip PCA for {self.config.data}/{self.config.content_model}: '
                    f'requested {pca_dim} >= max_components {max_components}'
                )

        return content_embeddings.astype(np.float32), {
            'content_model': self.config.content_model,
            'content_original_dim': original_dim,
            'content_dim': int(content_embeddings.shape[1]),
            'content_reduce_dim': pca_dim,
            'content_pca_explained_variance_ratio': explained_variance,
        }

    def _build_cluster_embeddings(self):
        source = self.config.embedding_source
        coll_embeddings = None
        content_embeddings = None
        content_summary = None

        if self._uses_collaborative:
            pnt(f'cluster embedding source={source}: loading collaborative word2vec block')
            coll_embeddings = self._load_collaborative_embeddings()
        if self._uses_content:
            pnt(
                f'cluster embedding source={source}: loading content block '
                f'model={self.config.content_model} reduce_dim={self.config.content_reduce_dim}'
            )
            content_embeddings, content_summary = self._prepare_content_embeddings()

        if source == 'collaborative':
            embeddings = coll_embeddings
        elif source == 'content':
            embeddings = content_embeddings
        else:
            if self.config.normalize_blocks:
                coll_embeddings = self._l2_normalize(coll_embeddings)
            coll_weight = float(np.sqrt(self.config.mix_alpha))
            content_weight = float(np.sqrt(1.0 - self.config.mix_alpha))
            embeddings = np.concatenate(
                [
                    coll_embeddings * coll_weight,
                    content_embeddings * content_weight,
                ],
                axis=1,
            ).astype(np.float32)
            pnt(
                f'cluster embedding source=concat final_dim={embeddings.shape[1]} '
                f'coll_dim={coll_embeddings.shape[1]} content_dim={content_embeddings.shape[1]} '
                f'alpha={self.config.mix_alpha}'
            )

        self.embedding_summary = {
            'source': source,
            'normalize_blocks': self.config.normalize_blocks,
            'mix_alpha': self.config.mix_alpha if source == 'concat' else None,
            'collaborative_dim': int(coll_embeddings.shape[1]) if coll_embeddings is not None else None,
            'final_dim': int(embeddings.shape[1]),
        }
        if content_summary:
            self.embedding_summary.update(content_summary)
        return embeddings.astype(np.float32)

    def _embedding_config_meta(self):
        return {
            'source': self.config.embedding_source,
            'content_model': self.config.content_model,
            'content_reduce_dim': self.config.content_reduce_dim,
            'normalize_blocks': self.config.normalize_blocks,
            'mix_alpha': self.config.mix_alpha,
        }

    def _new_node(self, node_levels, node_parents, level: int, parent_node_id: int):
        node_id = len(node_levels)
        node_levels.append(level)
        node_parents.append(parent_node_id)
        return node_id

    def _cluster_group(self, embeddings: np.ndarray, item_indices: list[int], requested_clusters: int):
        if len(item_indices) <= 1 or requested_clusters <= 1:
            return {0: item_indices}
        actual_clusters = min(requested_clusters, len(item_indices))
        if actual_clusters == len(item_indices):
            ordered_items = sorted(
                item_indices,
                key=lambda item_index: (-self.item_frequency.get(self.item_ids[item_index], 0), item_index),
            )
            return {cluster_index: [item_index] for cluster_index, item_index in enumerate(ordered_items)}

        local_embeddings = embeddings[item_indices]
        kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=self.config.seed,
            batch_size=min(self.config.cluster_batch_size, len(item_indices)),
            max_iter=self.config.cluster_max_iter,
            n_init=self.config.cluster_n_init,
        )
        labels = kmeans.fit_predict(local_embeddings)
        groups = defaultdict(list)
        for item_index, label in zip(item_indices, labels.tolist()):
            groups[int(label)].append(item_index)
        ordered_labels = sorted(groups, key=lambda label: (-len(groups[label]), min(groups[label])))
        return {
            new_label: groups[old_label]
            for new_label, old_label in enumerate(ordered_labels)
        }

    def _build_hierarchy(self, embeddings: np.ndarray):
        num_items = len(self.item_ids)
        depth = self.hierarchy_depth
        item_node_ids = np.full((num_items, depth), fill_value=-1, dtype=np.int64)
        item_labels = np.full((num_items, depth), fill_value=-1, dtype=np.int64)

        node_levels: list[int] = []
        node_parents: list[int] = []
        node_child_counts: list[int] = []
        child_rows = []
        leaf_rows = []

        root_node_id = self._new_node(node_levels, node_parents, level=0, parent_node_id=-1)
        node_child_counts.append(0)

        def recurse(node_id: int, level_index: int, item_indices: list[int]):
            if level_index == len(self.resolved_levels):
                ordered_items = sorted(
                    item_indices,
                    key=lambda item_index: (-self.item_frequency.get(self.item_ids[item_index], 0), item_index),
                )
                node_child_counts[node_id] = len(ordered_items)
                for local_label, item_index in enumerate(ordered_items):
                    item_node_ids[item_index, level_index] = node_id
                    item_labels[item_index, level_index] = local_label
                    leaf_rows.append(
                        {
                            'parent_node_id': node_id,
                            'local_label': local_label,
                            'item_uid': item_index,
                        }
                    )
                return

            groups = self._cluster_group(embeddings, item_indices, self.resolved_levels[level_index])
            node_child_counts[node_id] = len(groups)
            for child_label, child_items in groups.items():
                child_node_id = self._new_node(
                    node_levels,
                    node_parents,
                    level=level_index + 1,
                    parent_node_id=node_id,
                )
                node_child_counts.append(0)
                child_rows.append(
                    {
                        'parent_node_id': node_id,
                        'child_label': child_label,
                        'child_node_id': child_node_id,
                    }
                )
                for item_index in child_items:
                    item_node_ids[item_index, level_index] = node_id
                    item_labels[item_index, level_index] = child_label
                recurse(child_node_id, level_index + 1, child_items)

        recurse(root_node_id, 0, list(range(num_items)))

        node_meta = pd.DataFrame(
            {
                'node_id': list(range(len(node_levels))),
                'level': node_levels,
                'parent_node_id': node_parents,
                'child_count': node_child_counts,
            }
        )
        child_nodes = pd.DataFrame(child_rows).sort_values(['parent_node_id', 'child_label']).reset_index(drop=True)
        leaf_items = pd.DataFrame(leaf_rows).sort_values(['parent_node_id', 'local_label']).reset_index(drop=True)
        return item_node_ids, item_labels, node_meta, child_nodes, leaf_items

    def _save_meta(self, embeddings: np.ndarray):
        identity = clustered_artifact_identity(self.config, self.resolved_levels, self.output_dir)
        meta = {
            'version': self.VER,
            'dataset': self.config.data,
            'prepare_id': self.prepare_id,
            'artifact_identity': identity,
            'item_col': self.processor.IID_COL,
            'levels_spec': self.config.levels_spec,
            'resolved_levels': self.resolved_levels,
            'num_cluster_levels': len(self.resolved_levels),
            'hierarchy_depth': self.hierarchy_depth,
            'item_count': len(self.item_ids),
            'embedding_dim': int(embeddings.shape[1]),
            'embedding': self._embedding_config_meta(),
            'embedding_summary': self.embedding_summary,
            'word2vec': {
                'algorithm': 'pytorch-sgns',
                'vector_size': self.config.vector_size,
                'window': self.config.window,
                'patience': self.config.patience,
                'sg': self.config.sg,
                'negative': self.config.negative,
                'min_count': self.config.min_count,
                'workers': self.config.workers,
                'seed': self.config.seed,
                'max_epochs': self.config.max_epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.word2vec_batch_size,
                'valid_batch_size': self.config.valid_batch_size,
                'min_delta': self.config.min_delta,
                'summary': self.word2vec_summary,
            },
            'cluster': {
                'batch_size': self.config.cluster_batch_size,
                'max_iter': self.config.cluster_max_iter,
                'n_init': self.config.cluster_n_init,
            },
            'processed_items_path': str(Path(self.processor.store_dir) / 'items.parquet'),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        register_clustered_artifact(self.config, self.resolved_levels, self.output_dir, aliases=identity.get('aliases'))

    def run(self):
        pnt(
            f'prepare uid hierarchy for {self.config.data} '
            f'levels={self.config.levels_spec} -> {self.resolved_levels} '
            f'embedding_source={self.config.embedding_source} '
            f'content_model={self.config.content_model} '
            f'content_reduce_dim={self.config.content_reduce_dim} '
            f'mix_alpha={self.config.mix_alpha} '
            f'output={self.output_dir}'
        )
        if self.is_cached():
            pnt(f'cached cluster hierarchy found at {self.output_dir}')
            return self

        embeddings = self._build_cluster_embeddings()
        item_node_ids, item_labels, node_meta, child_nodes, leaf_items = self._build_hierarchy(embeddings)

        pd.DataFrame({self.processor.IID_COL: self.item_ids}).to_parquet(self.item_ids_path, index=False)
        np.save(self.item_embeddings_path, embeddings)
        np.save(self.item_node_ids_path, item_node_ids)
        np.save(self.item_labels_path, item_labels)
        node_meta.to_parquet(self.node_meta_path, index=False)
        child_nodes.to_parquet(self.child_nodes_path, index=False)
        leaf_items.to_parquet(self.leaf_items_path, index=False)
        self._save_meta(embeddings)
        pnt(
            f'uid hierarchy saved to {self.output_dir} '
            f'with depth={self.hierarchy_depth} nodes={len(node_meta)}'
        )
        return self


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description='Build hierarchical uid decoding assets from processed user sequences.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind.')
    parser.add_argument('--uid_cluster_levels', required=True, help='Hierarchy spec such as 10, 20,20, auto,auto, auto/10,auto/10.')
    parser.add_argument(
        '--cluster_embedding_source',
        default=None,
        choices=['collaborative', 'content', 'concat'],
        help='Embedding source for clustering.',
    )
    parser.add_argument('--cluster_content_model', default=None, help='Content embedding model, such as pretrain-text.')
    parser.add_argument(
        '--cluster_content_reduce_dim',
        default=None,
        type=int,
        help='PCA dimension for content embeddings; 0 disables PCA.',
    )
    parser.add_argument('--cluster_normalize_blocks', default=None, help='Whether to L2-normalize blocks before PCA/concat.')
    parser.add_argument(
        '--cluster_mix_alpha',
        default=None,
        type=float,
        help='Collaborative block distance weight for concat source.',
    )
    parser.add_argument('--cluster_vector_size', type=int, default=None)
    parser.add_argument('--cluster_window', type=int, default=None)
    parser.add_argument('--cluster_patience', type=int, default=None)
    parser.add_argument('--cluster_negative', type=int, default=None)
    parser.add_argument('--cluster_max_epochs', type=int, default=None)
    parser.add_argument('--cluster_learning_rate', type=float, default=None)
    parser.add_argument('--cluster_word2vec_batch_size', type=int, default=None)
    parser.add_argument('--cluster_valid_batch_size', type=int, default=None)
    parser.add_argument('--cluster_min_delta', type=float, default=None)
    parser.add_argument('--config', default='config/clusterer.yaml', help='Clusterer config path.')
    args = parser.parse_args()

    kwargs = {
        'data': args.data.lower(),
        'uid_cluster_levels': args.uid_cluster_levels,
        'cluster_embedding_source': args.cluster_embedding_source,
        'cluster_content_model': args.cluster_content_model,
        'cluster_content_reduce_dim': args.cluster_content_reduce_dim,
        'cluster_normalize_blocks': args.cluster_normalize_blocks,
        'cluster_mix_alpha': args.cluster_mix_alpha,
        'cluster_vector_size': args.cluster_vector_size,
        'cluster_window': args.cluster_window,
        'cluster_patience': args.cluster_patience,
        'cluster_negative': args.cluster_negative,
        'cluster_max_epochs': args.cluster_max_epochs,
        'cluster_learning_rate': args.cluster_learning_rate,
        'cluster_word2vec_batch_size': args.cluster_word2vec_batch_size,
        'cluster_valid_batch_size': args.cluster_valid_batch_size,
        'cluster_min_delta': args.cluster_min_delta,
        'config': args.config,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    configurations = ConfigInit(
        required_args=['data'],
        default_args=dict(
            config=args.config,
        ),
        makedirs=[],
    ).parse_kwargs(kwargs)
    config = ClustererConfig.from_refconfig(configurations)
    clusterer = Clusterer(config)
    clusterer.run()
