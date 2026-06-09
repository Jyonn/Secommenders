import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pigmento import pnt
from sklearn.cluster import MiniBatchKMeans

from processors.base_processor import Processor
from utils.artifact import ArtifactStore
from utils.compile import short_config_hash
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor
from utils.logging import setup_logging
from utils.uid_hierarchy import format_uid_cluster_levels, resolve_uid_cluster_levels


@dataclass
class ClustererConfig:
    data: str
    levels_spec: str
    vector_size: int
    window: int
    epochs: int
    sg: int
    negative: int
    min_count: int
    workers: int
    seed: int
    cluster_batch_size: int
    cluster_max_iter: int
    cluster_n_init: int

    @classmethod
    def from_refconfig(cls, configurations):
        return cls(
            data=str(configurations.config.data.name).lower(),
            levels_spec=str(configurations.config.data.levels).strip().lower(),
            vector_size=int(configurations.config.word2vec.vector_size),
            window=int(configurations.config.word2vec.window),
            epochs=int(configurations.config.word2vec.epochs),
            sg=int(configurations.config.word2vec.sg),
            negative=int(configurations.config.word2vec.negative),
            min_count=int(configurations.config.word2vec.min_count),
            workers=int(configurations.config.word2vec.workers),
            seed=int(configurations.config.word2vec.seed),
            cluster_batch_size=int(configurations.config.cluster.batch_size),
            cluster_max_iter=int(configurations.config.cluster.max_iter),
            cluster_n_init=int(configurations.config.cluster.n_init),
        )


class _HistoryCorpus:
    def __init__(self, histories):
        self.histories = histories

    def __iter__(self):
        for history in self.histories:
            yield [str(item) for item in history]


class Clusterer:
    VER = 'v1.0'

    def __init__(self, config: ClustererConfig):
        self.config = config
        self.processor: Processor = load_processor(self.config.data, data_dir=get_data_dir(self.config.data))
        self.processor.load()

        self.item_ids = [str(item_id) for item_id in self.processor.items[self.processor.IID_COL].tolist()]
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self.histories = self._load_histories()
        self.item_frequency = self._count_item_frequency(self.histories)
        self.resolved_levels = resolve_uid_cluster_levels(self.config.levels_spec, len(self.item_ids))
        self.prepare_id = self._build_prepare_id()
        self.output_dir = ArtifactStore(self.config.data).clustered_dir(self.prepare_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.output_dir / 'meta.json'
        self.item_ids_path = self.output_dir / 'item_ids.parquet'
        self.item_embeddings_path = self.output_dir / 'item_embeddings.npy'
        self.item_node_ids_path = self.output_dir / 'item_node_ids.npy'
        self.item_labels_path = self.output_dir / 'item_labels.npy'
        self.node_meta_path = self.output_dir / 'node_meta.parquet'
        self.child_nodes_path = self.output_dir / 'child_nodes.parquet'
        self.leaf_items_path = self.output_dir / 'leaf_items.parquet'

    def _load_histories(self):
        # Keep the hierarchy assets train-only to avoid leaking validation/test
        # interaction sequences into the hierarchical uid decoder.
        frame = self.processor.finetune_set
        histories = []
        if frame is not None:
            histories.extend(frame[self.processor.HIS_COL].tolist())
        if not histories:
            raise ValueError(f'No processed histories found for {self.config.data}')
        return histories

    @staticmethod
    def _count_item_frequency(histories):
        counter = Counter()
        for history in histories:
            counter.update(str(item) for item in history)
        return counter

    @property
    def hierarchy_depth(self):
        return len(self.resolved_levels) + 1

    def _build_prepare_id(self):
        payload = asdict(self.config).copy()
        payload['resolved_levels'] = self.resolved_levels
        payload_hash = short_config_hash(payload)
        return (
            f'w2v__lv{format_uid_cluster_levels(self.resolved_levels)}'
            f'__d{self.config.vector_size}__w{self.config.window}__e{self.config.epochs}'
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

    def _fit_word2vec(self):
        pnt(
            f'training word2vec on {len(self.histories)} histories '
            f'for {self.config.data} with vector_size={self.config.vector_size}'
        )
        corpus = _HistoryCorpus(self.histories)
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=self.config.sg,
            negative=self.config.negative,
            workers=self.config.workers,
            epochs=self.config.epochs,
            seed=self.config.seed,
        )
        embeddings = np.zeros((len(self.item_ids), self.config.vector_size), dtype=np.float32)
        missing = 0
        for index, item_id in enumerate(self.item_ids):
            if item_id in model.wv:
                embeddings[index] = model.wv[item_id]
            else:
                missing += 1
        if missing:
            pnt(f'word2vec missing vectors for {missing} items, filled with zeros')
        return embeddings

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
        meta = {
            'version': self.VER,
            'dataset': self.config.data,
            'prepare_id': self.prepare_id,
            'item_col': self.processor.IID_COL,
            'levels_spec': self.config.levels_spec,
            'resolved_levels': self.resolved_levels,
            'num_cluster_levels': len(self.resolved_levels),
            'hierarchy_depth': self.hierarchy_depth,
            'item_count': len(self.item_ids),
            'embedding_dim': int(embeddings.shape[1]),
            'word2vec': {
                'vector_size': self.config.vector_size,
                'window': self.config.window,
                'epochs': self.config.epochs,
                'sg': self.config.sg,
                'negative': self.config.negative,
                'min_count': self.config.min_count,
                'workers': self.config.workers,
                'seed': self.config.seed,
            },
            'cluster': {
                'batch_size': self.config.cluster_batch_size,
                'max_iter': self.config.cluster_max_iter,
                'n_init': self.config.cluster_n_init,
            },
            'processed_items_path': str(Path(self.processor.store_dir) / 'items.parquet'),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    def run(self):
        pnt(
            f'prepare uid hierarchy for {self.config.data} '
            f'levels={self.config.levels_spec} -> {self.resolved_levels} '
            f'output={self.output_dir}'
        )
        if self.is_cached():
            pnt(f'cached cluster hierarchy found at {self.output_dir}')
            return self

        embeddings = self._fit_word2vec()
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
    parser.add_argument('--config', default='config/clusterer.yaml', help='Clusterer config path.')
    args = parser.parse_args()

    configurations = ConfigInit(
        required_args=['data'],
        default_args=dict(
            config=args.config,
        ),
        makedirs=[],
    ).parse_kwargs(
        {
            'data': args.data.lower(),
            'uid_cluster_levels': args.uid_cluster_levels,
            'config': args.config,
        }
    )
    config = ClustererConfig.from_refconfig(configurations)
    clusterer = Clusterer(config)
    clusterer.run()
