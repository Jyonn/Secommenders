import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pigmento import pnt
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from processors.base_processor import Processor
from utils.artifact_identity import (
    clustered_artifact_identity,
    register_clustered_artifact,
    resolve_clustered_dir,
)
from utils.compile import short_config_hash
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor
from utils.gpu import GPU
from utils.logging import setup_logging
from utils.uid_hierarchy import format_uid_cluster_levels, resolve_uid_cluster_levels


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
            patience=int(configurations.config.word2vec.patience),
            sg=int(configurations.config.word2vec.sg),
            negative=int(configurations.config.word2vec.negative),
            min_count=int(configurations.config.word2vec.min_count),
            workers=int(configurations.config.word2vec.workers),
            seed=int(configurations.config.word2vec.seed),
            cluster_batch_size=int(configurations.config.cluster.batch_size),
            cluster_max_iter=int(configurations.config.cluster.max_iter),
            cluster_n_init=int(configurations.config.cluster.n_init),
        )


class _SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_embedding.weight)

    def forward(self, center_ids: torch.Tensor, positive_ids: torch.Tensor, negative_ids: torch.Tensor):
        center_vectors = self.input_embedding(center_ids)
        positive_vectors = self.output_embedding(positive_ids)
        negative_vectors = self.output_embedding(negative_ids)

        positive_logits = (center_vectors * positive_vectors).sum(dim=-1)
        negative_logits = torch.einsum('bd,bkd->bk', center_vectors, negative_vectors)

        positive_loss = F.logsigmoid(positive_logits)
        negative_loss = F.logsigmoid(-negative_logits).sum(dim=-1)
        return -(positive_loss + negative_loss).mean()

    def export_embeddings(self):
        return self.input_embedding.weight.detach().cpu().numpy().astype(np.float32)


class Clusterer:
    VER = 'v2.0'

    WORD2VEC_MAX_EPOCHS = 100
    WORD2VEC_BATCH_SIZE = 8192
    WORD2VEC_VALID_BATCH_SIZE = 16384
    WORD2VEC_LEARNING_RATE = 3e-3
    WORD2VEC_MIN_DELTA = 1e-4

    def __init__(self, config: ClustererConfig):
        self.config = config
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

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.set_num_threads(max(1, self.config.workers))

        self.device = torch.device(GPU.auto_choose(torch_format=True))
        self.processor: Processor = load_processor(self.config.data, data_dir=get_data_dir(self.config.data))
        self.processor.load()

        self.item_ids = [str(item_id) for item_id in self.processor.items[self.processor.IID_COL].tolist()]
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}

        self.train_histories = self._load_histories(self.processor.finetune_set, split_name='finetune')
        self.valid_histories = self._load_histories(self.processor.valid_set, split_name='valid')
        self.item_frequency = self._count_item_frequency(self.train_histories)
        self.seen_train_items = self._collect_seen_items(self.train_histories)
        self.train_pair_count = self._count_positive_pairs(self.train_histories, window=self.config.window)
        self.valid_pair_count = self._count_positive_pairs(self.valid_histories, window=self.config.window)

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

    @staticmethod
    def _collect_seen_items(histories: list[list[int]]):
        seen = set()
        for history in histories:
            seen.update(history)
        return seen

    def _count_item_frequency(self, histories: list[list[int]]):
        counter = Counter()
        for history in histories:
            for item_index in history:
                counter[self.item_ids[item_index]] += 1
        return counter

    @staticmethod
    def _count_positive_pairs(histories: list[list[int]], window: int | None = None):
        if window is None:
            raise ValueError('window is required')
        total = 0
        for history in histories:
            history_length = len(history)
            for center_pos in range(history_length):
                total += min(window, center_pos) + min(window, history_length - center_pos - 1)
        return total

    @property
    def hierarchy_depth(self):
        return len(self.resolved_levels) + 1

    def _build_prepare_id(self):
        payload = asdict(self.config).copy()
        payload.pop('seed', None)
        payload['resolved_levels'] = self.resolved_levels
        payload_hash = short_config_hash(payload)
        return (
            f'ptw2v__lv{format_uid_cluster_levels(self.resolved_levels)}'
            f'__d{self.config.vector_size}__w{self.config.window}__p{self.config.patience}'
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

    def _iter_pair_batches(self, histories: list[list[int]], batch_size: int, shuffle: bool, seed_offset: int):
        order = np.arange(len(histories))
        if shuffle:
            rng = np.random.default_rng(self.config.seed + seed_offset)
            rng.shuffle(order)

        center_buffer: list[int] = []
        positive_buffer: list[int] = []

        for history_index in order.tolist():
            history = histories[history_index]
            history_length = len(history)
            for center_pos, center_item in enumerate(history):
                left = max(0, center_pos - self.config.window)
                right = min(history_length, center_pos + self.config.window + 1)
                for context_pos in range(left, right):
                    if context_pos == center_pos:
                        continue
                    center_buffer.append(center_item)
                    positive_buffer.append(history[context_pos])
                    if len(center_buffer) >= batch_size:
                        yield (
                            np.asarray(center_buffer, dtype=np.int64),
                            np.asarray(positive_buffer, dtype=np.int64),
                        )
                        center_buffer.clear()
                        positive_buffer.clear()

        if center_buffer:
            yield (
                np.asarray(center_buffer, dtype=np.int64),
                np.asarray(positive_buffer, dtype=np.int64),
            )

    def _make_negative_ids(self, batch_size: int, generator: torch.Generator):
        negatives = torch.randint(
            low=0,
            high=len(self.item_ids),
            size=(batch_size, self.config.negative),
            generator=generator,
            device='cpu',
        )
        return negatives.to(self.device, non_blocking=self.device.type == 'cuda')

    def _run_word2vec_epoch(
            self,
            model: _SkipGramNegativeSampling,
            histories: list[list[int]],
            pair_count: int,
            batch_size: int,
            epoch_index: int,
            mode: str,
            optimizer: torch.optim.Optimizer | None,
    ):
        if pair_count <= 0:
            raise ValueError(f'No positive pairs available for word2vec {mode}')

        is_train = mode == 'train'
        model.train(is_train)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.config.seed + (epoch_index if is_train else 0) + (0 if is_train else 100_000))

        total_loss = 0.0
        total_pairs = 0
        progress = tqdm(
            total=pair_count,
            desc=f'w2v-{mode}@{epoch_index}',
            leave=False,
        )

        context = torch.enable_grad if is_train else torch.no_grad
        with context():
            for center_ids_np, positive_ids_np in self._iter_pair_batches(
                    histories=histories,
                    batch_size=batch_size,
                    shuffle=is_train,
                    seed_offset=epoch_index,
            ):
                batch_pairs = int(len(center_ids_np))
                center_ids = torch.from_numpy(center_ids_np).to(self.device, non_blocking=self.device.type == 'cuda')
                positive_ids = torch.from_numpy(positive_ids_np).to(self.device, non_blocking=self.device.type == 'cuda')
                negative_ids = self._make_negative_ids(batch_pairs, generator)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                loss = model(center_ids, positive_ids, negative_ids)
                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item()) * batch_pairs
                total_pairs += batch_pairs
                progress.update(batch_pairs)
                progress.set_postfix(loss=f'{loss.item():.4f}')

        progress.close()
        return total_loss / max(total_pairs, 1)

    def _fit_word2vec(self):
        pnt(
            f'training PyTorch word2vec on {len(self.train_histories)} finetune histories '
            f'and validating on {len(self.valid_histories)} valid histories '
            f'for {self.config.data} with vector_size={self.config.vector_size} window={self.config.window} '
            f'device={self.device} patience={self.config.patience}'
        )
        pnt(
            f'word2vec pairs train={self.train_pair_count} valid={self.valid_pair_count} '
            f'batch_size={self.WORD2VEC_BATCH_SIZE} valid_batch_size={self.WORD2VEC_VALID_BATCH_SIZE}'
        )

        model = _SkipGramNegativeSampling(
            vocab_size=len(self.item_ids),
            embedding_dim=self.config.vector_size,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.WORD2VEC_LEARNING_RATE)

        best_state = None
        best_epoch = 0
        best_valid_loss = float('inf')
        stale_epochs = 0

        for epoch_index in range(1, self.WORD2VEC_MAX_EPOCHS + 1):
            train_loss = self._run_word2vec_epoch(
                model=model,
                histories=self.train_histories,
                pair_count=self.train_pair_count,
                batch_size=self.WORD2VEC_BATCH_SIZE,
                epoch_index=epoch_index,
                mode='train',
                optimizer=optimizer,
            )
            valid_loss = self._run_word2vec_epoch(
                model=model,
                histories=self.valid_histories,
                pair_count=self.valid_pair_count,
                batch_size=self.WORD2VEC_VALID_BATCH_SIZE,
                epoch_index=epoch_index,
                mode='valid',
                optimizer=None,
            )

            improved = valid_loss < (best_valid_loss - self.WORD2VEC_MIN_DELTA)
            if improved:
                best_valid_loss = valid_loss
                best_epoch = epoch_index
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1

            pnt(
                f'word2vec epoch {epoch_index}/{self.WORD2VEC_MAX_EPOCHS} '
                f'train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} '
                f'best_valid_loss={best_valid_loss:.4f} stale={stale_epochs}/{self.config.patience}'
            )

            if stale_epochs >= self.config.patience:
                pnt(f'word2vec early stop triggered at epoch {epoch_index}, best_epoch={best_epoch}')
                break

        if best_state is None:
            raise RuntimeError('word2vec training did not produce a checkpoint')

        model.load_state_dict(best_state)
        embeddings = model.export_embeddings()

        missing = 0
        for item_index in range(len(self.item_ids)):
            if item_index not in self.seen_train_items:
                embeddings[item_index] = 0.0
                missing += 1
        if missing:
            pnt(f'word2vec unseen train items for {missing} items, filled with zeros')

        self.word2vec_summary = {
            'algorithm': 'pytorch-sgns',
            'device': str(self.device),
            'max_epochs': self.WORD2VEC_MAX_EPOCHS,
            'patience': self.config.patience,
            'learning_rate': self.WORD2VEC_LEARNING_RATE,
            'batch_size': self.WORD2VEC_BATCH_SIZE,
            'valid_batch_size': self.WORD2VEC_VALID_BATCH_SIZE,
            'min_delta': self.WORD2VEC_MIN_DELTA,
            'best_epoch': best_epoch,
            'best_valid_loss': best_valid_loss,
            'train_history_count': len(self.train_histories),
            'valid_history_count': len(self.valid_histories),
            'train_pair_count': self.train_pair_count,
            'valid_pair_count': self.valid_pair_count,
        }
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
