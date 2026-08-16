import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pigmento import pnt
from tqdm import tqdm

from embedders.base_model import BaseModel


class _SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_embedding.weight)

    def forward(self, center_ids, positive_ids, negative_ids):
        center_vectors = self.input_embedding(center_ids)
        positive_vectors = self.output_embedding(positive_ids)
        negative_vectors = self.output_embedding(negative_ids)
        positive_logits = (center_vectors * positive_vectors).sum(dim=-1)
        negative_logits = torch.einsum('bd,bkd->bk', center_vectors, negative_vectors)
        return -(F.logsigmoid(positive_logits) + F.logsigmoid(-negative_logits).sum(dim=-1)).mean()

    def export_embeddings(self):
        return self.input_embedding.weight.detach().cpu().numpy().astype(np.float32)


class Word2VecModel(BaseModel):
    """Collaborative item embedder trained with PyTorch SGNS."""

    KEY = 'word2vec'

    def __init__(self, device='cpu', batch_size=8192, **config):
        super().__init__(device=device, batch_size=batch_size)
        self.config = config
        self.config['batch_size'] = int(batch_size)
        if int(self.config['sg']) != 1:
            raise ValueError('word2vec embedder currently supports only skip-gram (sg=1)')
        if int(self.config['min_count']) != 1:
            raise ValueError('word2vec embedder requires min_count=1 to preserve the processed item vocabulary')
        if int(self.config['negative']) <= 0 or int(self.config['window']) <= 0:
            raise ValueError('word2vec negative and window must be positive')
        torch.set_num_threads(max(1, int(self.config['workers'])))
        self.summary = None

    def encode(self, samples, normalize=False):
        raise RuntimeError('word2vec embeds interaction histories; use fit_collaborative()')

    @staticmethod
    def _count_pairs(histories, window):
        return sum(
            min(window, pos) + min(window, len(history) - pos - 1)
            for history in histories
            for pos in range(len(history))
        )

    @staticmethod
    def _load_histories(frame, history_col, item_index, split_name):
        if frame is None or frame.empty:
            raise ValueError(f'No processed {split_name} set found for word2vec')
        histories = []
        for history in frame[history_col].tolist():
            encoded = [item_index[str(item)] for item in history if str(item) in item_index]
            if len(encoded) >= 2:
                histories.append(encoded)
        if not histories:
            raise ValueError(f'No usable {split_name} histories with length >= 2 found for word2vec')
        return histories

    def _iter_pair_batches(self, histories, window, batch_size, shuffle, seed):
        order = np.arange(len(histories))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        centers, positives = [], []
        for history_index in order.tolist():
            history = histories[history_index]
            for pos, center in enumerate(history):
                for context_pos in range(max(0, pos - window), min(len(history), pos + window + 1)):
                    if context_pos == pos:
                        continue
                    centers.append(center)
                    positives.append(history[context_pos])
                    if len(centers) >= batch_size:
                        yield np.asarray(centers, dtype=np.int64), np.asarray(positives, dtype=np.int64)
                        centers.clear()
                        positives.clear()
        if centers:
            yield np.asarray(centers, dtype=np.int64), np.asarray(positives, dtype=np.int64)

    def _run_epoch(self, model, histories, pair_count, config, epoch, mode, optimizer=None):
        is_train = mode == 'train'
        model.train(is_train)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(config['seed'] + (epoch if is_train else 100_000))
        batch_size = config['batch_size'] if is_train else config['valid_batch_size']
        total_loss = 0.0
        total_pairs = 0
        progress = tqdm(total=pair_count, desc=f'w2v-{mode}@{epoch}', leave=False)
        context = torch.enable_grad if is_train else torch.no_grad
        with context():
            batches = self._iter_pair_batches(
                histories, config['window'], batch_size, is_train, config['seed'] + epoch,
            )
            for center_np, positive_np in batches:
                count = len(center_np)
                center = torch.from_numpy(center_np).to(self.device)
                positive = torch.from_numpy(positive_np).to(self.device)
                negative = torch.randint(
                    0, config['item_count'], (count, config['negative']), generator=generator, device='cpu',
                ).to(self.device)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                loss = model(center, positive, negative)
                if is_train:
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss.item()) * count
                total_pairs += count
                progress.update(count)
        progress.close()
        return total_loss / max(total_pairs, 1)

    def fit_collaborative(self, processor, item_ids, normalize=False):
        config = dict(self.config)
        config['item_count'] = len(item_ids)
        item_index = {str(item): index for index, item in enumerate(item_ids)}
        train = self._load_histories(processor.finetune_set, processor.HIS_COL, item_index, 'finetune')
        valid = self._load_histories(processor.valid_set, processor.HIS_COL, item_index, 'valid')
        train_pairs = self._count_pairs(train, config['window'])
        valid_pairs = self._count_pairs(valid, config['window'])
        pnt(
            f'training word2vec embeddings on {len(train)} finetune histories and {len(valid)} valid histories '
            f'vector_size={config["vector_size"]} window={config["window"]} device={self.device}'
        )

        torch.manual_seed(config['seed'])
        model = _SkipGramNegativeSampling(len(item_ids), config['vector_size']).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        best_state, best_epoch, best_valid = None, 0, float('inf')
        stale = 0
        for epoch in range(1, config['max_epochs'] + 1):
            train_loss = self._run_epoch(model, train, train_pairs, config, epoch, 'train', optimizer)
            valid_loss = self._run_epoch(model, valid, valid_pairs, config, epoch, 'valid')
            if valid_loss < best_valid - config['min_delta']:
                best_valid, best_epoch, stale = valid_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            pnt(
                f'word2vec epoch {epoch}/{config["max_epochs"]} train_loss={train_loss:.4f} '
                f'valid_loss={valid_loss:.4f} best_valid_loss={best_valid:.4f} stale={stale}/{config["patience"]}'
            )
            if stale >= config['patience']:
                break
        if best_state is None:
            raise RuntimeError('word2vec training did not produce a checkpoint')
        model.load_state_dict(best_state)
        embeddings = model.export_embeddings()
        seen = set(index for history in train for index in history)
        for index in range(len(item_ids)):
            if index not in seen:
                embeddings[index] = 0.0
        if normalize:
            embeddings = self.normalize(embeddings).astype(np.float32)
        self.summary = {
            'algorithm': 'pytorch-sgns',
            'best_epoch': best_epoch,
            'best_valid_loss': best_valid,
            'train_history_count': len(train),
            'valid_history_count': len(valid),
            'train_pair_count': train_pairs,
            'valid_pair_count': valid_pairs,
            'unseen_item_count': len(item_ids) - len(seen),
        }
        return embeddings
