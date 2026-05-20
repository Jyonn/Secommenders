import json
from dataclasses import asdict

import torch
from pigmento import pnt
from tqdm import tqdm

from core import CompiledArtifacts, SequentialRecModel, TrainConfig
from core.dataset import CompiledSampleDataset
from utils import function
from utils.artifact import ArtifactStore
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logging import setup_logging


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.compiled = CompiledArtifacts(config).load()
        self.run_dir = ArtifactStore(config.data).trained_dir(config.run_id)
        self.device = self._resolve_device()
        self.model = SequentialRecModel(self.compiled, config).to(self.device)
        self.train_loader = None
        self.test_loader = None

    def _resolve_device(self):
        if self.config.device:
            return torch.device(self.config.device)
        return torch.device(GPU.auto_choose(torch_format=True))

    def build_dataloaders(self):
        self.train_loader, self.test_loader = function.build_dataloaders(
            CompiledSampleDataset(self.compiled.finetune),
            CompiledSampleDataset(self.compiled.test),
            batch_size=self.config.batch_size,
        )
        pnt(
            f'built dataloaders finetune={len(self.train_loader.dataset)} '
            f'test={len(self.test_loader.dataset)} batch_size={self.config.batch_size}'
        )

    def _metric_name(self):
        if self.config.task_type == 'uid':
            return 'uid_acc'
        if self.config.task_type == 'sid':
            return 'sid_seq_acc'
        return 'embedding_cosine'

    def build_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in params)
        pnt(f'build optimizer with trainable params {trainable:,}/{total:,}')
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _run_loader(self, loader, optimizer=None, desc='train'):
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss = 0.0
        total_batches = 0
        metric_sums = {}

        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            if is_train:
                optimizer.zero_grad()
            loss, metrics = self.model.forward_batch(batch)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            postfix = {'loss': f'{loss.item():.4f}'}
            for key, value in metrics.items():
                postfix[key] = f'{value:.4f}'
            iterator.set_postfix(postfix)

        summary = {'loss': total_loss / max(total_batches, 1)}
        for key, value in metric_sums.items():
            summary[key] = value / max(total_batches, 1)
        return summary

    def _save_checkpoint(self, epoch: int, best_loss: float, finetune_metrics: dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.trainable_state_dict(),
            'checkpoint_kind': 'trainable_only',
            'config': asdict(self.config),
            'finetune_metrics': finetune_metrics,
            'best_finetune_loss': best_loss,
        }
        path = self.run_dir / 'best.pt'
        torch.save(checkpoint, path)
        pnt(f'saved best checkpoint to {path} with trainable-only state')

    def _save_meta(self, best_epoch: int, best_finetune_loss: float, test_metrics: dict):
        meta = {
            'config': asdict(self.config),
            'compiled_dir': str(self.compiled.compile_dir),
            'run_dir': str(self.run_dir),
            'best_epoch': best_epoch,
            'best_finetune_loss': best_finetune_loss,
            'metric_name': self._metric_name(),
            'test_metrics': test_metrics,
        }
        path = self.run_dir / 'meta.json'
        path.write_text(json.dumps(meta, indent=2) + '\n')
        pnt(f'wrote trainer meta to {path}')

    def train(self):
        self.build_dataloaders()
        optimizer = self.build_optimizer()
        metric_name = self._metric_name()
        best_metric = float('inf')
        best_epoch = 0
        wait = 0
        unlimited_epochs = self.config.epochs <= 0

        pnt(
            f'start training on {self.config.data} with {self.config.model} '
            f'repr={self.config.repr_type} task={self.config.task_type} device={self.device} '
            f'epochs={"until-early-stop" if unlimited_epochs else self.config.epochs}'
        )

        epoch = 0
        while True:
            epoch += 1
            train_metrics = self._run_loader(self.train_loader, optimizer=optimizer, desc=f'train@{epoch}')
            pnt(
                f'epoch {epoch:03d} train_loss={train_metrics["loss"]:.4f} '
                f'{metric_name}={train_metrics.get(metric_name, 0.0):.4f}'
            )

            current_loss = train_metrics['loss']
            if current_loss < best_metric:
                best_metric = current_loss
                best_epoch = epoch
                wait = 0
                self._save_checkpoint(epoch, best_metric, train_metrics)
            else:
                wait += 1
                pnt(f'no finetune loss improvement for {wait} epoch(s), patience={self.config.patience}')
                if wait >= self.config.patience:
                    pnt(f'early stop at epoch {epoch:03d} with best_finetune_loss={best_metric:.4f}')
                    break

            if not unlimited_epochs and epoch >= self.config.epochs:
                break

        checkpoint = torch.load(self.run_dir / 'best.pt', map_location=self.device)
        load_info = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        missing = getattr(load_info, 'missing_keys', [])
        unexpected = getattr(load_info, 'unexpected_keys', [])
        if unexpected:
            raise RuntimeError(f'unexpected checkpoint keys: {unexpected}')
        pnt(f'loaded best checkpoint with {len(missing)} missing frozen/base keys')
        test_metrics = self._run_loader(self.test_loader, optimizer=None, desc='test')
        pnt(
            f'best_epoch={best_epoch} test_loss={test_metrics["loss"]:.4f} '
            f'{metric_name}={test_metrics.get(metric_name, 0.0):.4f}'
        )
        self._save_meta(best_epoch, best_metric, test_metrics)


if __name__ == '__main__':
    setup_logging()

    configurations = ConfigInit(
        required_args=[],
        default_args=dict(
            config='config/trainer.yaml',
        ),
        makedirs=[],
    ).parse()

    config = TrainConfig.from_refconfig(configurations)
    trainer = Trainer(config)
    trainer.train()
