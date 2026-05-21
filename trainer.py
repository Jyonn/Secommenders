import os
import json
import random
import socket
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pigmento import pnt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from core import CompiledArtifacts, SequentialRecModel, TrainConfig
from core.dataset import CompiledFinetuneTrajectoryDataset, CompiledTestSampleDataset
from utils import function
from utils.artifact import ArtifactStore
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logging import setup_logging


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = self.world_size > 1
        self._init_distributed()
        self.compiled = CompiledArtifacts(config).load()
        self.run_dir = ArtifactStore(config.data).trained_dir(config.run_id)
        self.device = self._resolve_device()
        self.model_core = SequentialRecModel(self.compiled, config).to(self.device)
        if self.distributed:
            ddp_kwargs = {
                'module': self.model_core,
                'find_unused_parameters': True,
            }
            if self.device.type == 'cuda':
                ddp_kwargs['device_ids'] = [self.local_rank]
                ddp_kwargs['output_device'] = self.local_rank
            self.model = DDP(**ddp_kwargs)
        else:
            self.model = self.model_core
        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None
        self.alignment_rng = random.Random(self.config.seed + self.rank)
        self.alignment_step = 0
        self.alignment_sources = self._resolve_alignment_sources()
        if not self.alignment_sources:
            self.config.alignment_enable = False

    @property
    def is_main_process(self):
        return self.rank == 0

    def _pnt(self, text: str):
        if self.is_main_process:
            pnt(text)

    def _resolve_alignment_sources(self):
        sources = []
        candidate_views = []
        for view_name in self.config.compile_config.repr_types:
            if view_name not in candidate_views:
                candidate_views.append(view_name)
        if 'text' not in candidate_views:
            candidate_views.append('text')
        for view_name in candidate_views:
            if view_name == self.config.task_type:
                continue
            if view_name not in self.compiled.item_views:
                continue
            sources.append(view_name)
        return sources

    def _sample_alignment_batch(self):
        if not self.alignment_sources:
            return None, None
        source_view = self.alignment_sources[self.alignment_step % len(self.alignment_sources)]
        self.alignment_step += 1
        item_count = self.compiled.num_items
        batch_size = min(self.config.batch_size, item_count)
        if item_count <= batch_size:
            target_uids = list(range(item_count))
        else:
            target_uids = self.alignment_rng.sample(range(item_count), batch_size)
        batch = [{'target_uid': int(uid)} for uid in target_uids]
        return batch, source_view

    def _init_distributed(self):
        if not self.distributed:
            return
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

    def _resolve_device(self):
        if self.distributed:
            if torch.cuda.is_available():
                return torch.device(f'cuda:{self.local_rank}')
            return torch.device('cpu')
        if self.config.device:
            return torch.device(self.config.device)
        return torch.device(GPU.auto_choose(torch_format=True))

    def build_dataloaders(self):
        train_dataset = CompiledFinetuneTrajectoryDataset(self.compiled.finetune)
        test_dataset = CompiledTestSampleDataset(self.compiled.test)
        if self.distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
        self.train_loader, self.test_loader = function.build_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=self.config.batch_size,
            train_sampler=self.train_sampler,
        )
        if not self.is_main_process:
            self.test_loader = None
        self._pnt(
            f'built dataloaders finetune={len(self.train_loader.dataset)} '
            f'test={len(test_dataset)} batch_size={self.config.batch_size} '
            f'world_size={self.world_size}'
        )
        if self.config.task_type == 'sid':
            self._pnt(
                f'sid test uses constrained beam search beam_width={self.config.sid_beam_width} '
                f'ks={self.model_core.sid_ranking_ks()}'
            )
        if self.config.alignment_enable:
            self._pnt(
                f'alignment enabled weight={self.config.alignment_weight:g} '
                f'sources={self.alignment_sources} sampler=uniform'
            )

    def _metric_name(self):
        if self.config.task_type == 'uid':
            return 'uid_acc'
        if self.config.task_type == 'sid':
            return 'sid_token_acc'
        return 'embedding_cosine'

    def _test_metric_name(self):
        return f'ndcg@{max(self.model_core.ranking_ks())}'

    def build_optimizer(self):
        params = [param for param in self.model_core.parameters() if param.requires_grad]
        total = sum(param.numel() for param in self.model_core.parameters())
        trainable = sum(param.numel() for param in params)
        self._pnt('trainable parameters:')
        for label, shape, _, example in function.summarize_trainable_parameters(self.model_core.named_parameters()):
            self._pnt(f'  {label}: {shape}')
            if label != example:
                self._pnt(f'    example: {example}')
        if self.compiled.model_kind == 'llm':
            self._pnt('text/model token embeddings come from the LLM backbone input embedding table')
            if self.model_core.freeze_backbone:
                self._pnt('that embedding table is frozen with the backbone; current trainable text-path params are LoRA adapters only')
            else:
                self._pnt('that embedding table is trainable because the backbone is not frozen')
        self._pnt(f'build optimizer with trainable params {trainable:,}/{total:,}')
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

        iterator = tqdm(loader, desc=desc, leave=False, disable=not self.is_main_process)
        model_runner = self.model if is_train or not self.distributed else self.model_core
        for batch in iterator:
            if is_train:
                optimizer.zero_grad()
            if is_train:
                rec_loss, metrics = model_runner(batch, mode='finetune')
                loss = rec_loss
                if self.config.alignment_enable:
                    align_batch, source_view = self._sample_alignment_batch()
                    if align_batch:
                        align_loss, align_metrics = model_runner(align_batch, mode='alignment', source_view=source_view)
                        loss = rec_loss + self.config.alignment_weight * align_loss
                        metrics = dict(metrics)
                        metrics['rec_loss'] = float(rec_loss.item())
                        metrics['align_loss'] = float(align_loss.item())
                        for key, value in align_metrics.items():
                            metrics[f'align_{key}'] = value
            else:
                with torch.no_grad():
                    loss, metrics = model_runner(batch, mode='test')
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            for key, value in metrics.items():
                if isinstance(value, str):
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            postfix = {'loss': f'{loss.item():.4f}'}
            for key, value in metrics.items():
                if isinstance(value, str):
                    postfix[key] = value
                    continue
                postfix[key] = f'{value:.4f}'
            iterator.set_postfix(postfix)

        if self.distributed:
            totals = torch.tensor([total_loss, float(total_batches)], dtype=torch.float64, device=self.device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss = float(totals[0].item())
            total_batches = int(totals[1].item())
            reduced_metric_sums = {}
            for key, value in metric_sums.items():
                tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_metric_sums[key] = float(tensor.item())
            metric_sums = reduced_metric_sums

        summary = {'loss': total_loss / max(total_batches, 1)}
        for key, value in metric_sums.items():
            summary[key] = value / max(total_batches, 1)
        return summary

    def _save_checkpoint(self, epoch: int, best_loss: float, finetune_metrics: dict):
        if not self.is_main_process:
            return
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_core.trainable_state_dict(),
            'checkpoint_kind': 'trainable_only',
            'config': asdict(self.config),
            'finetune_metrics': finetune_metrics,
            'best_finetune_loss': best_loss,
        }
        path = self.run_dir / 'best.pt'
        torch.save(checkpoint, path)
        self._pnt(f'saved best checkpoint to {path} with trainable-only state')

    def _save_meta(self, best_epoch: int, best_finetune_loss: float, test_metrics: dict):
        if not self.is_main_process:
            return
        meta = {
            'config': asdict(self.config),
            'compiled_dir': str(self.compiled.compile_dir),
            'run_dir': str(self.run_dir),
            'best_epoch': best_epoch,
            'best_finetune_loss': best_finetune_loss,
            'train_metric_name': self._metric_name(),
            'test_metric_name': self._test_metric_name(),
            'test_metrics': test_metrics,
            'world_size': self.world_size,
        }
        path = self.run_dir / 'meta.json'
        path.write_text(json.dumps(meta, indent=2) + '\n')
        self._pnt(f'wrote trainer meta to {path}')

    def _evaluate_test_set(self, desc: str = 'test'):
        if self.distributed:
            dist.barrier()
        test_metrics = None
        if self.is_main_process:
            test_metrics = self._run_loader(self.test_loader, optimizer=None, desc=desc)
        if self.distributed:
            dist.barrier()
        return test_metrics

    def train(self):
        self.build_dataloaders()
        optimizer = self.build_optimizer()
        metric_name = self._metric_name()
        test_metric_name = self._test_metric_name()
        best_metric = float('inf')
        best_epoch = 0
        wait = 0
        unlimited_epochs = self.config.epochs <= 0

        self._pnt(
            f'start training on {self.config.data} with {self.config.model} '
            f'repr={self.config.repr_type} task={self.config.task_type} device={self.device} '
            f'world_size={self.world_size} rank={self.rank} local_rank={self.local_rank} '
            f'epochs={"until-early-stop" if unlimited_epochs else self.config.epochs} '
            f'test_eval=every-epoch'
        )

        epoch = 0
        while True:
            epoch += 1
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            train_metrics = self._run_loader(self.train_loader, optimizer=optimizer, desc=f'train@{epoch}')
            train_metric_name = metric_name if metric_name in train_metrics else (
                'sid_token_acc' if 'sid_token_acc' in train_metrics else metric_name
            )
            self._pnt(
                f'epoch {epoch:03d} train_loss={train_metrics["loss"]:.4f} '
                f'{train_metric_name}={train_metrics.get(train_metric_name, 0.0):.4f}'
            )
            epoch_test_metrics = self._evaluate_test_set(desc=f'test@{epoch}')
            if self.is_main_process:
                self._pnt(
                    f'epoch {epoch:03d} test_loss={epoch_test_metrics["loss"]:.4f} '
                    f'{test_metric_name}={epoch_test_metrics.get(test_metric_name, 0.0):.4f}'
                )

            current_loss = train_metrics['loss']
            if current_loss < best_metric:
                best_metric = current_loss
                best_epoch = epoch
                wait = 0
                self._save_checkpoint(epoch, best_metric, train_metrics)
            else:
                wait += 1
                self._pnt(f'no finetune loss improvement for {wait} epoch(s), patience={self.config.patience}')
                if wait >= self.config.patience:
                    self._pnt(f'early stop at epoch {epoch:03d} with best_finetune_loss={best_metric:.4f}')
                    break

            if not unlimited_epochs and epoch >= self.config.epochs:
                break

        if self.is_main_process:
            checkpoint = torch.load(self.run_dir / 'best.pt', map_location=self.device)
            load_info = self.model_core.load_state_dict(checkpoint['model_state_dict'], strict=False)
            missing = getattr(load_info, 'missing_keys', [])
            unexpected = getattr(load_info, 'unexpected_keys', [])
            if unexpected:
                raise RuntimeError(f'unexpected checkpoint keys: {unexpected}')
            self._pnt(f'loaded best checkpoint with {len(missing)} missing frozen/base keys')
        test_metrics = self._evaluate_test_set(desc='test')
        if self.is_main_process:
            self._pnt(
                f'best_epoch={best_epoch} test_loss={test_metrics["loss"]:.4f} '
                f'{test_metric_name}={test_metrics.get(test_metric_name, 0.0):.4f}'
            )
            self._save_meta(best_epoch, best_metric, test_metrics)

        if self.distributed:
            dist.destroy_process_group()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _requested_world_size(config: TrainConfig):
    requested = int(getattr(config, 'num_gpus', 1))
    if requested <= 0:
        requested = torch.cuda.device_count()
    if requested < 1:
        return 1
    return requested


def _distributed_env_present():
    return 'LOCAL_RANK' in os.environ or 'RANK' in os.environ or 'WORLD_SIZE' in os.environ


def _run_trainer(config: TrainConfig):
    trainer = Trainer(config)
    trainer.train()


def _spawn_worker(local_rank: int, world_size: int, config: TrainConfig):
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    _run_trainer(config)


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
    requested_world_size = _requested_world_size(config)

    if _distributed_env_present() or requested_world_size <= 1:
        _run_trainer(config)
    else:
        available_gpus = torch.cuda.device_count()
        if available_gpus < requested_world_size:
            raise RuntimeError(
                f'requested num_gpus={requested_world_size}, but only {available_gpus} CUDA devices are available'
            )
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(_find_free_port()))
        pnt(f'launch distributed training via python entrypoint with num_gpus={requested_world_size}')
        mp.spawn(
            _spawn_worker,
            args=(requested_world_size, config),
            nprocs=requested_world_size,
            join=True,
        )
