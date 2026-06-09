import os
import json
import shlex
import socket
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pigmento import pnt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from core import CompiledArtifacts, SequentialRecModel, TrainConfig
from core.dataset import CompiledFinetuneTrajectoryDataset, CompiledTestSampleDataset, CompiledValidSampleDataset
from utils import function
from utils.artifact import ArtifactStore
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logging import attach_run_log, setup_logging


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = self.world_size > 1
        self._init_distributed()
        self.run_dir = ArtifactStore(config.data).trained_dir(config.run_id)
        self.compiled = CompiledArtifacts(config).load()
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
        self.valid_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None
        self.meta_path = self.run_dir / 'meta.json'
        self.pid_path = self.run_dir / 'pid.json'
        self.log_path = self.run_dir / 'train.log'

    @property
    def is_main_process(self):
        return self.rank == 0

    def _pnt(self, text: str):
        if self.is_main_process:
            pnt(text)

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
        valid_dataset = CompiledValidSampleDataset(self.compiled.valid)
        test_dataset = CompiledTestSampleDataset(self.compiled.test)
        if self.distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        self.train_loader, self.valid_loader = function.build_dataloaders(
            train_dataset,
            valid_dataset,
            batch_size=self.config.batch_size,
            train_sampler=self.train_sampler,
        )
        _, self.test_loader = function.build_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=self.config.batch_size,
        )
        if not self.is_main_process:
            self.valid_loader = None
            self.test_loader = None
        effective_batch_size = self.config.batch_size * self.config.accumulate_batch * self.world_size
        self._pnt(
            f'built dataloaders finetune={len(self.train_loader.dataset)} '
            f'valid={len(valid_dataset)} test={len(test_dataset)} batch_size={self.config.batch_size} '
            f'accumulate_batch={self.config.accumulate_batch} effective_batch_size={effective_batch_size} '
            f'world_size={self.world_size} order=length-sorted'
        )
        if self.config.task_type == 'sid':
            sid_decoding = self.model_core._sid_decoding_mode()
            self._pnt(
                f'sid decoding={sid_decoding} '
                + (
                    f'beam_width={self.config.code_beam_width} '
                    f'beam_chunk_size={self.config.code_beam_chunk_size} '
                    f'ks={self.model_core.sid_ranking_ks()}'
                    if sid_decoding == 'sequential'
                    else f'item_scoring ks={self.model_core.sid_ranking_ks()}'
                )
            )
        elif self.config.task_type == 'hash':
            self._pnt(f'hash decoding=parallel item_scoring ks={self.model_core.ranking_ks()}')
        if self.config.alignment_weight > 0 and self.config.repr_combine != 'add':
            self._pnt(
                f'alignment enabled weight={self.config.alignment_weight:g} '
                f'mode=integrated-mixed-view'
            )
        elif self.config.alignment_weight > 0 and self.config.repr_combine == 'add':
            self._pnt('alignment disabled for repr.combine=add fused-history protocol')

    def _metric_name(self):
        if self.config.task_type == 'uid':
            return 'uid_acc'
        if self.config.task_type == 'sid':
            return 'sid_token_acc'
        if self.config.task_type == 'hash':
            return 'hash_token_acc'
        return 'embedding_cosine'

    def _default_ranking_metric(self):
        return f'ndcg@{max(self.model_core.ranking_ks())}'

    def _main_metric_name(self):
        return self.config.main_metric or 'loss'

    def _main_metric_higher_is_better(self):
        metric = self._main_metric_name()
        return metric != 'loss'

    def _format_test_metrics(self, metrics: dict):
        selected_keys = [metric for metric in self.config.metrics if metric in metrics]
        if not selected_keys:
            fallback_metric = self._default_ranking_metric()
            selected_keys = [fallback_metric] if fallback_metric in metrics else ['loss']
        parts = [f'{key}={metrics.get(key, 0.0):.4f}' for key in selected_keys]
        return ' '.join(parts)

    def _progress_postfix(self, *, is_train: bool, raw_loss: float, metrics: dict):
        postfix = {'loss': f'{raw_loss:.4f}'}
        if is_train:
            for key, value in metrics.items():
                if isinstance(value, str):
                    postfix[key] = value
                    continue
                postfix[key] = f'{value:.4f}'
            return postfix

        candidate_keys = []
        task_metric = self._metric_name()
        if task_metric in metrics:
            candidate_keys.append(task_metric)
        if self.config.task_type == 'sid' and 'sid_seq_acc' in metrics:
            candidate_keys.append('sid_seq_acc')
        main_metric = self._main_metric_name()
        if main_metric != 'loss' and main_metric in metrics:
            candidate_keys.append(main_metric)
        ranking_metric = self._default_ranking_metric()
        if ranking_metric in metrics:
            candidate_keys.append(ranking_metric)

        seen = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            value = metrics[key]
            if isinstance(value, str):
                postfix[key] = value
                continue
            postfix[key] = f'{value:.4f}'
        return postfix

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

    def _run_loader(self, loader, optimizer=None, desc='train', distributed_reduce=True):
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss = 0.0
        total_batches = 0
        metric_sums = {}
        accumulate_batch = max(1, self.config.accumulate_batch)

        iterator = tqdm(loader, desc=desc, leave=False, disable=not self.is_main_process)
        model_runner = self.model if is_train or not self.distributed else self.model_core
        if is_train:
            optimizer.zero_grad()
        for batch_index, batch in enumerate(iterator):
            is_last_micro_batch = batch_index == (len(loader) - 1)
            should_step = ((batch_index + 1) % accumulate_batch == 0) or is_last_micro_batch
            if is_train:
                sync_context = nullcontext()
                if self.distributed and not should_step and hasattr(self.model, 'no_sync'):
                    sync_context = self.model.no_sync()
                with sync_context:
                    rec_loss, metrics = model_runner(batch, mode='finetune')
                    loss = rec_loss / accumulate_batch
            else:
                with torch.no_grad():
                    rec_loss, metrics = model_runner(batch, mode='test')
            if is_train:
                loss.backward()
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad()

            raw_loss = float(rec_loss.item())
            total_loss += raw_loss
            total_batches += 1
            for key, value in metrics.items():
                if isinstance(value, str):
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            iterator.set_postfix(self._progress_postfix(is_train=is_train, raw_loss=raw_loss, metrics=metrics))

        if self.distributed and distributed_reduce:
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

    def _save_checkpoint(self, epoch: int, best_metric: float, valid_metrics: dict):
        if not self.is_main_process:
            return
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_core.trainable_state_dict(),
            'checkpoint_kind': 'trainable_only',
            'config': asdict(self.config),
            'valid_metrics': valid_metrics,
            'best_valid_metric': best_metric,
            'main_metric': self._main_metric_name(),
        }
        path = self.run_dir / 'best.pt'
        torch.save(checkpoint, path)
        self._pnt(f'saved best checkpoint to {path} with trainable-only state')

    def _save_meta(self, best_epoch: int, best_valid_metric: float, test_metrics: dict):
        if not self.is_main_process:
            return
        meta = self._load_meta_stub()
        meta.update({
            'config': asdict(self.config),
            'compiled_dir': str(self.compiled.compile_dir),
            'run_dir': str(self.run_dir),
            'best_epoch': best_epoch,
            'main_metric': self._main_metric_name(),
            'best_valid_metric': best_valid_metric,
            'train_metric_name': self._metric_name(),
            'test_metric_name': self._default_ranking_metric(),
            'declared_test_metrics': self.config.metrics,
            'test_metrics': test_metrics,
            'world_size': self.world_size,
            'status': 'finished',
            'finished_at': _utc_now_iso(),
        })
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        self._pnt(f'wrote trainer meta to {self.meta_path}')

    def _save_valid_only_meta(self, valid_metrics: dict):
        if not self.is_main_process:
            return
        meta = self._load_meta_stub()
        meta.update({
            'config': asdict(self.config),
            'compiled_dir': str(self.compiled.compile_dir),
            'run_dir': str(self.run_dir),
            'main_metric': self._main_metric_name(),
            'train_metric_name': self._metric_name(),
            'test_metric_name': self._default_ranking_metric(),
            'declared_test_metrics': self.config.metrics,
            'valid_metrics': valid_metrics,
            'world_size': self.world_size,
            'status': 'valid_only_finished',
            'finished_at': _utc_now_iso(),
        })
        meta.pop('test_metrics', None)
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        self._pnt(f'wrote valid-only meta to {self.meta_path}')

    def _load_meta_stub(self):
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _evaluate_eval_set(self, loader, desc: str):
        if self.distributed:
            dist.barrier()
        metrics = None
        if self.is_main_process:
            metrics = self._run_loader(loader, optimizer=None, desc=desc, distributed_reduce=False)
        if self.distributed:
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        if self.distributed:
            dist.barrier()
        return metrics

    def train(self):
        self.build_dataloaders()
        if self.config.valid_only:
            self._pnt(
                f'start valid-only evaluation on {self.config.data} with {self.config.model} '
                f'repr={self.config.repr_type} task={self.config.task_type} device={self.device} '
                f'world_size={self.world_size} rank={self.rank} local_rank={self.local_rank} '
                f'batch_size={self.config.batch_size} valid_eval=single-shot'
            )
            valid_metrics = self._evaluate_eval_set(self.valid_loader, desc='valid-only')
            if self.is_main_process:
                self._pnt(
                    f'valid_only valid_loss={valid_metrics["loss"]:.4f} '
                    f'{self._format_test_metrics(valid_metrics)}'
                )
                self._save_valid_only_meta(valid_metrics)
            if self.distributed:
                dist.destroy_process_group()
            return
        optimizer = self.build_optimizer()
        metric_name = self._metric_name()
        main_metric_name = self._main_metric_name()
        higher_is_better = self._main_metric_higher_is_better()
        best_metric = float('-inf') if higher_is_better else float('inf')
        best_epoch = 0
        wait = 0
        unlimited_epochs = self.config.epochs <= 0

        self._pnt(
                f'start training on {self.config.data} with {self.config.model} '
                f'repr={self.config.repr_type} task={self.config.task_type} device={self.device} '
                f'world_size={self.world_size} rank={self.rank} local_rank={self.local_rank} '
                f'batch_size={self.config.batch_size} accumulate_batch={self.config.accumulate_batch} '
                f'effective_batch_size={self.config.batch_size * self.config.accumulate_batch * self.world_size} '
                f'epochs={"until-early-stop" if unlimited_epochs else self.config.epochs} '
                f'valid_main_metric={main_metric_name} valid_eval=every-epoch test_eval=final-only'
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
            epoch_valid_metrics = self._evaluate_eval_set(self.valid_loader, desc=f'valid@{epoch}')
            if self.is_main_process:
                self._pnt(
                    f'epoch {epoch:03d} valid_loss={epoch_valid_metrics["loss"]:.4f} '
                    f'{self._format_test_metrics(epoch_valid_metrics)}'
                )
            if main_metric_name not in epoch_valid_metrics:
                raise KeyError(
                    f'evaluator.main_metric={main_metric_name} not found in valid metrics: '
                    f'{sorted(epoch_valid_metrics.keys())}'
                )
            current_metric = epoch_valid_metrics[main_metric_name]
            improved = current_metric > best_metric if higher_is_better else current_metric < best_metric
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                wait = 0
                self._save_checkpoint(epoch, best_metric, epoch_valid_metrics)
            else:
                wait += 1
                self._pnt(
                    f'no valid {main_metric_name} improvement for {wait} epoch(s), '
                    f'patience={self.config.patience}'
                )
                if wait >= self.config.patience:
                    self._pnt(
                        f'early stop at epoch {epoch:03d} with best_valid_{main_metric_name}={best_metric:.4f}'
                    )
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
        test_metrics = self._evaluate_eval_set(self.test_loader, desc='test')
        if self.is_main_process:
            self._pnt(
                f'best_epoch={best_epoch} test_loss={test_metrics["loss"]:.4f} '
                f'{self._format_test_metrics(test_metrics)}'
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


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _command_string():
    parts = [sys.executable, *sys.argv]
    return ' '.join(shlex.quote(part) for part in parts)


def _run_dir_for_config(config: TrainConfig):
    return ArtifactStore(config.data).trained_dir(config.run_id)


def _write_pid_record(run_dir: Path, config: TrainConfig):
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if rank != 0:
        return
    payload = {
        'pid': os.getpid(),
        'hostname': socket.gethostname(),
        'command': _command_string(),
        'run_dir': str(run_dir),
        'compile_dir': str(ArtifactStore(config.data).compiled_dir(config.compile_config.prepare_id)),
        'rank': rank,
        'world_size': world_size,
        'created_at': _utc_now_iso(),
    }
    (run_dir / 'pid.json').write_text(json.dumps(payload, indent=2) + '\n')


def _write_initial_meta(run_dir: Path, config: TrainConfig):
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if rank != 0:
        return
    meta = {
        'config': asdict(config),
        'run_dir': str(run_dir),
        'compiled_dir': str(ArtifactStore(config.data).compiled_dir(config.compile_config.prepare_id)),
        'command': _command_string(),
        'pid': os.getpid(),
        'hostname': socket.gethostname(),
        'world_size': world_size,
        'status': 'running',
        'started_at': _utc_now_iso(),
        'log_path': str(run_dir / 'train.log'),
    }
    (run_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def _setup_run_artifacts(config: TrainConfig):
    run_dir = _run_dir_for_config(config)
    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        attach_run_log(run_dir / 'train.log')
        _write_pid_record(run_dir, config)
        _write_initial_meta(run_dir, config)
    return run_dir


def _run_trainer(config: TrainConfig):
    _setup_run_artifacts(config)
    try:
        trainer = Trainer(config)
        trainer.train()
    except Exception as exc:
        rank = int(os.environ.get('RANK', '0'))
        if rank == 0:
            run_dir = _run_dir_for_config(config)
            meta_path = run_dir / 'meta.json'
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    meta = {}
            meta.update(
                {
                    'status': 'failed',
                    'error': repr(exc),
                    'failed_at': _utc_now_iso(),
                }
            )
            meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        raise


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
