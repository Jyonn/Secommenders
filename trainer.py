import os
import json
import math
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from core import CompiledArtifacts, SequentialRecModel, TrainConfig
from core.dataset import CompiledFinetuneTrajectoryDataset, CompiledTestSampleDataset, CompiledValidSampleDataset
from utils import function
from utils.artifact_identity import (
    migrate_train_config_dict,
    register_trained_artifact,
    resolve_compiled_dir,
    resolve_trained_run_dir,
    trained_artifact_identity,
)
from utils.config_init import ConfigInit
from utils.frequency_breakdown import FrequencyBreakdownAccumulator, count_finetune_target_frequencies
from utils.gpu import GPU
from utils.logging import attach_run_log, setup_logging
from utils.pipeline import ensure_clustered
from utils.representation_schema import semantic_graph_contract
from utils.server import Server


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = self.world_size > 1
        if self.distributed and self.config.frequency_breakdown:
            raise ValueError('frequency breakdown currently supports single-process evaluation only')
        self._init_distributed()
        self.run_dir = resolve_trained_run_dir(config)
        self._prepare_uid_hierarchy_if_needed()
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
        self.frequency_breakdown_path = self.run_dir / 'analysis' / 'frequency_breakdown_test.json'

    def _prepare_uid_hierarchy_if_needed(self):
        if self.config.task_type != 'uid' or self.config.uid_decoding != 'hierarchical':
            return
        uid_upstream = (getattr(self.config, 'upstreams', {}) or {}).get('uid') or {}
        clusterer_spec = uid_upstream.get('clusterer') or {}
        clusterer = ensure_clustered(self.config.data, self.config.uid_cluster_levels, clusterer_spec)
        setattr(self.config, 'uid_hierarchy_dir', str(clusterer.output_dir))

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
        if self.config.is_multi_task:
            self._pnt(
                f'multi decoding tasks={self.config.task_type} fusion={self.config.multi_fusion} '
                f'candidate_topk={self.config.multi_candidate_topk} output_topk={self.config.multi_output_topk} '
                f'normalization={self.config.multi_score_normalization} '
                f'frequency_threshold={self.config.multi_frequency_threshold:g} '
                f'frequency_smoothing={self.config.multi_frequency_smoothing:g}'
            )
        elif self.config.task_type == 'uid' and self.config.uid_decoding == 'hierarchical':
            self._pnt(
                f'uid decoding=hierarchical levels={self.config.uid_cluster_levels} '
                f'topk={self.config.uid_cluster_topk}'
            )
        elif self.config.task_type == 'sid':
            sid_decoding = self.model_core._sid_decoding_mode()
            kv_cache_supported = self.model_core._sid_kv_cache_supported()
            kv_sample_chunk = max(1, self.config.code_beam_chunk_size // max(1, self.config.code_beam_width))
            self._pnt(
                f'sid decoding={sid_decoding} '
                + (
                    f'beam_width={self.config.code_beam_width} '
                    f'beam_chunk_size={self.config.code_beam_chunk_size} '
                    f'kv_cache={kv_cache_supported} '
                    f'kv_beam_batching={"micro-batch" if kv_cache_supported else "disabled"} '
                    f'kv_sample_chunk={kv_sample_chunk} '
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

    def build_test_loader_only(self):
        test_dataset = CompiledTestSampleDataset(self.compiled.test)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        self.train_loader = None
        self.valid_loader = None
        self.train_sampler = None
        self._pnt(
            f'built test-only dataloader test={len(test_dataset)} batch_size={self.config.batch_size} '
            f'world_size={self.world_size} order=length-sorted'
        )
        if self.config.is_multi_task:
            self._pnt(
                f'multi decoding tasks={self.config.task_type} fusion={self.config.multi_fusion} '
                f'candidate_topk={self.config.multi_candidate_topk} output_topk={self.config.multi_output_topk} '
                f'normalization={self.config.multi_score_normalization}'
            )
        elif self.config.task_type == 'uid' and self.config.uid_decoding == 'hierarchical':
            self._pnt(
                f'uid decoding=hierarchical levels={self.config.uid_cluster_levels} '
                f'topk={self.config.uid_cluster_topk}'
            )
        elif self.config.task_type == 'sid':
            sid_decoding = self.model_core._sid_decoding_mode()
            kv_cache_supported = self.model_core._sid_kv_cache_supported()
            kv_sample_chunk = max(1, self.config.code_beam_chunk_size // max(1, self.config.code_beam_width))
            self._pnt(
                f'sid decoding={sid_decoding} '
                + (
                    f'beam_width={self.config.code_beam_width} '
                    f'beam_chunk_size={self.config.code_beam_chunk_size} '
                    f'kv_cache={kv_cache_supported} '
                    f'kv_beam_batching={"micro-batch" if kv_cache_supported else "disabled"} '
                    f'kv_sample_chunk={kv_sample_chunk} '
                    f'ks={self.model_core.sid_ranking_ks()}'
                    if sid_decoding == 'sequential'
                    else f'item_scoring ks={self.model_core.sid_ranking_ks()}'
                )
            )
        elif self.config.task_type == 'hash':
            self._pnt(f'hash decoding=parallel item_scoring ks={self.model_core.ranking_ks()}')

    def _metric_name(self):
        if self.config.is_multi_task:
            return 'uid_acc'
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
        return '|'.join(self._main_metric_names())

    def _main_metric_names(self):
        names = []
        for value in str(self.config.main_metric or 'loss').split('|'):
            metric = value.strip().lower()
            if metric and metric not in names:
                names.append(metric)
        return names or ['loss']

    @staticmethod
    def _metric_higher_is_better(metric: str):
        return metric != 'loss' and not metric.endswith('_loss')

    def _main_metric_higher_is_better(self, metric: str | None = None):
        return self._metric_higher_is_better(metric or self._main_metric_names()[0])

    def _update_main_metric_bests(self, valid_metrics: dict, best_metrics: dict[str, float]):
        improved_metrics = []
        for metric in self._main_metric_names():
            current_metric = valid_metrics[metric]
            improved = (
                current_metric > best_metrics[metric]
                if self._main_metric_higher_is_better(metric)
                else current_metric < best_metrics[metric]
            )
            if improved:
                best_metrics[metric] = current_metric
                improved_metrics.append(metric)
        return improved_metrics

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
        for main_metric in self._main_metric_names():
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

    def build_lr_scheduler(self, optimizer):
        accumulate_batch = max(1, int(self.config.accumulate_batch))
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accumulate_batch))
        if self.config.epochs <= 0:
            self._pnt('lr scheduler=constant warmup_ratio=0 optimizer_steps=unbounded')
            return None

        total_steps = steps_per_epoch * int(self.config.epochs)
        warmup_steps = int(total_steps * float(self.config.warmup_ratio))
        if self.config.warmup_ratio > 0:
            warmup_steps = max(1, warmup_steps)
        warmup_steps = min(warmup_steps, total_steps)
        scheduler_name = self.config.lr_scheduler

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            if scheduler_name == 'constant':
                return 1.0
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(max((current_step - warmup_steps) / decay_steps, 0.0), 1.0)
            if scheduler_name == 'linear':
                return max(0.0, 1.0 - progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self._pnt(
            f'lr scheduler={scheduler_name} warmup_ratio={self.config.warmup_ratio:g} '
            f'warmup_steps={warmup_steps} total_steps={total_steps} '
            f'optimizer_steps_per_epoch={steps_per_epoch}'
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _run_loader(
        self,
        loader,
        optimizer=None,
        lr_scheduler=None,
        desc='train',
        distributed_reduce=True,
        max_batches=None,
    ):
        is_train = optimizer is not None
        self.model.train(is_train)
        frequency_accumulator = None
        collect_frequency_breakdown = (
            not is_train
            and desc in {'test', 'test-only'}
            and self.config.frequency_breakdown
        )
        if collect_frequency_breakdown:
            target_frequencies = count_finetune_target_frequencies(self.compiled.finetune)
            frequency_accumulator = FrequencyBreakdownAccumulator(
                target_frequencies,
                self.config.frequency_buckets,
                self.model_core.ranking_ks(),
            )
            self.model_core.enable_ranking_trace(True)
            self._pnt(
                f'{desc} frequency breakdown enabled buckets={self.config.frequency_buckets} '
                f'finetune_target_items={len(target_frequencies):,}'
            )
        profile_sid_decoding = (
            desc == 'valid-only'
            and self.config.task_type == 'sid'
            and self.model_core._sid_decoding_mode() == 'sequential'
        )
        self.model_core.sid_decoding_timing_enabled = profile_sid_decoding
        total_loss = 0.0
        total_samples = 0
        metric_sums = {}
        accumulate_batch = max(1, self.config.accumulate_batch)

        iterator = tqdm(loader, desc=desc, leave=False, disable=not self.is_main_process)
        model_runner = self.model if is_train or not self.distributed else self.model_core
        if is_train:
            optimizer.zero_grad()
        for batch_index, batch in enumerate(iterator):
            if max_batches is not None and batch_index >= max_batches:
                break
            effective_loader_len = len(loader) if max_batches is None else min(len(loader), max_batches)
            is_last_micro_batch = batch_index == (effective_loader_len - 1)
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
                if frequency_accumulator is not None:
                    ranking_records = self.model_core.pop_ranking_trace_records()
                    if len(ranking_records) != len(batch):
                        raise RuntimeError(
                            f'frequency breakdown expected one target rank per sample, '
                            f'got {len(ranking_records)} records for batch size {len(batch)}'
                        )
                    for record in ranking_records:
                        target_uid = int(record['target_uid'])
                        frequency_accumulator.add(
                            target_uid,
                            record['rank'],
                            raw_item_id=self.compiled.uid_raw_items[target_uid],
                        )
            if is_train:
                loss.backward()
                if should_step:
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()

            raw_loss = float(rec_loss.item())
            batch_size = len(batch)
            total_loss += raw_loss * batch_size
            total_samples += batch_size
            for key, value in metrics.items():
                if isinstance(value, str):
                    continue
                if key.startswith('sid_time_'):
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size

            if profile_sid_decoding:
                timing_parts = [
                    f'{key.removeprefix("sid_time_").removesuffix("_ms")}={float(value):.1f}ms'
                    for key, value in metrics.items()
                    if key.startswith('sid_time_') and key.endswith('_ms')
                ]
                self._pnt(f'{desc} batch={batch_index + 1} SID decoding timing: ' + ' '.join(timing_parts))
                self._pnt(
                    f'{desc} batch={batch_index + 1} SID KV-cache diagnostic: '
                    f'{metrics.get("sid_kv_diagnostic", "missing")}'
                )

            iterator.set_postfix(self._progress_postfix(is_train=is_train, raw_loss=raw_loss, metrics=metrics))

        if self.distributed and distributed_reduce:
            totals = torch.tensor([total_loss, float(total_samples)], dtype=torch.float64, device=self.device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss = float(totals[0].item())
            total_samples = int(totals[1].item())
            reduced_metric_sums = {}
            for key, value in metric_sums.items():
                tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_metric_sums[key] = float(tensor.item())
            metric_sums = reduced_metric_sums

        self.model_core.sid_decoding_timing_enabled = False
        self.model_core.enable_ranking_trace(False)
        summary = {'loss': total_loss / max(total_samples, 1)}
        if is_train:
            summary['learning_rate'] = float(optimizer.param_groups[0]['lr'])
        for key, value in metric_sums.items():
            summary[key] = value / max(total_samples, 1)
        if frequency_accumulator is not None and self.is_main_process:
            breakdown = frequency_accumulator.summary()
            breakdown.update({
                'data': self.config.data,
                'model': self.config.model,
                'repr_type': self.config.repr_type,
                'task_type': self.config.task_type,
                'split': 'test',
                'compiled_dir': str(self.compiled.compile_dir),
                'run_dir': str(self.run_dir),
                'generated_at': _utc_now_iso(),
            })
            self.frequency_breakdown_path.parent.mkdir(parents=True, exist_ok=True)
            self.frequency_breakdown_path.write_text(json.dumps(breakdown, indent=2) + '\n')
            for label, bucket in breakdown['buckets'].items():
                metric_text = ' '.join(
                    f'{key}={value:.4f}'
                    for key, value in bucket.items()
                    if key in {'mrr', *[f'hr@{k}' for k in self.model_core.ranking_ks()],
                               *[f'ndcg@{k}' for k in self.model_core.ranking_ks()]}
                    and value is not None
                )
                self._pnt(
                    f'{desc} frequency={label} targets={bucket["target_count"]:,} '
                    f'share={bucket["target_share"] or 0:.2%} {metric_text}'
                )
            self._pnt(f'wrote test frequency breakdown to {self.frequency_breakdown_path}')
        return summary

    def _save_checkpoint(self, epoch: int, best_metrics: dict[str, float], valid_metrics: dict):
        if not self.is_main_process:
            return
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_core.trainable_state_dict(),
            'checkpoint_kind': 'trainable_only',
            'config': asdict(self.config),
            'valid_metrics': valid_metrics,
            'best_valid_metric': best_metrics[self._main_metric_names()[0]],
            'best_valid_metrics': best_metrics,
            'main_metric': self._main_metric_name(),
        }
        path = self.run_dir / 'best.pt'
        torch.save(checkpoint, path)
        self._pnt(f'saved best checkpoint to {path} with trainable-only state')

    def _save_meta(self, best_epoch: int, best_valid_metrics: dict[str, float], test_metrics: dict):
        if not self.is_main_process:
            return
        meta = self._load_meta_stub()
        meta.update({
            'config': asdict(self.config),
            'compiled_dir': str(self.compiled.compile_dir),
            'run_dir': str(self.run_dir),
            'best_epoch': best_epoch,
            'main_metric': self._main_metric_name(),
            'best_valid_metric': best_valid_metrics[self._main_metric_names()[0]],
            'best_valid_metrics': best_valid_metrics,
            'train_metric_name': self._metric_name(),
            'test_metric_name': self._default_ranking_metric(),
            'declared_test_metrics': self.config.metrics,
            'test_metrics': test_metrics,
            'sample_counts': self._compiled_sample_counts(),
            'world_size': self.world_size,
            'status': 'finished',
            'finished_at': _utc_now_iso(),
            'artifact_identity': trained_artifact_identity(self.config, self.run_dir),
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
            'sample_counts': self._compiled_sample_counts(),
            'world_size': self.world_size,
            'status': 'valid_only_finished',
            'finished_at': _utc_now_iso(),
            'artifact_identity': trained_artifact_identity(self.config, self.run_dir),
        })
        meta.pop('test_metrics', None)
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        self._pnt(f'wrote valid-only meta to {self.meta_path}')

    def _save_test_only_meta(self, checkpoint: dict | None, test_metrics: dict):
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
            'loaded_checkpoint': str(self.config.load_ckpt) if self.config.load_ckpt else None,
            'checkpoint_epoch': checkpoint.get('epoch') if checkpoint else None,
            'checkpoint_best_valid_metric': checkpoint.get('best_valid_metric') if checkpoint else None,
            'checkpoint_best_valid_metrics': checkpoint.get('best_valid_metrics') if checkpoint else None,
            'checkpoint_main_metric': checkpoint.get('main_metric') if checkpoint else None,
            'checkpoint_valid_metrics': checkpoint.get('valid_metrics') if checkpoint else None,
            'test_metrics': test_metrics,
            'sample_counts': self._compiled_sample_counts(),
            'world_size': self.world_size,
            'status': 'test_only_finished',
            'finished_at': _utc_now_iso(),
            'artifact_identity': trained_artifact_identity(self.config, self.run_dir),
        })
        if self.config.frequency_breakdown:
            analysis = dict(meta.get('analysis') or {})
            analysis['frequency_breakdown_test'] = str(self.frequency_breakdown_path)
            meta['analysis'] = analysis
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        self._pnt(f'wrote test-only meta to {self.meta_path}')

    def _compiled_sample_counts(self):
        return {
            'finetune': int(len(self.compiled.finetune)),
            'valid': int(len(self.compiled.valid)),
            'test': int(len(self.compiled.test)),
        }

    def _load_meta_stub(self):
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _evaluate_eval_set(self, loader, desc: str, max_batches=None):
        if self.distributed:
            dist.barrier()
        metrics = None
        if self.is_main_process:
            metrics = self._run_loader(
                loader,
                optimizer=None,
                desc=desc,
                distributed_reduce=False,
                max_batches=max_batches,
            )
        if self.distributed:
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        if self.distributed:
            dist.barrier()
        return metrics

    def _assert_checkpoint_compatible(self, checkpoint: dict):
        saved_config = checkpoint.get('config') or {}
        # Older checkpoints may predate recently added config fields. Normalize
        # them to their historical defaults so we only reject true structural
        # mismatches instead of schema evolution.
        normalized_saved_config = dict(saved_config)
        normalized_saved_config.setdefault(
            'representation_pair_bias_mode',
            'shared' if normalized_saved_config.get('representation_pair_bias') else 'none',
        )
        normalized_saved_config.setdefault('representation_pair_bias_residual_scale', None)
        normalized_saved_config.setdefault('uid_decoding', 'flat')
        normalized_saved_config.setdefault('uid_cluster_levels', None)
        normalized_saved_config.setdefault('uid_cluster_topk', None)
        required_keys = [
            'data', 'model', 'repr_type', 'repr_source_model', 'sid_export', 'sid_coder', 'hash_coder',
            'repr_combine', 'task_type', 'maxitems', 'model_max_length', 'item_text_max_tokens',
            'freeze_backbone', 'uid_decoding', 'uid_cluster_levels', 'uid_cluster_topk',
            'code_decoding', 'model_dtype', 'use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout',
            'lora_layers', 'hidden_size', 'num_layers', 'num_heads', 'dropout',
            'representation_pair_bias', 'representation_pair_bias_mode',
        ]
        if self.config.representation_pair_bias_mode == 'head':
            required_keys.append('representation_pair_bias_residual_scale')
        if self.config.representation_graph:
            for key in (
                'repr_source_model', 'sid_export', 'sid_coder', 'hash_coder',
                'uid_decoding', 'uid_cluster_levels', 'uid_cluster_topk', 'code_decoding',
            ):
                required_keys.remove(key)
        current_config = asdict(self.config)
        mismatches = []
        for key in required_keys:
            if normalized_saved_config.get(key) != current_config.get(key):
                mismatches.append((key, normalized_saved_config.get(key), current_config.get(key)))
        if self.config.representation_graph:
            migrated_saved = migrate_train_config_dict(normalized_saved_config)
            saved_contract = semantic_graph_contract(migrated_saved['representation_graph'])
            current_contract = semantic_graph_contract(self.config.representation_graph)
            if saved_contract != current_contract:
                mismatches.append(('representation_graph', saved_contract, current_contract))
        if mismatches:
            preview = ', '.join(
                f'{key}: ckpt={saved!r} current={current!r}'
                for key, saved, current in mismatches[:5]
            )
            raise ValueError(f'Checkpoint config is incompatible with current config ({preview})')

    @staticmethod
    def _graph_marker_names(graph):
        names = list(dict.fromkeys(graph['encoder']['representations']))
        targets = [target['representation'] for target in graph['decoder']['targets']]
        if len(targets) > 1:
            names.append('decoder_' + '_'.join(targets))
        return names

    def _adapt_checkpoint_state_dict(self, state_dict: dict, saved_config: dict | None = None):
        if not self.config.representation_graph:
            return state_dict
        adapted = dict(state_dict)
        current = self.model_core.state_dict()
        saved_config = dict(saved_config or {})
        saved_original_graph = saved_config.get('representation_graph')
        migrated_saved = migrate_train_config_dict(saved_config)
        saved_graph = migrated_saved['representation_graph']
        current_graph = self.config.representation_graph
        saved_names = saved_graph['encoder']['representations']
        current_names = current_graph['encoder']['representations']
        positional_name_map = dict(zip(current_names, saved_names))
        marker_key = 'type_marker_embedding.weight'
        if marker_key in adapted and marker_key in current:
            from models.base import TYPE_MARKER_ORDER

            old = adapted[marker_key]
            rebuilt = current[marker_key].clone()
            current_marker_map = self.compiled.special_vocab['marker_to_index']
            if saved_original_graph:
                saved_marker_map = {
                    name: index for index, name in enumerate(self._graph_marker_names(saved_graph))
                }
                for current_name, current_index in current_marker_map.items():
                    saved_name = positional_name_map.get(current_name)
                    if current_name.startswith('decoder_'):
                        saved_targets = [target['representation'] for target in saved_graph['decoder']['targets']]
                        saved_name = 'decoder_' + '_'.join(saved_targets)
                    saved_index = saved_marker_map.get(saved_name)
                    if saved_index is not None and saved_index < old.shape[0]:
                        rebuilt[current_index] = old[saved_index]
            else:
                for name, current_index in current_marker_map.items():
                    if name.startswith('decoder_'):
                        legacy_name = '+'.join(
                            self.config.compile_config.representation_kind(target)
                            for target in self.config.compile_config.target_names
                        )
                    else:
                        legacy_name = self.config.compile_config.representation_kind(name)
                    if legacy_name in TYPE_MARKER_ORDER:
                        old_index = TYPE_MARKER_ORDER.index(legacy_name)
                        if old_index < old.shape[0]:
                            rebuilt[current_index] = old[old_index]
            adapted[marker_key] = rebuilt

        embedding_names = self.config.compile_config.names_for_kind('embedding')
        if not saved_original_graph and len(embedding_names) == 1:
            name = embedding_names[0]
            aliases = {
                'embedding_projection.weight': f'embedding_projections.{name}.weight',
                'embedding_head.weight': f'embedding_heads.{name}.weight',
            }
            for old_key, new_key in aliases.items():
                if old_key in adapted and new_key not in adapted:
                    adapted[new_key] = adapted.pop(old_key)
        elif saved_original_graph:
            for current_name in embedding_names:
                saved_name = positional_name_map.get(current_name)
                if not saved_name:
                    continue
                for module_name in ('embedding_projections', 'embedding_heads'):
                    old_key = f'{module_name}.{saved_name}.weight'
                    new_key = f'{module_name}.{current_name}.weight'
                    if old_key in adapted and old_key != new_key and new_key not in adapted:
                        adapted[new_key] = adapted.pop(old_key)

        sid_names = self.config.compile_config.names_for_kind('sid')
        if len(sid_names) == 1:
            name = sid_names[0]
            aliases = {
                'sid_embedding.weight': f'sid_embeddings.{name}.weight',
                'sid_head.weight': f'sid_heads.{name}.weight',
                'sid_head.bias': f'sid_heads.{name}.bias',
            }
            for old_key, new_key in aliases.items():
                if old_key in adapted and new_key not in adapted:
                    adapted[new_key] = adapted.pop(old_key)
        if saved_original_graph:
            for current_name in sid_names:
                saved_name = positional_name_map.get(current_name)
                if not saved_name:
                    continue
                for module_name in ('sid_embeddings', 'sid_heads'):
                    for suffix in ('weight', 'bias'):
                        old_key = f'{module_name}.{saved_name}.{suffix}'
                        new_key = f'{module_name}.{current_name}.{suffix}'
                        if old_key in adapted and old_key != new_key and new_key not in adapted:
                            adapted[new_key] = adapted.pop(old_key)

        ignored = []
        for key in list(adapted):
            if key in current and adapted[key].shape != current[key].shape:
                ignored.append(key)
                adapted.pop(key)
        if ignored:
            self._pnt(f'checkpoint migration ignored {len(ignored)} shape-incompatible keys: {ignored[:5]}')
        return adapted

    def _load_checkpoint_for_eval(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._assert_checkpoint_compatible(checkpoint)
        if self.is_main_process:
            state_dict = self._adapt_checkpoint_state_dict(
                checkpoint['model_state_dict'],
                checkpoint.get('config'),
            )
            load_info = self.model_core.load_state_dict(state_dict, strict=False)
            missing = getattr(load_info, 'missing_keys', [])
            unexpected = getattr(load_info, 'unexpected_keys', [])
            if unexpected:
                preview = ', '.join(unexpected[:5])
                suffix = '' if len(unexpected) <= 5 else f' ... (+{len(unexpected) - 5} more)'
                self._pnt(
                    'ignoring unexpected checkpoint keys from older/newer model variants: '
                    f'{preview}{suffix}'
                )
            self._pnt(
                f'loaded checkpoint {checkpoint_path} '
                f'with {len(missing)} missing frozen/base keys and {len(unexpected)} ignored unexpected keys'
            )
        return checkpoint

    def train(self):
        if self.config.test_only:
            self.build_test_loader_only()
            self._pnt(
                f'start test-only evaluation on {self.config.data} with {self.config.model} '
                f'repr={self.config.repr_type} task={self.config.task_type} device={self.device} '
                f'world_size={self.world_size} rank={self.rank} local_rank={self.local_rank} '
                f'batch_size={self.config.batch_size} checkpoint={self.config.load_ckpt}'
            )
            checkpoint = None
            if self.config.load_ckpt:
                checkpoint = self._load_checkpoint_for_eval(self.config.load_ckpt)
            elif self.is_main_process:
                self._pnt('warning: test_only=true without load_ckpt; evaluating current model weights without loading a checkpoint')
            test_metrics = self._evaluate_eval_set(self.test_loader, desc='test-only')
            if self.is_main_process:
                self._pnt(
                    f'test_only test_loss={test_metrics["loss"]:.4f} '
                    f'{self._format_test_metrics(test_metrics)}'
                )
                self._save_test_only_meta(checkpoint, test_metrics)
            if self.distributed:
                dist.destroy_process_group()
            return
        self.build_dataloaders()
        if self.config.valid_only:
            valid_only_batches = None if self.config.valid_only == -1 else int(self.config.valid_only)
            self._pnt(
                f'start valid-only evaluation on {self.config.data} with {self.config.model} '
                f'repr={self.config.repr_type} task={self.config.task_type} device={self.device} '
                f'world_size={self.world_size} rank={self.rank} local_rank={self.local_rank} '
                f'batch_size={self.config.batch_size} '
                f'valid_eval={"full" if valid_only_batches is None else f"{valid_only_batches}-batch-smoke"}'
            )
            valid_metrics = self._evaluate_eval_set(
                self.valid_loader,
                desc='valid-only',
                max_batches=valid_only_batches,
            )
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
        lr_scheduler = self.build_lr_scheduler(optimizer)
        metric_name = self._metric_name()
        main_metric_names = self._main_metric_names()
        main_metric_name = '|'.join(main_metric_names)
        best_metrics = {
            metric: float('-inf') if self._main_metric_higher_is_better(metric) else float('inf')
            for metric in main_metric_names
        }
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
            train_metrics = self._run_loader(
                self.train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                desc=f'train@{epoch}',
            )
            train_metric_name = metric_name if metric_name in train_metrics else (
                'sid_token_acc' if 'sid_token_acc' in train_metrics else metric_name
            )
            self._pnt(
                f'epoch {epoch:03d} train_loss={train_metrics["loss"]:.4f} '
                f'{train_metric_name}={train_metrics.get(train_metric_name, 0.0):.4f} '
                f'lr={train_metrics["learning_rate"]:.6g}'
            )
            epoch_valid_metrics = self._evaluate_eval_set(self.valid_loader, desc=f'valid@{epoch}')
            if self.is_main_process:
                self._pnt(
                    f'epoch {epoch:03d} valid_loss={epoch_valid_metrics["loss"]:.4f} '
                    f'{self._format_test_metrics(epoch_valid_metrics)}'
                )
            missing_main_metrics = [metric for metric in main_metric_names if metric not in epoch_valid_metrics]
            if missing_main_metrics:
                raise KeyError(
                    f'evaluator.main_metric entries={missing_main_metrics} not found in valid metrics: '
                    f'{sorted(epoch_valid_metrics.keys())}'
                )
            improved_metrics = self._update_main_metric_bests(epoch_valid_metrics, best_metrics)
            if improved_metrics:
                best_epoch = epoch
                wait = 0
                self._pnt(f'valid improvement in {"|".join(improved_metrics)}; reset patience')
                self._save_checkpoint(epoch, best_metrics, epoch_valid_metrics)
            else:
                wait += 1
                self._pnt(
                    f'no valid {main_metric_name} improvement for {wait} epoch(s), '
                    f'patience={self.config.patience}'
                )
                if wait >= self.config.patience:
                    self._pnt(
                        f'early stop at epoch {epoch:03d} with best_valid_metrics={best_metrics}'
                    )
                    break

            if not unlimited_epochs and epoch >= self.config.epochs:
                break

        if self.is_main_process:
            self._load_checkpoint_for_eval(self.run_dir / 'best.pt')
        test_metrics = self._evaluate_eval_set(self.test_loader, desc='test')
        if self.is_main_process:
            self._pnt(
                f'best_epoch={best_epoch} test_loss={test_metrics["loss"]:.4f} '
                f'{self._format_test_metrics(test_metrics)}'
            )
            self._save_meta(best_epoch, best_metrics, test_metrics)

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
    return resolve_trained_run_dir(config)


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
        'compile_dir': str(resolve_compiled_dir(config.compile_config)),
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
    meta = _read_json_if_exists(run_dir / 'meta.json') or {}
    meta.update({
        'config': asdict(config),
        'run_dir': str(run_dir),
        'compiled_dir': str(resolve_compiled_dir(config.compile_config)),
        'command': _command_string(),
        'pid': os.getpid(),
        'hostname': socket.gethostname(),
        'world_size': world_size,
        'status': 'running',
        'started_at': _utc_now_iso(),
        'log_path': str(run_dir / 'train.log'),
        'artifact_identity': trained_artifact_identity(config, run_dir),
    })
    (run_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    register_trained_artifact(config, run_dir)


def _setup_run_artifacts(config: TrainConfig):
    run_dir = _run_dir_for_config(config)
    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        attach_run_log(run_dir / 'train.log')
        _write_pid_record(run_dir, config)
        _write_initial_meta(run_dir, config)
    return run_dir


def _report_server():
    return Server.from_env()


def _report_session():
    return os.environ.get('SECOMMENDER_REPORT_SESSION')


def _report_phase():
    return os.environ.get('SECOMMENDER_REPORT_PHASE', '')


def _read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _completed_run_status(config: TrainConfig, run_dir: Path):
    meta = _read_json_if_exists(run_dir / 'meta.json') or {}
    status = str(meta.get('status') or '').lower()
    if config.test_only:
        complete_statuses = {'test_only_finished'}
        required_paths = [run_dir / 'meta.json']
        if config.frequency_breakdown:
            required_paths.append(run_dir / 'analysis' / 'frequency_breakdown_test.json')
    elif config.valid_only:
        complete_statuses = {'valid_only_finished'}
        required_paths = [run_dir / 'meta.json']
    else:
        complete_statuses = {'finished'}
        required_paths = [run_dir / 'meta.json', run_dir / 'best.pt']
    complete = status in complete_statuses and all(path.exists() for path in required_paths)
    return complete, status, meta


def _pid_is_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _active_running_record(run_dir: Path, meta: dict):
    if str(meta.get('status') or '').lower() != 'running':
        return None
    pid_record = _read_json_if_exists(run_dir / 'pid.json') or {}
    pid = pid_record.get('pid') or meta.get('pid')
    hostname = pid_record.get('hostname') or meta.get('hostname')
    current_hostname = socket.gethostname()
    if hostname and str(hostname) != current_hostname:
        return None
    if not _pid_is_alive(pid):
        return None
    return {
        'pid': int(pid),
        'hostname': hostname or current_hostname,
        'command': pid_record.get('command') or meta.get('command') or '-',
        'started_at': meta.get('started_at') or pid_record.get('created_at') or '-',
    }


def _print_running_run_summary(run_dir: Path, record: dict, overwrite: str):
    pnt(
        f'trainer run is already running at {run_dir}; '
        f'pid={record["pid"]} hostname={record["hostname"]} overwrite={overwrite}, skipping. '
        'Use --overwrite true to rerun anyway.'
    )
    pnt(f'running started_at={record.get("started_at", "-")} command={record.get("command", "-")}')


def _format_metric_summary(metrics: dict | None):
    if not isinstance(metrics, dict) or not metrics:
        return '-'
    parts = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f'{key}={value:.6g}')
        else:
            parts.append(f'{key}={value}')
    return ' '.join(parts)


def _print_completed_run_summary(run_dir: Path, status: str, meta: dict):
    pnt(
        f'trainer run already complete at {run_dir}; '
        f'status={status} overwrite=auto, skipping. '
        'Use --overwrite true to rerun.'
    )
    if meta.get('best_epoch') is not None:
        pnt(
            f'completed result best_epoch={meta.get("best_epoch")} '
            f'main_metric={meta.get("main_metric", "-")} '
            f'best_valid={meta.get("best_valid_metric", "-")}'
        )
    if isinstance(meta.get('valid_metrics'), dict):
        pnt(f'completed valid_metrics {_format_metric_summary(meta.get("valid_metrics"))}')
    if isinstance(meta.get('test_metrics'), dict):
        pnt(f'completed test_metrics {_format_metric_summary(meta.get("test_metrics"))}')
    if isinstance(meta.get('checkpoint_valid_metrics'), dict):
        pnt(f'completed checkpoint_valid_metrics {_format_metric_summary(meta.get("checkpoint_valid_metrics"))}')


def _should_skip_completed_run(config: TrainConfig, run_dir: Path):
    overwrite = str(getattr(config, 'overwrite', 'auto') or 'auto').strip().lower()
    complete, status, meta = _completed_run_status(config, run_dir)
    if overwrite == 'true':
        return False
    active_record = _active_running_record(run_dir, meta)
    if active_record is not None:
        _print_running_run_summary(run_dir, active_record, overwrite)
        return True
    if overwrite == 'false' and run_dir.exists():
        raise RuntimeError(
            f'trainer run exists at {run_dir} with status={status or "unknown"}; '
            'use --overwrite true to rerun'
        )
    if overwrite == 'auto' and complete:
        _print_completed_run_summary(run_dir, status, meta)
        return True
    if overwrite == 'auto' and run_dir.exists():
        pnt(
            f'trainer run exists but is not complete at {run_dir}; '
            f'status={status or "unknown"} overwrite=auto, rerunning.'
        )
    return False


def _looks_like_tqdm_progress(line: str):
    text = str(line).strip()
    if not text:
        return False
    if text.count('|') < 2:
        return False
    return any(token in text for token in ('%|', 'it/s', 's/it'))


def _collapse_tqdm_progress(text: str):
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    collapsed = []
    pending_progress = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if _looks_like_tqdm_progress(line):
            pending_progress = line
            continue
        if pending_progress is not None:
            collapsed.append(pending_progress)
            pending_progress = None
        collapsed.append(line)
    if pending_progress is not None:
        collapsed.append(pending_progress)
    return '\n'.join(collapsed)


def _read_log_for_report(path: Path, max_bytes: int = 2_000_000):
    if not path.exists():
        return ''
    data = path.read_bytes()
    text = _collapse_tqdm_progress(data.decode('utf-8', errors='ignore'))
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode('utf-8', errors='ignore')


def _report_performance_from_meta(meta: dict | None):
    if not isinstance(meta, dict):
        return None
    for key in ('test_metrics', 'valid_metrics', 'checkpoint_valid_metrics'):
        metrics = meta.get(key)
        if isinstance(metrics, dict):
            return metrics
    return None


def _register_remote_experiment(run_dir: Path, config: TrainConfig):
    rank = int(os.environ.get('RANK', '0'))
    session = _report_session()
    server = _report_server()
    if rank != 0 or not session or server is None:
        return
    try:
        reply = server.register_experiment(
            session,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            run_dir=str(run_dir),
            log_path=str(run_dir / 'train.log'),
            command=_command_string(),
            phase=_report_phase(),
        )
    except Exception as exc:
        pnt(f'warning: failed to register remote experiment session={session}: {repr(exc)}')
        return
    if not reply.ok:
        pnt(f'warning: failed to register remote experiment session={session}: {reply.msg or reply.identifier}')


def _update_remote_experiment(run_dir: Path, *, status: str, error_text: str = ''):
    rank = int(os.environ.get('RANK', '0'))
    session = _report_session()
    server = _report_server()
    if rank != 0 or not session or server is None:
        return

    meta = _read_json_if_exists(run_dir / 'meta.json') or {}
    log_text = _read_log_for_report(run_dir / 'train.log')
    performance = _report_performance_from_meta(meta)
    if len(log_text.encode('utf-8')) >= 2_000_000:
        meta = dict(meta)
        meta['report_log_truncated'] = True
    try:
        reply = server.update_experiment(
            session,
            status=status,
            phase=_report_phase(),
            meta=meta,
            performance=performance,
            log=log_text,
            error=error_text,
        )
    except Exception as exc:
        pnt(f'warning: failed to update remote experiment session={session}: {repr(exc)}')
        return
    if not reply.ok:
        pnt(f'warning: failed to update remote experiment session={session}: {reply.msg or reply.identifier}')


def _run_trainer(config: TrainConfig):
    run_dir = _run_dir_for_config(config)
    if _should_skip_completed_run(config, run_dir):
        return
    run_dir = _setup_run_artifacts(config)
    _register_remote_experiment(run_dir, config)
    try:
        trainer = Trainer(config)
        trainer.train()
        _update_remote_experiment(run_dir, status='completed')
    except Exception as exc:
        rank = int(os.environ.get('RANK', '0'))
        if rank == 0:
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
        _update_remote_experiment(run_dir, status='failed', error_text=repr(exc))
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
