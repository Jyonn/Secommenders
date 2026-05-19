import json
from pathlib import Path

import numpy as np
import pandas as pd
import pigmento
import torch
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoders.data.base import TensorSpec, create_dataloaders, split_dataset
from autoencoders.data.embeddings import EmbeddingMatrix, EmbeddingTensorDataset
from autoencoders.function import resolve_device, set_seed
from autoencoders.models.loading import load_model
from autoencoders.training.display import (
    DISPLAY_METRIC_KEY_BY_SHORT_NAME,
    SCALAR_METRIC_SPECS,
    TrainerDisplay,
    style,
)
from autoencoders.training.trainer import TrainingConfig, VQTrainer
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor

DEFAULT_BEST_DIRECTIONS = {
    'loss': 'min',
    'recon': 'min',
    'coll': 'min',
    'book': 'min',
    'commit': 'min',
    'dead': 'min',
    'codes': 'max',
    'usage': 'max',
    'ppl': 'max',
}


def _format_spec(spec):
    return style(str(spec), fg='green')


def _print_pipeline_trace(model):
    if not hasattr(model, 'get_pipeline_trace'):
        return

    print(style(' Shape Trace ', fg='white', bg='magenta', bold=True))
    pipeline = model.get_pipeline_trace()
    if not pipeline:
        print(style('  <empty>', fg='yellow', dim=True))
        print(style(' End Trace ', fg='black', bg='yellow', bold=True))
        print()
        return

    first_step = pipeline[0]
    print(
        f"{style(first_step.name, fg='cyan', bold=True)} "
        f"{style(':', fg='magenta', dim=True)} "
        f'{_format_spec(first_step.output_spec)}'
    )

    for step in pipeline[1:]:
        print(
            f"{style(step.name, fg='cyan', bold=True)} "
            f"{style('->', fg='magenta', dim=True)} "
            f'{_format_spec(step.output_spec)}'
        )
        for child in step.children or []:
            print(
                f"  {style('↳', fg='yellow', bold=True)} "
                f"{style(child.name, fg='blue')} "
                f"{style('->', fg='magenta', dim=True)} "
                f'{_format_spec(child.output_spec)}'
            )

    print(style(' End Trace ', fg='black', bg='yellow', bold=True))
    print()


class QuantizerTrainerDisplay(TrainerDisplay):
    def log_best_epoch(
        self,
        *,
        epoch_label: str,
        epoch_metrics: dict[str, float | int],
        metric_name: str = 'loss',
    ) -> None:
        metric_key = self.resolve_metric_key(metric_name)

        if metric_name == 'codes':
            if 'validation_active_codes' not in epoch_metrics or 'validation_codebook_size' not in epoch_metrics:
                return
            value = (
                f"{int(epoch_metrics['validation_active_codes'])}/"
                f"{int(epoch_metrics['validation_codebook_size'])}"
            )
            value_fg = self.config.meta_value_fg
        else:
            spec = next(
                (
                    metric_spec
                    for metric_spec in SCALAR_METRIC_SPECS
                    if metric_spec[1] == metric_name or metric_spec[0] == metric_key
                ),
                None,
            )
            if spec is None:
                metric_value = epoch_metrics.get(f'validation_{metric_key}')
                if metric_value is None:
                    return
                value = str(metric_value)
                value_fg = self.config.metric_value_fg
            else:
                _metric_key_name, _short_name, color_attr, format_spec = spec
                value = format(float(epoch_metrics[f'validation_{metric_key}']), format_spec)
                value_fg = getattr(self.config, color_attr)

        parts = [
            style(epoch_label, fg=self.config.epoch_index_fg, bold=True),
            self.format_metric(metric_name, value, value_fg=value_fg),
        ]
        parts.extend(
            self._summary_metric_segments(
                epoch_metrics,
                train_validation=False,
                exclude_short_names={metric_name},
            )
        )
        self._print_log('BEST', self._join_segments(*parts), fg=self.config.best_label_fg, bg=self.config.best_label_bg)


class QuantizerTrainer(VQTrainer):
    def __init__(self, *args, metric_directions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_directions = dict(metric_directions or {})

    @staticmethod
    def _best_dir_name(metric_name):
        return 'best' if metric_name == 'loss' else f'best-{metric_name}'

    def _is_better(self, metric_name, metric_value, best_value):
        direction = self.metric_directions.get(metric_name, 'min')
        if direction == 'max':
            return metric_value > best_value
        return metric_value < best_value

    def fit(
        self,
        dataloaders,
        metadata=None,
    ):
        best_validation_metrics = {}
        for metric_name in self.args.save_best_by:
            direction = self.metric_directions.get(metric_name, 'min')
            best_validation_metrics[metric_name] = float('-inf') if direction == 'max' else float('inf')

        best_epochs_by_metric = {metric_name: None for metric_name in self.args.save_best_by}
        epochs_without_improvement = 0
        stopped_early = False
        history = []
        output_dir = Path(self.args.output_dir)
        best_output_dirs = {
            metric_name: output_dir / self._best_dir_name(metric_name)
            for metric_name in self.args.save_best_by
        }
        epoch = 0
        max_epochs = self.args.epochs if self.args.epochs > 0 else None
        self.max_epochs = max_epochs
        self.configure_optimizers_for_fit(
            total_train_batches=len(dataloaders.train),
            max_epochs=max_epochs,
        )

        self.display.log_run_start(
            model_name=metadata.get('model', self.model.__class__.__name__) if metadata else self.model.__class__.__name__,
            dataset_name=metadata.get('dataset', 'unknown') if metadata else 'unknown',
            device=str(self.device),
            epoch_budget='early-stop' if max_epochs is None else str(max_epochs),
        )

        while max_epochs is None or epoch < max_epochs:
            epoch += 1
            self.on_epoch_start(epoch)
            train_metrics = self.train_epoch(dataloaders.train)
            validation_metrics = self.evaluate(dataloaders.validation)
            epoch_metrics = {'epoch': epoch}
            epoch_metrics.update(self.get_epoch_metrics())
            epoch_metrics.update({f'train_{name}': value for name, value in train_metrics.items()})
            epoch_metrics.update({f'validation_{name}': value for name, value in validation_metrics.items()})
            history.append(epoch_metrics)

            improved_metrics = []
            for metric_name in self.args.save_best_by:
                metric_key = DISPLAY_METRIC_KEY_BY_SHORT_NAME[metric_name]
                if metric_key not in validation_metrics:
                    raise KeyError(
                        f"Configured save_best_by metric '{metric_name}' was not produced by validation metrics."
                    )
                metric_value = validation_metrics[metric_key]
                if self._is_better(metric_name, metric_value, best_validation_metrics[metric_name]):
                    best_validation_metrics[metric_name] = metric_value
                    best_epochs_by_metric[metric_name] = epoch
                    self.model.save_pretrained(best_output_dirs[metric_name])
                    improved_metrics.append(metric_name)

            improved = 'loss' in improved_metrics

            if improved_metrics:
                if improved:
                    epochs_without_improvement = 0
                self.display.clear_live_line()
                for metric_name in improved_metrics:
                    self.display.log_best_epoch(
                        epoch_label=self.format_epoch_label(),
                        epoch_metrics=epoch_metrics,
                        metric_name=metric_name,
                    )
            else:
                if self.args.show_only_best_epochs:
                    self.display.log_epoch_summary(
                        epoch_label=self.format_epoch_label(),
                        epoch_metrics=epoch_metrics,
                        persist=False,
                    )
                else:
                    self.display.clear_live_line()
                    self.display.log_epoch_summary(
                        epoch_label=self.format_epoch_label(),
                        epoch_metrics=epoch_metrics,
                        persist=True,
                    )
                epochs_without_improvement += 1
                if self.args.patience is not None and epochs_without_improvement >= self.args.patience:
                    stopped_early = True
                    break

        test_metrics = self.evaluate(dataloaders.test)
        self.model.save_pretrained(output_dir / 'final')

        best_validation_loss = best_validation_metrics.get('loss')
        best_epoch = best_epochs_by_metric.get('loss')

        metrics = {
            'device': str(self.device),
            'best_validation_loss': best_validation_loss,
            'best_epoch': best_epoch,
            'best_validation_metrics': best_validation_metrics,
            'best_epochs_by_metric': best_epochs_by_metric,
            'best_metric_directions': self.metric_directions,
            'epochs_completed': len(history),
            'final_test_loss': test_metrics['loss'],
            'final_test_metrics': test_metrics,
            'history': history,
            'stopped_early': stopped_early,
            'training_args': self.args.to_dict(),
        }
        metrics['advice'] = self.generate_advice(metrics) if self.args.advice else []
        if metadata:
            metrics.update(metadata)

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / 'metrics.json'
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')

        self.display.log_run_end(
            test_metrics=test_metrics,
            output_dir=output_dir,
            metrics_path=metrics_path,
            stopped_early=stopped_early,
            best_epoch=best_epoch,
            current_epoch=self.current_epoch,
            best_output_dirs=best_output_dirs,
        )
        if metrics['advice']:
            self.display.log_advice(metrics['advice'])
        return metrics


class Quantizer:
    def __init__(self, data, model, config):
        self.config = config

        self.data = data
        self.embedding_model = model.replace('.', '').lower()
        self.quantizer_name = self.config.quantizer.name

        self.processor = load_processor(self.data, data_dir=get_data_dir(self.data))
        self.processor.load()

        self.embedding_dir = Path(self.processor.store_dir) / 'embeddings'
        self.embedding_path = self.embedding_dir / f'{self.embedding_model}.npy'
        self.embedding_item_ids_path = self.embedding_dir / f'{self.embedding_model}.item_ids.parquet'
        self.embedding_meta_path = self.embedding_dir / f'{self.embedding_model}.meta.json'
        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f'Embedding file not found: {self.embedding_path}. '
                f'Run `python embedder.py --data {self.data} --model {self.embedding_model}` first.'
            )

        self.output_dir = Path(self.config.trainer.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.codes_path = self.output_dir / 'codebook_indices.npy'
        self.quantized_path = self.output_dir / 'quantized_latents.npy'
        self.codebooks_path = self.output_dir / 'codebooks.npy'
        self.item_ids_path = self.output_dir / 'item_ids.parquet'
        self.meta_path = self.output_dir / 'meta.json'

        self.embedding_matrix = None
        self.item_ids = None
        self.dataset = None
        self.trainer_args = None
        self.model = None
        self.export_metric_name = None

    def _normalize_metric_directions(self):
        save_best_by = list(self.trainer_args.save_best_by)
        direction_config = getattr(self.config.trainer, 'save_best_mode', None)
        direction_config = direction_config() if callable(direction_config) else direction_config

        directions = {}
        for metric_name in save_best_by:
            direction = DEFAULT_BEST_DIRECTIONS.get(metric_name, 'min')
            if direction_config and metric_name in direction_config:
                direction = str(direction_config[metric_name]).lower()
            if direction not in {'min', 'max'}:
                raise ValueError(
                    f"Unsupported save_best_mode for '{metric_name}': {direction}. Use 'min' or 'max'."
                )
            directions[metric_name] = direction
        return directions

    def _resolve_export_metric_name(self):
        export_metric_name = getattr(self.config.trainer, 'export_best_by', None)
        if export_metric_name:
            export_metric_name = str(export_metric_name)
            if export_metric_name not in self.trainer_args.save_best_by:
                raise ValueError(
                    f"export_best_by={export_metric_name} must be included in save_best_by={self.trainer_args.save_best_by}"
                )
            return export_metric_name

        if 'loss' in self.trainer_args.save_best_by:
            return 'loss'
        return self.trainer_args.save_best_by[0]

    def _load_item_ids(self, expected_size):
        if self.embedding_item_ids_path.exists():
            item_ids = pd.read_parquet(self.embedding_item_ids_path)[self.processor.IID_COL].tolist()
        else:
            item_ids = self.processor.items[self.processor.IID_COL].tolist()
        if len(item_ids) != expected_size:
            raise ValueError(
                f'Item id count {len(item_ids)} does not match embedding rows {expected_size} '
                f'for {self.embedding_path}'
            )
        return item_ids

    def load_embedding_matrix(self):
        pnt(f'loading embeddings from {self.embedding_path}')
        embeddings = np.load(self.embedding_path)
        if embeddings.ndim != 2:
            raise ValueError(f'Expected a 2D embedding matrix, got shape {embeddings.shape}')

        item_ids = self._load_item_ids(len(embeddings))
        self.item_ids = item_ids
        matrix = torch.tensor(embeddings, dtype=torch.float32)
        token_to_index = {str(item_id): index for index, item_id in enumerate(item_ids)}

        metadata = {}
        if self.embedding_meta_path.exists():
            metadata = json.loads(self.embedding_meta_path.read_text())

        self.embedding_matrix = EmbeddingMatrix(
            tokens=[str(item_id) for item_id in item_ids],
            matrix=matrix,
            token_to_index=token_to_index,
            source_path=str(self.embedding_path),
            name=f'{self.data}-{self.embedding_model}',
            metadata=metadata,
        )
        return self.embedding_matrix

    def build_dataloaders(self):
        self.dataset = EmbeddingTensorDataset(self.embedding_matrix)
        splits = split_dataset(
            self.dataset,
            validation_ratio=float(self.config.trainer.validation_ratio),
            test_ratio=float(self.config.trainer.test_ratio),
            seed=int(self.config.trainer.seed),
        )
        dataloaders = create_dataloaders(
            splits,
            batch_size=int(self.config.trainer.batch_size),
        )
        return dataloaders

    def build_model(self):
        sample_spec = TensorSpec(shape=(self.embedding_matrix.embedding_dim,))
        decoder_name = None
        decoder_config = None
        if getattr(self.config, 'decoder', None):
            decoder_name = self.config.decoder.name or None
            decoder_config = self.config.decoder.config() if self.config.decoder.config else None

        self.model = load_model(
            self.config.quantizer.name,
            sample_spec=sample_spec,
            encoder=self.config.encoder.name or None,
            encoder_config=self.config.encoder.config() if self.config.encoder.config else None,
            decoder=decoder_name,
            decoder_config=decoder_config,
            **self.config.quantizer.config(),
        )
        return self.model

    def build_trainer(self):
        self.trainer_args = TrainingConfig(**self.config.trainer())
        metric_directions = self._normalize_metric_directions()
        self.export_metric_name = self._resolve_export_metric_name()
        return QuantizerTrainer(
            model=self.model,
            args=self.trainer_args,
            display=QuantizerTrainerDisplay(),
            metric_directions=metric_directions,
        )

    def train(self):
        set_seed(int(self.config.trainer.seed))
        self.load_embedding_matrix()
        dataloaders = self.build_dataloaders()
        self.build_model()
        _print_pipeline_trace(self.model)
        trainer = self.build_trainer()

        pnt(f'training {self.quantizer_name} on {self.data}/{self.embedding_model}')
        metrics = trainer.fit(
            dataloaders,
            metadata={
                'dataset': self.data,
                'embedding_model': self.embedding_model,
                'quantizer_model': self.quantizer_name,
            },
        )
        return metrics

    def load_best_model(self):
        best_dir_name = 'best' if self.export_metric_name == 'loss' else f'best-{self.export_metric_name}'
        best_dir = self.output_dir / best_dir_name
        if not best_dir.exists():
            raise FileNotFoundError(f'Best checkpoint not found: {best_dir}')

        weights_name = getattr(self.model.__class__, 'weights_name', 'pytorch_model.bin')
        weights_path = best_dir / weights_name
        if not weights_path.exists():
            raise FileNotFoundError(f'Best checkpoint weights not found: {weights_path}')

        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        device = resolve_device(self.trainer_args.device)
        self.model.to(device)
        self.model.eval()
        return self.model, device

    def export_codes(self):
        if self.embedding_matrix is None:
            self.load_embedding_matrix()
        if self.dataset is None:
            self.dataset = EmbeddingTensorDataset(self.embedding_matrix)
        if self.model is None or self.trainer_args is None:
            raise RuntimeError('Model and trainer args must be initialized before export.')

        model, device = self.load_best_model()
        export_loader = DataLoader(
            self.dataset,
            batch_size=int(self.trainer_args.batch_size),
            shuffle=False,
            num_workers=0,
        )

        codebook_indices = []
        quantized_latents = []
        codebooks = None

        pnt(f'exporting quantized codes to {self.output_dir}')
        for batch in tqdm(export_loader, total=len(export_loader)):
            batch = batch.to(device)
            artifact = model.export(batch, include_reconstruction=False)
            codebook_indices.append(artifact.codebook_indices.detach().cpu().numpy())
            quantized_latents.append(artifact.quantized_latents.detach().cpu().numpy())
            if codebooks is None and 'codebooks' in artifact.extras:
                codebooks = artifact.extras['codebooks'].detach().cpu().numpy()

        codebook_indices = np.concatenate(codebook_indices, axis=0)
        quantized_latents = np.concatenate(quantized_latents, axis=0).astype(np.float32)

        np.save(self.codes_path, codebook_indices)
        np.save(self.quantized_path, quantized_latents)
        if codebooks is not None:
            np.save(self.codebooks_path, codebooks.astype(np.float32))

        pd.DataFrame({self.processor.IID_COL: self.item_ids}).to_parquet(self.item_ids_path, index=False)

        meta = {
            'dataset': self.data,
            'embedding_model': self.embedding_model,
            'embedding_path': str(self.embedding_path),
            'embedding_meta_path': str(self.embedding_meta_path),
            'quantizer_model': self.quantizer_name,
            'checkpoint': 'best',
            'item_count': int(self.embedding_matrix.num_embeddings),
            'embedding_dim': int(self.embedding_matrix.embedding_dim),
            'trainer_output_dir': str(self.output_dir),
            'codebook_indices_path': str(self.codes_path),
            'quantized_latents_path': str(self.quantized_path),
            'item_ids_path': str(self.item_ids_path),
            'trainer_args': self.trainer_args.to_dict(),
            'quantizer_config': self.config.quantizer.config(),
            'export_metric_name': self.export_metric_name,
        }
        if codebooks is not None:
            meta['codebooks_path'] = str(self.codebooks_path)
            meta['codebook_shape'] = list(codebooks.shape)
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')

        pnt(f'codebook indices saved to {self.codes_path}')
        pnt(f'quantized latents saved to {self.quantized_path}')
        return meta

    def run(self):
        self.train()
        self.export_codes()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False,
    )

    configurations = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            config='config/quantizer.yaml',
        ),
        makedirs=[],
    ).parse()

    quantizer = Quantizer(configurations.data, configurations.model, configurations.config)
    quantizer.run()
