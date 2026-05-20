import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoders.data.base import TensorSpec, create_dataloaders, split_dataset
from autoencoders.data.embeddings import EmbeddingMatrix, EmbeddingTensorDataset
from autoencoders.function import resolve_device, set_seed
from autoencoders.models.loading import load_model
from autoencoders.training.display import style
from autoencoders.training.trainer import TrainingConfig, VQTrainer
from utils.config_init import ConfigInit
from utils.artifact import ArtifactStore
from utils.data import get_data_dir
from utils.function import load_processor
from utils.gpu import GPU
from utils.logging import setup_logging


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

class Quantizer:
    def __init__(self, data, model, config):
        self.config = config

        self.data = data
        self.embedding_model = model.replace('.', '').lower()
        self.quantizer_name = self.config.quantizer.name

        self.processor = load_processor(self.data, data_dir=get_data_dir(self.data))
        self.processor.load()

        artifacts = ArtifactStore(self.data)
        self.embedding_dir = artifacts.embedded_dir(self.embedding_model)
        self.embedding_path = self.embedding_dir / 'embeddings.npy'
        self.embedding_item_ids_path = self.embedding_dir / 'item_ids.parquet'
        self.embedding_meta_path = self.embedding_dir / 'meta.json'
        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f'Embedding file not found: {self.embedding_path}. '
                f'Run `python embedder.py --data {self.data} --model {self.embedding_model}` first.'
            )

        self.output_dir = Path(getattr(self.config.trainer, 'output_dir', artifacts.quantized_dir(self.embedding_model)))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_matrix = None
        self.item_ids = None
        self.dataset = None
        self.trainer_args = None
        self.model = None

    def _resolve_device(self):
        device = getattr(self.config.trainer, 'device', None)
        if device is not None and device != 'auto':
            return device
        return GPU.auto_choose(torch_format=True)

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
            full_dataset_as_splits=bool(getattr(self.config.trainer, 'full_dataset_as_splits', False)),
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
        trainer_config = self.config.trainer()
        trainer_config['device'] = self._resolve_device()
        self.trainer_args = TrainingConfig(**trainer_config)
        return VQTrainer(model=self.model, args=self.trainer_args)

    def train(self):
        set_seed(int(self.config.trainer.seed))
        self.load_embedding_matrix()
        dataloaders = self.build_dataloaders()
        self.build_model()
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

    @staticmethod
    def _checkpoint_dir_name(metric_name):
        return 'best' if metric_name == 'loss' else f'best-{metric_name}'

    def _checkpoint_dir_names(self, metric_name):
        names = [self._checkpoint_dir_name(metric_name)]
        # Upstream trainer names the utilization checkpoint `best-usage`,
        # while our config still refers to the metric as `codes`.
        if metric_name == 'codes':
            names.append('best-usage')
        elif metric_name == 'usage':
            names.append('best-codes')
        return names

    def _checkpoint_candidates(self, metric_name):
        candidates = []
        for name in self._checkpoint_dir_names(metric_name):
            candidates.append(self.output_dir / name)
            candidates.append(self.output_dir / 'checkpoints' / name)
        return candidates

    def _export_dir(self, metric_name):
        return self.output_dir / 'exports' / metric_name

    def load_checkpoint_model(self, metric_name):
        weights_name = getattr(self.model.__class__, 'weights_name', 'pytorch_model.bin')
        checkpoint_dir = None
        weights_path = None
        for candidate in self._checkpoint_candidates(metric_name):
            candidate_weights = candidate / weights_name
            if candidate_weights.exists():
                checkpoint_dir = candidate
                weights_path = candidate_weights
                break

        if checkpoint_dir is None or weights_path is None:
            searched = ', '.join(str(path) for path in self._checkpoint_candidates(metric_name))
            raise FileNotFoundError(
                f'Checkpoint not found for {metric_name}. '
                f'Searched: {searched}'
            )

        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        device = resolve_device(self.trainer_args.device)
        self.model.to(device)
        self.model.eval()
        return self.model, device, checkpoint_dir

    def export_checkpoint(self, metric_name):
        if self.embedding_matrix is None:
            self.load_embedding_matrix()
        if self.dataset is None:
            self.dataset = EmbeddingTensorDataset(self.embedding_matrix)
        if self.model is None or self.trainer_args is None:
            raise RuntimeError('Model and trainer args must be initialized before export.')

        export_dir = self._export_dir(metric_name)
        export_dir.mkdir(parents=True, exist_ok=True)

        codes_path = export_dir / 'codebook_indices.npy'
        quantized_path = export_dir / 'quantized_latents.npy'
        codebooks_path = export_dir / 'codebooks.npy'
        item_ids_path = export_dir / 'item_ids.parquet'
        meta_path = export_dir / 'meta.json'

        model, device, checkpoint_dir = self.load_checkpoint_model(metric_name)
        export_loader = DataLoader(
            self.dataset,
            batch_size=int(self.trainer_args.batch_size),
            shuffle=False,
            num_workers=0,
        )

        codebook_indices = []
        quantized_latents = []
        codebooks = None

        pnt(f'exporting {metric_name} checkpoint codes to {export_dir}')
        for batch in tqdm(export_loader, total=len(export_loader)):
            batch = batch.to(device)
            artifact = model.export(batch, include_reconstruction=False)
            codebook_indices.append(artifact.codebook_indices.detach().cpu().numpy())
            quantized_latents.append(artifact.quantized_latents.detach().cpu().numpy())
            if codebooks is None and 'codebooks' in artifact.extras:
                codebooks = artifact.extras['codebooks'].detach().cpu().numpy()

        codebook_indices = np.concatenate(codebook_indices, axis=0)
        quantized_latents = np.concatenate(quantized_latents, axis=0).astype(np.float32)

        np.save(codes_path, codebook_indices)
        np.save(quantized_path, quantized_latents)
        if codebooks is not None:
            np.save(codebooks_path, codebooks.astype(np.float32))

        pd.DataFrame({self.processor.IID_COL: self.item_ids}).to_parquet(item_ids_path, index=False)

        meta = {
            'dataset': self.data,
            'embedding_model': self.embedding_model,
            'embedding_path': str(self.embedding_path),
            'embedding_meta_path': str(self.embedding_meta_path),
            'quantizer_model': self.quantizer_name,
            'processed_items_path': str(Path(self.processor.store_dir) / 'items.parquet'),
            'checkpoint_metric': metric_name,
            'checkpoint_dir': str(checkpoint_dir),
            'item_count': int(self.embedding_matrix.num_embeddings),
            'embedding_dim': int(self.embedding_matrix.embedding_dim),
            'trainer_output_dir': str(self.output_dir),
            'export_dir': str(export_dir),
            'codebook_indices_path': str(codes_path),
            'quantized_latents_path': str(quantized_path),
            'item_ids_path': str(item_ids_path),
            'trainer_args': self.trainer_args.to_dict(),
            'quantizer_config': self.config.quantizer.config(),
        }
        if codebooks is not None:
            meta['codebooks_path'] = str(codebooks_path)
            meta['codebook_shape'] = list(codebooks.shape)
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

        pnt(f'codebook indices saved to {codes_path}')
        pnt(f'quantized latents saved to {quantized_path}')
        return meta

    def export_all_checkpoints(self):
        exports = {}
        for metric_name in self.trainer_args.save_best_by:
            exports[metric_name] = self.export_checkpoint(metric_name)
        return exports

    def run(self):
        self.train()
        self.export_all_checkpoints()


if __name__ == '__main__':
    setup_logging()

    configurations = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            config='config/quantizer.yaml',
        ),
        makedirs=[],
    ).parse()

    quantizer = Quantizer(configurations.data, configurations.model, configurations.config)
    quantizer.run()
