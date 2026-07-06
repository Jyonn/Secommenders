import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoindexers.loading import load_indexer
from autoencoders.data.base import TensorSpec, create_dataloaders, split_dataset
from autoencoders.data.embeddings import EmbeddingMatrix, EmbeddingTensorDataset
from autoencoders.function import resolve_device, set_seed
from autoencoders.models.loading import load_model
from autoencoders.training.display import style
from autoencoders.training.trainer import TrainingConfig, VQTrainer
from utils.config_init import ConfigInit
from utils.artifact import ArtifactStore
from utils.artifact_identity import (
    quantized_artifact_identity,
    register_quantized_artifact,
    resolve_quantized_dir,
)
from utils.data import get_data_dir
from utils.function import load_processor
from utils.gpu import GPU
from utils.logging import setup_logging
from utils.pipeline import ensure_embedded


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
    SUPPORTED_QUANTIZERS = ('rqvae', 'pqvae', 'opqvae', 'lsh', 'simhash', 'pcahash', 'itq')
    HASH_INDEXERS = ('lsh', 'simhash', 'pcahash', 'itq')
    VQ_QUANTIZERS = ('rqvae', 'pqvae', 'opqvae')

    def __init__(self, data, model, config):
        self.config = config

        self.data = data
        self.embedding_model = model.replace('.', '').lower()
        self.quantizer_name = self.config.quantizer.name
        self.quantizer_scheme = self._infer_quantizer_scheme(self.quantizer_name)
        self.recommended_decoding = self._recommended_decoding(self.quantizer_scheme)
        self.requested_latent_dim = None
        self.resolved_latent_dim = None
        self.resolved_quantizer_config = None
        self.resolved_encoder_config = None

        self.processor = load_processor(self.data, data_dir=get_data_dir(self.data))
        self.processor.load()

        artifacts = ArtifactStore(self.data)
        self.embedding_dir = artifacts.embedded_dir(self.embedding_model)
        self.embedding_path = self.embedding_dir / 'embeddings.npy'
        self.embedding_item_ids_path = self.embedding_dir / 'item_ids.parquet'
        self.embedding_meta_path = self.embedding_dir / 'meta.json'
        if not self.embedding_path.exists():
            ensure_embedded(self.data, self.embedding_model)
        if not self.embedding_path.exists():
            raise FileNotFoundError(f'Embedding file not found after auto preparation: {self.embedding_path}')

        self.output_dir = resolve_quantized_dir(self.data, self.embedding_model, self.config)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_matrix = None
        self.item_ids = None
        self.dataset = None
        self.trainer_args = None
        self.model = None

    @classmethod
    def _infer_quantizer_scheme(cls, quantizer_name: str) -> str:
        normalized_name = str(quantizer_name).strip().lower()
        if normalized_name not in cls.SUPPORTED_QUANTIZERS:
            supported = ', '.join(cls.SUPPORTED_QUANTIZERS)
            raise ValueError(
                f'Unsupported quantizer "{quantizer_name}". '
                f'Only {supported} are supported.'
            )
        if normalized_name == 'rqvae':
            return 'rq'
        if normalized_name in {'pqvae', 'opqvae'}:
            return 'pq'
        if normalized_name in cls.HASH_INDEXERS:
            return 'hash'
        raise ValueError(f'Unsupported quantizer "{quantizer_name}"')

    @staticmethod
    def _recommended_decoding(scheme: str) -> str:
        if scheme == 'rq':
            return 'sequential'
        if scheme in {'pq', 'hash'}:
            return 'parallel'
        raise ValueError(f'Unsupported quantizer scheme "{scheme}"')

    @property
    def is_hash_indexer(self):
        return self.quantizer_name in self.HASH_INDEXERS

    @property
    def is_vq_quantizer(self):
        return self.quantizer_name in self.VQ_QUANTIZERS

    def _resolve_device(self):
        device = getattr(self.config.trainer, 'device', None)
        if device is not None and device != 'auto':
            return device
        return GPU.auto_choose(torch_format=True)

    @staticmethod
    def _parse_hidden_dims(value):
        if value is None:
            return None
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(',') if part.strip()]
            if not parts:
                raise ValueError('encoder.config.hidden_dims must contain at least one dimension.')
            return [int(part) for part in parts]
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError('encoder.config.hidden_dims must contain at least one dimension.')
            return [int(part) for part in value]
        raise ValueError(
            'encoder.config.hidden_dims must be a comma-separated string or a list/tuple of integers.'
        )

    def _resolve_encoder_config(self):
        if not getattr(self.config, 'encoder', None) or not self.config.encoder.config:
            return None
        encoder_config = dict(self.config.encoder.config())
        if 'hidden_dims' in encoder_config:
            encoder_config['hidden_dims'] = self._parse_hidden_dims(encoder_config['hidden_dims'])
        return encoder_config

    def _resolve_quantizer_config(self):
        quantizer_config = dict(self.config.quantizer.config())
        if isinstance(quantizer_config.get('sinkhorn_epsilon'), str):
            quantizer_config['sinkhorn_epsilon'] = [
                float(part.strip())
                for part in quantizer_config['sinkhorn_epsilon'].split(',')
                if part.strip()
            ]
        requested_latent_dim = quantizer_config.get('latent_dim')
        self.requested_latent_dim = int(requested_latent_dim) if requested_latent_dim is not None else None

        if self.quantizer_scheme == 'pq':
            num_codebooks = int(quantizer_config.get('num_codebooks', 0))
            if num_codebooks <= 0:
                raise ValueError(
                    f'{self.quantizer_name} requires a positive num_codebooks, got {num_codebooks}.'
                )
            if self.requested_latent_dim is None:
                raise ValueError(
                    f'{self.quantizer_name} requires latent_dim to be configured when auto-resolving PQ slots.'
                )
            resolved_latent_dim = int(math.ceil(self.requested_latent_dim / num_codebooks) * num_codebooks)
            if resolved_latent_dim != self.requested_latent_dim:
                pnt(
                    f'adjusting latent_dim for {self.quantizer_name}: '
                    f'{self.requested_latent_dim} -> {resolved_latent_dim} '
                    f'to make it divisible by num_codebooks={num_codebooks}'
                )
            quantizer_config['latent_dim'] = resolved_latent_dim
            self.resolved_latent_dim = resolved_latent_dim
        else:
            self.resolved_latent_dim = self.requested_latent_dim

        self.resolved_quantizer_config = quantizer_config
        return quantizer_config

    def _resolve_hash_config(self):
        if not getattr(self.config, 'hash', None):
            raise ValueError(
                f'{self.quantizer_name} requires a hash config section in config/quantizer.yaml.'
            )
        hash_config = dict(self.config.hash.config())
        hash_config.setdefault('seed', int(self.config.trainer.seed))
        return hash_config

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
        if self.is_hash_indexer:
            hash_config = self._resolve_hash_config()
            self.resolved_quantizer_config = hash_config
            self.model = load_indexer(
                self.quantizer_name,
                sample_spec=sample_spec,
                **hash_config,
            )
            return self.model

        decoder_name = None
        decoder_config = None
        if getattr(self.config, 'decoder', None):
            decoder_name = self.config.decoder.name or None
            decoder_config = self.config.decoder.config() if self.config.decoder.config else None
        quantizer_config = self._resolve_quantizer_config()
        encoder_config = self._resolve_encoder_config()
        self.resolved_encoder_config = encoder_config

        self.model = load_model(
            self.config.quantizer.name,
            sample_spec=sample_spec,
            encoder=self.config.encoder.name or None,
            encoder_config=encoder_config,
            decoder=decoder_name,
            decoder_config=decoder_config,
            **quantizer_config,
        )
        return self.model

    def build_trainer(self):
        trainer_config = self.config.trainer()
        if isinstance(trainer_config.get('save_best_by'), str):
            trainer_config['save_best_by'] = [
                part.strip()
                for part in trainer_config['save_best_by'].split(',')
                if part.strip()
            ]
        trainer_config['device'] = self._resolve_device()
        self.trainer_args = TrainingConfig(**trainer_config)
        return VQTrainer(model=self.model, args=self.trainer_args)

    def train(self):
        if self.is_hash_indexer:
            raise RuntimeError(
                f'{self.quantizer_name} is a hash indexer and does not use the VQ trainer. '
                'Call run() to build and export hash codes directly.'
            )
        set_seed(int(self.config.trainer.seed))
        self.load_embedding_matrix()
        dataloaders = self.build_dataloaders()
        self.build_model()
        trainer = self.build_trainer()

        pnt(f'training {self.quantizer_name} on {self.data}/{self.embedding_model}')
        if self.requested_latent_dim is not None:
            pnt(
                f'quantizer latent_dim requested={self.requested_latent_dim} '
                f'resolved={self.resolved_latent_dim}'
            )
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
            'quantizer_scheme': self.quantizer_scheme,
            'recommended_decoding': self.recommended_decoding,
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
            'code_shape': list(codebook_indices.shape),
            'trainer_args': self.trainer_args.to_dict(),
            'requested_latent_dim': self.requested_latent_dim,
            'resolved_latent_dim': self.resolved_latent_dim,
            'quantizer_config': self.resolved_quantizer_config or self.config.quantizer.config(),
            'encoder_name': self.config.encoder.name or None,
            'encoder_config': self.resolved_encoder_config,
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
        self.save_root_meta(exports)
        return exports

    def _hash_export_dir(self):
        return self.output_dir / 'exports' / 'hash'

    def _extract_binary_bits(self, indexer):
        if hasattr(indexer, 'binary_codes') and isinstance(indexer.binary_codes, torch.Tensor):
            bits = indexer.binary_codes.detach().cpu().to(torch.uint8)
            return bits, list(bits.shape), {'bit_source': 'binary_codes'}
        if hasattr(indexer, 'hash_codes') and isinstance(indexer.hash_codes, torch.Tensor):
            hash_codes = indexer.hash_codes.detach().cpu().to(torch.uint8)
            flattened = hash_codes.reshape(hash_codes.shape[0], -1).contiguous()
            extras = {
                'bit_source': 'hash_codes',
                'hash_code_shape': list(hash_codes.shape),
            }
            return flattened, list(flattened.shape), extras
        raise ValueError(
            f'Unsupported hash indexer export for {self.quantizer_name}: '
            'expected binary_codes or hash_codes tensor.'
        )

    def export_hash_indexer(self):
        if self.embedding_matrix is None:
            self.load_embedding_matrix()
        if self.model is None:
            self.build_model()

        export_dir = self._hash_export_dir()
        export_dir.mkdir(parents=True, exist_ok=True)
        indexer_dir = export_dir / 'indexer'
        bits_path = export_dir / 'binary_bits.npy'
        item_ids_path = export_dir / 'item_ids.parquet'
        meta_path = export_dir / 'meta.json'

        pnt(f'building hash indexer {self.quantizer_name} for {self.data}/{self.embedding_model}')
        self.model.build(self.embedding_matrix.matrix, item_ids=self.item_ids)
        self.model.save_pretrained(indexer_dir)

        binary_bits, binary_shape, extras = self._extract_binary_bits(self.model)
        np.save(bits_path, binary_bits.numpy().astype(np.uint8))
        pd.DataFrame({self.processor.IID_COL: self.item_ids}).to_parquet(item_ids_path, index=False)

        total_bits = int(binary_bits.shape[1]) if binary_bits.ndim == 2 else int(binary_bits.numel())
        meta = {
            'dataset': self.data,
            'embedding_model': self.embedding_model,
            'embedding_path': str(self.embedding_path),
            'embedding_meta_path': str(self.embedding_meta_path),
            'representation_family': 'hash',
            'hash_model': self.quantizer_name,
            'quantizer_model': self.quantizer_name,
            'quantizer_scheme': self.quantizer_scheme,
            'recommended_decoding': self.recommended_decoding,
            'processed_items_path': str(Path(self.processor.store_dir) / 'items.parquet'),
            'item_count': int(self.embedding_matrix.num_embeddings),
            'embedding_dim': int(self.embedding_matrix.embedding_dim),
            'trainer_output_dir': str(self.output_dir),
            'export_dir': str(export_dir),
            'indexer_dir': str(indexer_dir),
            'binary_bits_path': str(bits_path),
            'item_ids_path': str(item_ids_path),
            'binary_bits_shape': binary_shape,
            'num_bits_total': total_bits,
            'recommended_token_count': 3,
            'bit_packing': 'grouped-msb-first',
            'hash_config': self.resolved_quantizer_config or {},
        }
        meta.update(extras)
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        self.save_root_meta({'hash': meta})

        pnt(f'binary hash bits saved to {bits_path}')
        return meta

    def save_root_meta(self, exports: dict):
        identity = quantized_artifact_identity(self.data, self.embedding_model, self.config, self.output_dir)
        root_meta = {
            'stage': 'quantized',
            'dataset': self.data,
            'embedding_model': self.embedding_model,
            'quantizer_model': self.quantizer_name,
            'quantizer_scheme': self.quantizer_scheme,
            'recommended_decoding': self.recommended_decoding,
            'trainer_output_dir': str(self.output_dir),
            'exports': exports,
            'artifact_identity': identity,
        }
        (self.output_dir / 'meta.json').write_text(json.dumps(root_meta, indent=2) + '\n')
        register_quantized_artifact(
            self.data,
            self.embedding_model,
            self.config,
            self.output_dir,
            aliases=identity.get('aliases'),
        )

    def run(self):
        if self.is_hash_indexer:
            set_seed(int(self.config.trainer.seed))
            self.load_embedding_matrix()
            self.build_model()
            self.export_hash_indexer()
            return
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
