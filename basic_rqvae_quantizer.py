import importlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pigmento import pnt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.artifact import ArtifactStore
from utils.compile import normalize_model_name
from utils.config_init import ConfigInit
from utils.function import load_processor
from utils.gpu import GPU
from utils.logging import setup_logging
from utils.pipeline import ensure_embedded


def _parse_list(value, cast, name: str):
    if value is None:
        raise ValueError(f'{name} must not be null')
    if isinstance(value, (list, tuple)):
        values = [cast(part) for part in value]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        values = [cast(part) for part in parts]
    else:
        values = [cast(value)]
    if not values:
        raise ValueError(f'{name} must contain at least one value')
    return values


def _import_basic_rqvae():
    rqvae_file = Path(__file__).resolve().parent / '.rqvae'
    if not rqvae_file.exists():
        raise FileNotFoundError(
            f'basic-rqvae locator file not found: {rqvae_file}. '
            'Please create .rqvae with the path to the basic-rqvae repository or its RQ-VAE subdirectory.'
        )
    raw_root = rqvae_file.read_text().strip()
    if not raw_root:
        raise ValueError(f'basic-rqvae locator file is empty: {rqvae_file}')

    configured_root = Path(raw_root).expanduser()
    if not configured_root.is_absolute():
        configured_root = (rqvae_file.parent / configured_root).resolve()

    root = configured_root
    if not root.exists():
        raise FileNotFoundError(
            f'basic-rqvae source directory not found: {root}. '
            f'Configured by {rqvae_file} -> {raw_root}'
        )

    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == 'utils' or name == 'datasets' or name == 'trainer' or name == 'structure' or name == 'models' or name.startswith('models.'):
            sys.modules.pop(name, None)

    dataset_module = importlib.import_module('datasets')
    trainer_module = importlib.import_module('trainer')
    rqvae_module = importlib.import_module('models.rqvae')
    return dataset_module.EmbDataset, trainer_module.Trainer, rqvae_module.RQVAE, root


BasicEmbDataset, BasicTrainer, BasicRQVAE, BASIC_RQVAE_ROOT = _import_basic_rqvae()


class BasicRQVAEQuantizer:
    QUANTIZER_NAME = 'basic-rqvae'
    EXPORT_NAME = 'loss'

    def __init__(self, data: str, model: str, config):
        self.data = data
        self.embedding_model = normalize_model_name(model)
        self.config = config
        self.processor = load_processor(self.data)
        self.processor.load()

        self.store = ArtifactStore(self.data)
        self.embedding_dir = self.store.embedded_dir(self.embedding_model)
        self.embedding_path = self.embedding_dir / 'embeddings.npy'
        self.embedding_item_ids_path = self.embedding_dir / 'item_ids.parquet'
        self.embedding_meta_path = self.embedding_dir / 'meta.json'
        if not self.embedding_path.exists():
            ensure_embedded(self.data, self.embedding_model)
        if not self.embedding_path.exists():
            raise FileNotFoundError(f'Embedding file not found after auto preparation: {self.embedding_path}')

        self.output_dir = Path(getattr(self.config.trainer, 'output_dir', self.store.quantized_dir(self.embedding_model, self.QUANTIZER_NAME)))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.output_dir / 'exports' / self.EXPORT_NAME
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device()
        self.embeddings = None
        self.item_ids = None

    def _resolve_device(self):
        device = getattr(self.config.trainer, 'device', None)
        if device is not None and str(device).strip().lower() != 'auto':
            return str(device)
        return GPU.auto_choose(torch_format=True)

    def _set_seed(self):
        seed = int(getattr(self.config.trainer, 'seed', 2024))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_item_ids(self, expected_size: int):
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

    def load_embeddings(self):
        pnt(f'loading embeddings from {self.embedding_path}')
        embeddings = np.load(self.embedding_path)
        if embeddings.ndim != 2:
            raise ValueError(f'Expected a 2D embedding matrix, got shape {embeddings.shape}')
        self.embeddings = embeddings.astype(np.float32)
        self.item_ids = self._load_item_ids(len(self.embeddings))
        return self.embeddings

    def _build_args(self):
        num_emb_list = _parse_list(self.config.model.num_emb_list, int, 'model.num_emb_list')
        sk_epsilons = _parse_list(self.config.model.sk_epsilons, float, 'model.sk_epsilons')
        layers = _parse_list(self.config.model.layers, int, 'model.layers')
        if len(sk_epsilons) != len(num_emb_list):
            raise ValueError(
                'model.sk_epsilons must have the same number of entries as model.num_emb_list '
                f'({len(sk_epsilons)} vs {len(num_emb_list)})'
            )
        return Namespace(
            lr=float(self.config.trainer.lr),
            epochs=int(self.config.trainer.epochs),
            batch_size=int(self.config.trainer.batch_size),
            num_workers=int(self.config.trainer.num_workers),
            eval_step=int(self.config.trainer.eval_step),
            learner=str(self.config.trainer.learner),
            lr_scheduler_type=str(self.config.trainer.lr_scheduler_type),
            warmup_epochs=int(self.config.trainer.warmup_epochs),
            data_path=str(self.embedding_path),
            weight_decay=float(self.config.trainer.weight_decay),
            dropout_prob=float(self.config.model.dropout_prob),
            bn=bool(self.config.model.bn),
            loss_type=str(self.config.model.loss_type),
            kmeans_init=bool(self.config.model.kmeans_init),
            kmeans_iters=int(self.config.model.kmeans_iters),
            sk_epsilons=sk_epsilons,
            sk_iters=int(self.config.model.sk_iters),
            device=self.device,
            num_emb_list=num_emb_list,
            e_dim=int(self.config.model.e_dim),
            quant_loss_weight=float(self.config.model.quant_loss_weight),
            beta=float(self.config.model.beta),
            layers=layers,
            save_limit=int(self.config.trainer.save_limit),
            ckpt_dir=str(self.output_dir / 'checkpoints'),
        )

    def _build_model(self, args):
        return BasicRQVAE(
            in_dim=int(self.embeddings.shape[1]),
            num_emb_list=list(args.num_emb_list),
            e_dim=int(args.e_dim),
            layers=list(args.layers),
            dropout_prob=float(args.dropout_prob),
            bn=bool(args.bn),
            loss_type=str(args.loss_type),
            quant_loss_weight=float(args.quant_loss_weight),
            beta=float(args.beta),
            kmeans_init=bool(args.kmeans_init),
            kmeans_iters=int(args.kmeans_iters),
            sk_epsilons=list(args.sk_epsilons),
            sk_iters=int(args.sk_iters),
            use_linear=int(getattr(self.config.model, 'use_linear', 0)),
        )

    def train(self, args):
        self._set_seed()
        dataset = BasicEmbDataset(str(self.embedding_path))
        dataloader = DataLoader(
            dataset,
            num_workers=int(args.num_workers),
            batch_size=int(args.batch_size),
            shuffle=True,
            pin_memory=str(self.device).startswith('cuda'),
        )
        model = self._build_model(args)
        trainer = BasicTrainer(args, model, len(dataloader))
        pnt(f'training {self.QUANTIZER_NAME} on {self.data}/{self.embedding_model} via {BASIC_RQVAE_ROOT}')
        best_loss, best_collision_rate = trainer.fit(dataloader)
        checkpoint_path = Path(trainer.ckpt_dir) / trainer.best_loss_ckpt
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'best loss checkpoint not found after training: {checkpoint_path}')
        return checkpoint_path, best_loss, best_collision_rate

    def _resolve_checkpoint_path(self, args):
        load_ckpt = getattr(self.config.trainer, 'load_ckpt', None)
        if load_ckpt:
            checkpoint_path = Path(str(load_ckpt)).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
            pnt(f'using existing checkpoint {checkpoint_path}')
            return checkpoint_path, None, None
        return self.train(args)

    def export(self, checkpoint_path: Path, args):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = self._build_model(args)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()

        codebook_sizes = list(args.num_emb_list)
        unique_sizes = sorted(set(int(size) for size in codebook_sizes))
        if len(unique_sizes) != 1:
            raise ValueError(
                f'{self.QUANTIZER_NAME} export expects equal per-slot codebook sizes for compiler compatibility, '
                f'got {codebook_sizes}'
            )
        codebook_size = int(unique_sizes[0])

        export_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.embeddings)),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=str(self.device).startswith('cuda'),
        )

        codebook_indices = []
        quantized_latents = []

        pnt(f'exporting loss checkpoint codes to {self.export_dir}')
        with torch.no_grad():
            for (batch,) in tqdm(export_loader, total=len(export_loader)):
                batch = batch.to(self.device)
                encoded = model.encoder(batch)
                x_q, _, indices, _ = model.rq(encoded, use_sk=False)
                codebook_indices.append(indices.view(-1, indices.shape[-1]).cpu().numpy().astype(np.int64))
                quantized_latents.append(x_q.cpu().numpy().astype(np.float32))

        codebook_indices = np.concatenate(codebook_indices, axis=0)
        quantized_latents = np.concatenate(quantized_latents, axis=0).astype(np.float32)
        codebooks = model.rq.get_codebook().detach().cpu().numpy().astype(np.float32)

        codes_path = self.export_dir / 'codebook_indices.npy'
        quantized_path = self.export_dir / 'quantized_latents.npy'
        codebooks_path = self.export_dir / 'codebooks.npy'
        item_ids_path = self.export_dir / 'item_ids.parquet'
        meta_path = self.export_dir / 'meta.json'

        np.save(codes_path, codebook_indices)
        np.save(quantized_path, quantized_latents)
        np.save(codebooks_path, codebooks)
        pd.DataFrame({self.processor.IID_COL: self.item_ids}).to_parquet(item_ids_path, index=False)

        meta = {
            'dataset': self.data,
            'embedding_model': self.embedding_model,
            'embedding_path': str(self.embedding_path),
            'embedding_meta_path': str(self.embedding_meta_path),
            'quantizer_model': self.QUANTIZER_NAME,
            'quantizer_scheme': 'rq',
            'recommended_decoding': 'sequential',
            'processed_items_path': str(self.store.processed_dir() / 'items.parquet'),
            'checkpoint_metric': self.EXPORT_NAME,
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_dir': str(checkpoint_path.parent),
            'item_count': int(len(self.item_ids)),
            'embedding_dim': int(self.embeddings.shape[1]),
            'trainer_output_dir': str(self.output_dir),
            'export_dir': str(self.export_dir),
            'codebook_indices_path': str(codes_path),
            'quantized_latents_path': str(quantized_path),
            'codebooks_path': str(codebooks_path),
            'item_ids_path': str(item_ids_path),
            'code_shape': list(codebook_indices.shape),
            'codebook_shape': list(codebooks.shape),
            'trainer_args': vars(args),
            'requested_latent_dim': int(args.e_dim),
            'resolved_latent_dim': int(args.e_dim),
            'quantizer_config': {
                'codebook_size': codebook_size,
                'num_quantizers': int(len(codebook_sizes)),
                'num_emb_list': [int(size) for size in codebook_sizes],
                'latent_dim': int(args.e_dim),
                'e_dim': int(args.e_dim),
                'layers': [int(layer) for layer in args.layers],
                'dropout_prob': float(args.dropout_prob),
                'bn': bool(args.bn),
                'loss_type': str(args.loss_type),
                'quant_loss_weight': float(args.quant_loss_weight),
                'beta': float(args.beta),
                'kmeans_init': bool(args.kmeans_init),
                'kmeans_iters': int(args.kmeans_iters),
                'sk_epsilons': [float(value) for value in args.sk_epsilons],
                'sk_iters': int(args.sk_iters),
                'use_linear': int(getattr(self.config.model, 'use_linear', 0)),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

        pnt(f'codebook indices saved to {codes_path}')
        pnt(f'quantized latents saved to {quantized_path}')
        pnt(f'codebooks saved to {codebooks_path}')
        pnt(f'item ids saved to {item_ids_path}')
        pnt(f'basic-rqvae export ready under {self.export_dir}')

    def run(self):
        self.load_embeddings()
        args = self._build_args()
        checkpoint_path, _, _ = self._resolve_checkpoint_path(args)
        self.export(checkpoint_path, args)


if __name__ == '__main__':
    setup_logging()
    configurations = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            config='config/basic_rqvae_quantizer.yaml',
        ),
        makedirs=[],
    ).parse()
    quantizer = BasicRQVAEQuantizer(configurations.data, configurations.model, configurations.config)
    quantizer.run()
