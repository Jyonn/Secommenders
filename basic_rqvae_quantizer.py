import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pigmento import pnt

from utils.artifact import ArtifactStore
from utils.compile import normalize_model_name
from utils.config_init import ConfigInit
from utils.function import load_processor
from utils.gpu import GPU
from utils.logging import setup_logging
from utils.pipeline import ensure_embedded


EXTERNAL_TRAIN_CODE = r"""
import json
import logging
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer

config_path = Path(sys.argv[1])
cfg = json.loads(config_path.read_text())

logging.basicConfig(level=logging.INFO)

seed = int(cfg["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = Namespace(**cfg["trainer_args"])
data = EmbDataset(cfg["embedding_path"])
print(f"[basic-rqvae-train] loading embeddings from {cfg['embedding_path']}", flush=True)
model = RQVAE(
    in_dim=data.dim,
    num_emb_list=cfg["model_args"]["num_emb_list"],
    e_dim=cfg["model_args"]["e_dim"],
    layers=cfg["model_args"]["layers"],
    dropout_prob=cfg["model_args"]["dropout_prob"],
    bn=cfg["model_args"]["bn"],
    loss_type=cfg["model_args"]["loss_type"],
    quant_loss_weight=cfg["model_args"]["quant_loss_weight"],
    beta=cfg["model_args"]["beta"],
    kmeans_init=cfg["model_args"]["kmeans_init"],
    kmeans_iters=cfg["model_args"]["kmeans_iters"],
    sk_epsilons=cfg["model_args"]["sk_epsilons"],
    sk_iters=cfg["model_args"]["sk_iters"],
    use_linear=cfg["model_args"]["use_linear"],
)
dataloader = DataLoader(
    data,
    num_workers=int(args.num_workers),
    batch_size=int(args.batch_size),
    shuffle=True,
    pin_memory=str(args.device).startswith("cuda"),
)
print(
    f"[basic-rqvae-train] start training items={len(data)} batch_size={args.batch_size} "
    f"device={args.device} ckpt_dir={args.ckpt_dir}",
    flush=True,
)
trainer = Trainer(args, model, len(dataloader))
best_loss, best_collision_rate = trainer.fit(dataloader)
checkpoint_path = Path(trainer.ckpt_dir) / trainer.best_loss_ckpt
if not checkpoint_path.exists():
    raise FileNotFoundError(f"best loss checkpoint not found after training: {checkpoint_path}")

result = {
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_dir": str(checkpoint_path.parent),
    "best_loss": float(best_loss),
    "best_collision_rate": float(best_collision_rate),
}
Path(cfg["result_path"]).write_text(json.dumps(result, indent=2) + "\n")
"""


EXTERNAL_EXPORT_CODE = r"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.rqvae import RQVAE


config_path = Path(sys.argv[1])
cfg = json.loads(config_path.read_text())
args = Namespace(**cfg["trainer_args"])
model_args = cfg["model_args"]
device = cfg["device"]

embeddings = np.load(cfg["embedding_path"]).astype(np.float32)
item_ids = pd.read_parquet(cfg["item_ids_path"])[cfg["item_col"]].tolist()
if len(item_ids) != len(embeddings):
    raise ValueError(
        f"Item id count {len(item_ids)} does not match embedding rows {len(embeddings)} "
        f"for {cfg['embedding_path']}"
    )

model = RQVAE(
    in_dim=int(embeddings.shape[1]),
    num_emb_list=model_args["num_emb_list"],
    e_dim=model_args["e_dim"],
    layers=model_args["layers"],
    dropout_prob=model_args["dropout_prob"],
    bn=model_args["bn"],
    loss_type=model_args["loss_type"],
    quant_loss_weight=model_args["quant_loss_weight"],
    beta=model_args["beta"],
    kmeans_init=model_args["kmeans_init"],
    kmeans_iters=model_args["kmeans_iters"],
    sk_epsilons=model_args["sk_epsilons"],
    sk_iters=model_args["sk_iters"],
    use_linear=model_args["use_linear"],
)
checkpoint = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)
model.eval()

codebook_sizes = [int(size) for size in model_args["num_emb_list"]]
unique_sizes = sorted(set(codebook_sizes))
if len(unique_sizes) != 1:
    raise ValueError(
        "basic-rqvae export expects equal per-slot codebook sizes for compiler compatibility, "
        f"got {codebook_sizes}"
    )
codebook_size = int(unique_sizes[0])

loader = DataLoader(
    TensorDataset(torch.from_numpy(embeddings)),
    batch_size=int(args.batch_size),
    shuffle=False,
    num_workers=0,
    pin_memory=str(device).startswith("cuda"),
)

codebook_indices = []
quantized_latents = []
print(f"[basic-rqvae-export] exporting {len(embeddings)} embeddings to {cfg['export_dir']}", flush=True)
with torch.no_grad():
    for (batch,) in tqdm(loader, total=len(loader)):
        batch = batch.to(device)
        encoded = model.encoder(batch)
        x_q, _, indices, _ = model.rq(encoded, use_sk=False)
        codebook_indices.append(indices.view(-1, indices.shape[-1]).cpu().numpy().astype(np.int64))
        quantized_latents.append(x_q.cpu().numpy().astype(np.float32))

codebook_indices = np.concatenate(codebook_indices, axis=0)
quantized_latents = np.concatenate(quantized_latents, axis=0).astype(np.float32)
codebooks = model.rq.get_codebook().detach().cpu().numpy().astype(np.float32)

export_dir = Path(cfg["export_dir"])
export_dir.mkdir(parents=True, exist_ok=True)
codes_path = export_dir / "codebook_indices.npy"
quantized_path = export_dir / "quantized_latents.npy"
codebooks_path = export_dir / "codebooks.npy"
item_ids_out_path = export_dir / "item_ids.parquet"
meta_path = export_dir / "meta.json"

np.save(codes_path, codebook_indices)
np.save(quantized_path, quantized_latents)
np.save(codebooks_path, codebooks)
pd.DataFrame({cfg["item_col"]: item_ids}).to_parquet(item_ids_out_path, index=False)

meta = {
    "dataset": cfg["dataset"],
    "embedding_model": cfg["embedding_model"],
    "embedding_path": cfg["embedding_path"],
    "embedding_meta_path": cfg["embedding_meta_path"],
    "quantizer_model": "basic-rqvae",
    "quantizer_scheme": "rq",
    "recommended_decoding": "sequential",
    "processed_items_path": cfg["processed_items_path"],
    "checkpoint_metric": "loss",
    "checkpoint_path": cfg["checkpoint_path"],
    "checkpoint_dir": str(Path(cfg["checkpoint_path"]).parent),
    "item_count": int(len(item_ids)),
    "embedding_dim": int(embeddings.shape[1]),
    "trainer_output_dir": cfg["trainer_output_dir"],
    "export_dir": str(export_dir),
    "codebook_indices_path": str(codes_path),
    "quantized_latents_path": str(quantized_path),
    "codebooks_path": str(codebooks_path),
    "item_ids_path": str(item_ids_out_path),
    "code_shape": list(codebook_indices.shape),
    "codebook_shape": list(codebooks.shape),
    "trainer_args": cfg["trainer_args"],
    "requested_latent_dim": int(model_args["e_dim"]),
    "resolved_latent_dim": int(model_args["e_dim"]),
    "quantizer_config": {
        "codebook_size": codebook_size,
        "num_quantizers": int(len(codebook_sizes)),
        "num_emb_list": codebook_sizes,
        "latent_dim": int(model_args["e_dim"]),
        "e_dim": int(model_args["e_dim"]),
        "layers": [int(layer) for layer in model_args["layers"]],
        "dropout_prob": float(model_args["dropout_prob"]),
        "bn": bool(model_args["bn"]),
        "loss_type": str(model_args["loss_type"]),
        "quant_loss_weight": float(model_args["quant_loss_weight"]),
        "beta": float(model_args["beta"]),
        "kmeans_init": bool(model_args["kmeans_init"]),
        "kmeans_iters": int(model_args["kmeans_iters"]),
        "sk_epsilons": [float(value) for value in model_args["sk_epsilons"]],
        "sk_iters": int(model_args["sk_iters"]),
        "use_linear": int(model_args["use_linear"]),
    },
}
meta_path.write_text(json.dumps(meta, indent=2) + "\n")
print(f"[basic-rqvae-export] wrote export files under {export_dir}", flush=True)
"""


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


def _resolve_basic_rqvae_root():
    rqvae_file = Path(__file__).resolve().parent / '.rqvae'
    if not rqvae_file.exists():
        raise FileNotFoundError(
            f'basic-rqvae locator file not found: {rqvae_file}. '
            'Please create .rqvae with the path to the RQ-VAE repository root.'
        )
    raw_root = rqvae_file.read_text().strip()
    if not raw_root:
        raise ValueError(f'basic-rqvae locator file is empty: {rqvae_file}')

    configured_root = Path(raw_root).expanduser()
    if not configured_root.is_absolute():
        configured_root = (rqvae_file.parent / configured_root).resolve()

    candidates = [configured_root, configured_root / 'RQ-VAE']
    for candidate in candidates:
        if (
            candidate.exists()
            and (candidate / 'trainer.py').exists()
            and (candidate / 'datasets.py').exists()
            and (candidate / 'models' / 'rqvae.py').exists()
        ):
            return candidate
    raise FileNotFoundError(
        'basic-rqvae source directory not found. '
        f'Tried: {candidates}. Configured by {rqvae_file} -> {raw_root}'
    )


class BasicRQVAEQuantizer:
    QUANTIZER_NAME = 'basic-rqvae'
    EXPORT_NAME = 'loss'

    def __init__(self, data: str, model: str, config):
        self.data = data
        self.embedding_model = normalize_model_name(model)
        self.config = config
        self.rqvae_root = _resolve_basic_rqvae_root()
        self.python_executable = sys.executable
        self.repo_root = Path(__file__).resolve().parent

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

        self.output_dir = Path(
            getattr(
                self.config.trainer,
                'output_dir',
                self.store.quantized_dir(self.embedding_model, self.QUANTIZER_NAME),
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.output_dir / 'exports' / self.EXPORT_NAME
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device()
        self.embeddings = None
        self.item_ids = None

    def _absolute(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()

    def _resolve_device(self):
        device = getattr(self.config.trainer, 'device', None)
        if device is not None and str(device).strip().lower() != 'auto':
            return str(device)
        return GPU.auto_choose(torch_format=True)

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

    def _model_args(self):
        num_emb_list = _parse_list(self.config.model.num_emb_list, int, 'model.num_emb_list')
        sk_epsilons = _parse_list(self.config.model.sk_epsilons, float, 'model.sk_epsilons')
        layers = _parse_list(self.config.model.layers, int, 'model.layers')
        if len(sk_epsilons) != len(num_emb_list):
            raise ValueError(
                'model.sk_epsilons must have the same number of entries as model.num_emb_list '
                f'({len(sk_epsilons)} vs {len(num_emb_list)})'
            )
        return {
            'num_emb_list': num_emb_list,
            'e_dim': int(self.config.model.e_dim),
            'layers': layers,
            'dropout_prob': float(self.config.model.dropout_prob),
            'bn': bool(self.config.model.bn),
            'loss_type': str(self.config.model.loss_type),
            'quant_loss_weight': float(self.config.model.quant_loss_weight),
            'beta': float(self.config.model.beta),
            'kmeans_init': bool(self.config.model.kmeans_init),
            'kmeans_iters': int(self.config.model.kmeans_iters),
            'sk_epsilons': sk_epsilons,
            'sk_iters': int(self.config.model.sk_iters),
            'use_linear': int(getattr(self.config.model, 'use_linear', 0)),
        }

    def _trainer_args(self):
        return {
            'lr': float(self.config.trainer.lr),
            'epochs': int(self.config.trainer.epochs),
            'batch_size': int(self.config.trainer.batch_size),
            'num_workers': int(self.config.trainer.num_workers),
            'eval_step': int(self.config.trainer.eval_step),
            'learner': str(self.config.trainer.learner),
            'lr_scheduler_type': str(self.config.trainer.lr_scheduler_type),
            'warmup_epochs': int(self.config.trainer.warmup_epochs),
            'data_path': str(self._absolute(self.embedding_path)),
            'weight_decay': float(self.config.trainer.weight_decay),
            'dropout_prob': float(self.config.model.dropout_prob),
            'bn': bool(self.config.model.bn),
            'loss_type': str(self.config.model.loss_type),
            'kmeans_init': bool(self.config.model.kmeans_init),
            'kmeans_iters': int(self.config.model.kmeans_iters),
            'sk_epsilons': _parse_list(self.config.model.sk_epsilons, float, 'model.sk_epsilons'),
            'sk_iters': int(self.config.model.sk_iters),
            'device': self.device,
            'num_emb_list': _parse_list(self.config.model.num_emb_list, int, 'model.num_emb_list'),
            'e_dim': int(self.config.model.e_dim),
            'quant_loss_weight': float(self.config.model.quant_loss_weight),
            'beta': float(self.config.model.beta),
            'layers': _parse_list(self.config.model.layers, int, 'model.layers'),
            'save_limit': int(self.config.trainer.save_limit),
            'verbose': int(getattr(self.config.trainer, 'verbose', 0)),
            'patience': int(getattr(self.config.trainer, 'patience', 250)),
            'ckpt_dir': str(self._absolute(self.output_dir / 'checkpoints')),
            'seed': int(getattr(self.config.trainer, 'seed', 2024)),
        }

    def _run_external_python(self, code: str, payload: dict, stage: str):
        with tempfile.NamedTemporaryFile('w', suffix=f'.{stage}.json', delete=False) as handle:
            config_path = Path(handle.name)
            handle.write(json.dumps(payload, indent=2) + '\n')
        try:
            pnt(f'running external {stage} via {self.python_executable} in {self.rqvae_root}')
            subprocess.run(
                [self.python_executable, '-c', code, str(config_path)],
                cwd=self.rqvae_root,
                check=True,
            )
        finally:
            config_path.unlink(missing_ok=True)

    def _resolve_checkpoint_path(self):
        load_ckpt = getattr(self.config.trainer, 'load_ckpt', None)
        if load_ckpt:
            checkpoint_path = Path(str(load_ckpt)).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
            pnt(f'using existing checkpoint {checkpoint_path}')
            return checkpoint_path

        result_path = self.output_dir / 'basic_rqvae_train_result.json'
        payload = {
            'embedding_path': str(self._absolute(self.embedding_path)),
            'model_args': self._model_args(),
            'trainer_args': self._trainer_args(),
            'seed': int(getattr(self.config.trainer, 'seed', 2024)),
            'result_path': str(self._absolute(result_path)),
        }
        self._run_external_python(EXTERNAL_TRAIN_CODE, payload, 'basic-rqvae-train')
        if not result_path.exists():
            raise FileNotFoundError(f'basic-rqvae training did not produce result file: {result_path}')
        result = json.loads(result_path.read_text())
        checkpoint_path = Path(result['checkpoint_path'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'basic-rqvae checkpoint not found after training: {checkpoint_path}')
        return checkpoint_path

    def export(self, checkpoint_path: Path):
        item_ids_path = self.embedding_item_ids_path
        if not item_ids_path.exists():
            fallback = self.store.processed_dir() / 'items.parquet'
            if not fallback.exists():
                raise FileNotFoundError(
                    f'Embedding item ids not found: {item_ids_path}, and fallback processed items missing: {fallback}'
                )
            item_ids_path = fallback
        payload = {
            'dataset': self.data,
            'embedding_model': self.embedding_model,
            'embedding_path': str(self._absolute(self.embedding_path)),
            'embedding_meta_path': str(self._absolute(self.embedding_meta_path)),
            'item_ids_path': str(self._absolute(item_ids_path)),
            'item_col': self.processor.IID_COL,
            'processed_items_path': str(self._absolute(self.store.processed_dir() / 'items.parquet')),
            'trainer_output_dir': str(self._absolute(self.output_dir)),
            'export_dir': str(self._absolute(self.export_dir)),
            'checkpoint_path': str(self._absolute(checkpoint_path)),
            'device': self.device,
            'model_args': self._model_args(),
            'trainer_args': self._trainer_args(),
        }
        self._run_external_python(EXTERNAL_EXPORT_CODE, payload, 'basic-rqvae-export')

    def run(self):
        self.load_embeddings()
        checkpoint_path = self._resolve_checkpoint_path()
        self.export(checkpoint_path)
        pnt(f'basic-rqvae export ready under {self.export_dir}')


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
