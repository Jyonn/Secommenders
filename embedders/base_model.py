import abc
import warnings
from typing import Iterable

import numpy as np
import torch

from utils import model


class BaseModel(abc.ABC):
    KEY = None
    BIT = 16

    def __init__(self, device='cpu', batch_size=32, bit=None):
        self.bit = bit or self.BIT
        self.device = device
        self.batch_size = batch_size
        self.key = model.match(self.get_name()) or self.KEY

        self.model = None
        self.tokenizer = None
        self.max_len = None

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Model', '').lower()

    def get_dtype(self):
        if str(self.device).startswith('cpu'):
            return torch.float32
        if self.bit == 16:
            return torch.bfloat16
        if self.bit == 32:
            return torch.float32
        warnings.warn(f'unsupported bit: {self.bit}, using auto')
        return 'auto'

    def post_init(self):
        if self.model is not None:
            self.model.eval()
        return self

    @staticmethod
    def normalize(embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms

    @abc.abstractmethod
    def encode(self, samples: list[str], normalize=False) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, sample: str, normalize=False) -> np.ndarray:
        return self.encode([sample], normalize=normalize)[0]

    def iter_batches(self, samples: Iterable[str]):
        batch = []
        for sample in samples:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
