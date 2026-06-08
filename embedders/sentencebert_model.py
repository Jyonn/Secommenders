import numpy as np
from sentence_transformers import SentenceTransformer

from embedders.base_model import BaseModel


class SentenceBertModel(BaseModel):
    KEY = 'efederici/sentence-bert-base'
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SentenceTransformer(self.key, device=self.device)

    def encode(self, samples: list[str], normalize=False) -> np.ndarray:
        embeddings = self.model.encode(
            samples,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


class SentenceT5Model(SentenceBertModel):
    KEY = 'sentence-transformers/sentence-t5-base'


class Qwen3Embedding06bModel(SentenceBertModel):
    KEY = 'Qwen/Qwen3-Embedding-0.6B'


class Qwen3Embedding4bModel(SentenceBertModel):
    KEY = 'Qwen/Qwen3-Embedding-4B'


class Qwen3Embedding8bModel(SentenceBertModel):
    KEY = 'Qwen/Qwen3-Embedding-8B'
