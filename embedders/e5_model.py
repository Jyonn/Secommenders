import abc

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embedders.base_model import BaseModel


class E5Model(BaseModel, abc.ABC):
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key)
        self.model = AutoModel.from_pretrained(self.key)
        self.model.to(self.device)
        self.max_len = self.model.config.max_position_embeddings

    def encode(self, samples: list[str], normalize=False) -> np.ndarray:
        inputs = self.tokenizer(
            samples,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        mask = inputs['attention_mask']

        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state
            embeddings = embeddings.masked_fill(~mask[..., None].bool(), 0.0)
            embeddings = embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

        embeddings = embeddings.float().cpu().numpy()
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings


class E5BaseModel(E5Model):
    KEY = 'intfloat/e5-base-v2'


class E5LargeModel(E5Model):
    KEY = 'intfloat/e5-large-v2'
