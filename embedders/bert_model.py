import abc

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embedders.base_model import BaseModel


class EncoderModel(BaseModel, abc.ABC):
    BIT = 32
    POOLING = 'cls'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.key,
            torch_dtype=self.get_dtype(),
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.max_len = getattr(self.model.config, 'max_position_embeddings', 512)

    def pool(self, hidden_states, attention_mask):
        if self.POOLING == 'mean':
            masked = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return hidden_states[:, 0, :]

    def encode(self, samples: list[str], normalize=False) -> np.ndarray:
        inputs = self.tokenizer(
            samples,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.pool(outputs.last_hidden_state, attention_mask)

        embeddings = embeddings.float().cpu().numpy()
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings


class BertModel(EncoderModel):
    KEY = 'bert-base-uncased'


class BertBaseModel(BertModel):
    pass


class BertLargeModel(EncoderModel):
    KEY = 'bert-large-uncased'


class RecformerModel(EncoderModel):
    KEY = 'allenai/longformer-base-4096'
