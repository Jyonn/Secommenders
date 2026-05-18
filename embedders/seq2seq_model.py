import abc

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from embedders.base_model import BaseModel


class Seq2SeqEncoderModel(BaseModel, abc.ABC):
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key, trust_remote_code=True)
        self.model = T5EncoderModel.from_pretrained(
            self.key,
            torch_dtype=self.get_dtype(),
        )
        self.model.to(self.device)
        self.max_len = getattr(self.model.config, 'n_positions', 512)

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
            hidden_states = outputs.last_hidden_state
            hidden_states = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        embeddings = embeddings.float().cpu().numpy()
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings


class P5Model(Seq2SeqEncoderModel):
    KEY = 't5-base'


class P5BeautyModel(P5Model):
    pass
