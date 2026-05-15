import abc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedders.base_model import BaseModel


class CausalLMModel(BaseModel, abc.ABC):
    BIT = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.key,
            torch_dtype=self.get_dtype(),
        )
        self.model.to(self.device)
        self.max_len = getattr(self.model.config, 'max_position_embeddings', 2048)

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
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            indices = attention_mask.sum(dim=1) - 1
            indices = torch.clamp(indices, min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            embeddings = hidden_states[batch_indices, indices]

        embeddings = embeddings.float().cpu().numpy()
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings


class Llama3Model(CausalLMModel):
    KEY = 'meta-llama/Meta-Llama-3-8B'


class Qwen2Model(CausalLMModel):
    KEY = 'Qwen/Qwen2-7B-Instruct'
