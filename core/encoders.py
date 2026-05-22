from peft import LoraConfig, TaskType, get_peft_model
from pigmento import pnt
import torch
from torch import nn
from transformers import AutoModel

from utils import function


class LLMSequenceEncoder(nn.Module):
    def __init__(
            self,
            model_key: str,
            freeze_backbone: bool,
            use_lora: bool,
            lora_rank: int,
            lora_alpha: int,
            lora_dropout: float,
            lora_target_modules: str,
            model_dtype: str,
    ):
        super().__init__()
        torch_dtype = function.resolve_torch_dtype(model_dtype)
        base_model = AutoModel.from_pretrained(
            model_key,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        hidden_size = getattr(base_model.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, 'd_model', None)
        if hidden_size is None:
            raise ValueError(f'Cannot resolve hidden size from model config for {model_key}')
        self.hidden_size = int(hidden_size)
        self.input_embed_dim = int(base_model.get_input_embeddings().embedding_dim)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.model_dtype = torch_dtype or next(base_model.parameters()).dtype
        self.compute_dtype = self.model_dtype

        if self.use_lora:
            target_modules = 'all-linear' if lora_target_modules == 'all-linear' else [
                value.strip() for value in lora_target_modules.split(',') if value.strip()
            ]
            self.model = get_peft_model(
                base_model,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                ),
            )
            trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
            total_params = sum(param.numel() for param in self.model.parameters())
            pnt(
                f'initialized LoRA for {model_key} '
                f'trainable={trainable_params:,}/{total_params:,} '
                f'r={lora_rank} alpha={lora_alpha} dropout={lora_dropout:g} '
                f'targets={lora_target_modules} dtype={function.format_torch_dtype(self.model_dtype)}'
            )
        else:
            self.model = base_model
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
            pnt(f'loaded backbone {model_key} with dtype={function.format_torch_dtype(self.model_dtype)}')

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and not self.use_lora:
            self.model.eval()
        return self

    def embed_model_tokens(self, token_ids: torch.Tensor):
        return self.model.get_input_embeddings()(token_ids)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor | None = None):
        use_no_grad = self.freeze_backbone and not self.use_lora
        with torch.set_grad_enabled(not use_no_grad):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
        return outputs.last_hidden_state


class ScratchSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float, max_length: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_embed_dim = hidden_size
        self.compute_dtype = torch.float32
        self.num_heads = num_heads
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def embed_model_tokens(self, token_ids: torch.Tensor):
        return self.token_embedding(token_ids)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor | None = None):
        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        else:
            positions = position_ids
        hidden = inputs_embeds + self.position_embedding(positions)
        src_key_padding_mask = None
        if attention_mask.dim() == 4:
            causal_mask = attention_mask.squeeze(1).repeat_interleave(self.num_heads, dim=0)
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=torch.bool),
                diagonal=1,
            )
            src_key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return hidden
