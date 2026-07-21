import inspect

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
            lora_layers: str | None,
            lora_target_modules: str,
            model_dtype: str,
    ):
        super().__init__()
        torch_dtype = function.resolve_torch_dtype(model_dtype)
        if torch_dtype is None and str(model_dtype).lower() == 'auto':
            torch_dtype = torch.bfloat16
        base_model = AutoModel.from_pretrained(
            model_key,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        hidden_size = getattr(base_model.config, 'word_embed_proj_dim', None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, 'd_model', None)
        if hidden_size is None:
            text_config = getattr(base_model.config, 'text_config', None)
            if text_config is not None:
                hidden_size = getattr(text_config, 'word_embed_proj_dim', None)
                if hidden_size is None:
                    hidden_size = getattr(text_config, 'hidden_size', None)
                if hidden_size is None:
                    hidden_size = getattr(text_config, 'd_model', None)
        if hidden_size is None:
            raise ValueError(f'Cannot resolve hidden size from model config for {model_key}')
        self.hidden_size = int(hidden_size)
        self.input_embed_dim = int(base_model.get_input_embeddings().embedding_dim)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.model_dtype = torch_dtype or next(base_model.parameters()).dtype
        self.compute_dtype = self.model_dtype
        self.total_hidden_layers = function.resolve_hidden_layer_count(base_model.config)
        self.forward_accepts_use_cache = False

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
            if lora_layers and self.total_hidden_layers is None:
                raise ValueError(f'Cannot resolve hidden layer count for model {model_key}; unable to apply LoRA layer selection {lora_layers}')
            selected_layers = function.parse_layer_selection(lora_layers, self.total_hidden_layers) if lora_layers else None
            if selected_layers is not None:
                kept_trainable = 0
                kept_layers = set()
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    layer_index = function.extract_layer_index(name)
                    if layer_index is None or layer_index not in selected_layers:
                        param.requires_grad = False
                        continue
                    kept_trainable += param.numel()
                    kept_layers.add(layer_index)
                if not kept_layers:
                    raise ValueError(
                        f'LoRA layer selection {lora_layers} did not match any trainable adapter parameters '
                        f'for model {model_key}'
                    )
                pnt(
                    f'filtered LoRA trainable layers={sorted(kept_layers)} '
                    f'from spec={lora_layers} kept_params={kept_trainable:,}'
                )
            trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
            total_params = sum(param.numel() for param in self.model.parameters())
            pnt(
                f'initialized LoRA for {model_key} '
                f'trainable={trainable_params:,}/{total_params:,} '
                f'r={lora_rank} alpha={lora_alpha} dropout={lora_dropout:g} '
                f'targets={lora_target_modules} layers={lora_layers or "all"} '
                f'dtype={function.format_torch_dtype(self.model_dtype)}'
            )
        else:
            self.model = base_model
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
            pnt(f'loaded backbone {model_key} with dtype={function.format_torch_dtype(self.model_dtype)}')
        self.forward_accepts_use_cache = self._accepts_use_cache(self.model)

    @staticmethod
    def _accepts_use_cache(model) -> bool:
        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return 'use_cache' in signature.parameters

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and not self.use_lora:
            self.model.eval()
        return self

    def embed_model_tokens(self, token_ids: torch.Tensor):
        return self.model.get_input_embeddings()(token_ids)

    def get_input_embedding_weight(self):
        return self.model.get_input_embeddings().weight

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor | None = None):
        use_no_grad = self.freeze_backbone and not self.use_lora
        model_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        if self.forward_accepts_use_cache:
            model_kwargs['use_cache'] = False
        with torch.set_grad_enabled(not use_no_grad):
            outputs = self.model(**model_kwargs)
        return outputs.last_hidden_state

    @staticmethod
    def cache_seq_length(past_key_values):
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, 'get_seq_length'):
            return int(past_key_values.get_seq_length())
        if isinstance(past_key_values, (list, tuple)) and past_key_values:
            first_layer = past_key_values[0]
            if isinstance(first_layer, (list, tuple)) and first_layer:
                key_tensor = first_layer[0]
            elif isinstance(first_layer, dict):
                key_tensor = first_layer.get('key')
                if key_tensor is None:
                    key_tensor = first_layer.get('k')
            else:
                key_tensor = first_layer
            if torch.is_tensor(key_tensor):
                return int(key_tensor.shape[-2])
        return None

    @staticmethod
    def reorder_cache(past_key_values, beam_indices: torch.Tensor):
        """Select and duplicate legacy KV-cache rows for the next beam batch."""
        if past_key_values is None or not isinstance(past_key_values, (list, tuple)):
            return None
        reordered_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (list, tuple)):
                return None
            reordered_entries = []
            for entry in layer:
                if torch.is_tensor(entry):
                    if entry.ndim == 0:
                        reordered_entries.append(entry)
                    else:
                        reordered_entries.append(entry.index_select(0, beam_indices.to(entry.device)))
                else:
                    reordered_entries.append(entry)
            reordered_layers.append(type(layer)(reordered_entries))
        return type(past_key_values)(reordered_layers)

    def forward_with_cache(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            past_key_values=None,
    ):
        if not self.forward_accepts_use_cache:
            return self.forward(inputs_embeds, attention_mask, position_ids=position_ids), None
        use_no_grad = self.freeze_backbone and not self.use_lora
        model_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        with torch.set_grad_enabled(not use_no_grad):
            outputs = self.model(**model_kwargs)
        past_key_values = getattr(outputs, 'past_key_values', None)
        raw_cache_type = type(past_key_values).__name__ if past_key_values is not None else 'None'
        converted_to_legacy = hasattr(past_key_values, 'to_legacy_cache')
        if hasattr(past_key_values, 'to_legacy_cache'):
            past_key_values = past_key_values.to_legacy_cache()
        self.last_cache_diagnostic = (
            f'raw={raw_cache_type} converted_to_legacy={converted_to_legacy} '
            f'returned={type(past_key_values).__name__ if past_key_values is not None else "None"}'
        )
        return outputs.last_hidden_state, past_key_values


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

    def get_input_embedding_weight(self):
        return self.token_embedding.weight

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
