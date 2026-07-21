import torch

from core.encoders import LLMSequenceEncoder, ScratchLlamaSequenceEncoder
from models import build_backbone


def test_scratch_names_select_new_and_legacy_architectures():
    assert build_backbone('scratch', []).kind == 'scratch'
    assert build_backbone('scratchlegacy', []).kind == 'scratchlegacy'


def test_scratch_llama_supports_configurable_depth_and_kv_cache():
    encoder = ScratchLlamaSequenceEncoder(
        vocab_size=5,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_length=64,
    )
    inputs_embeds = torch.randn(2, 5, 32)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    hidden, cache = encoder.forward_with_cache(inputs_embeds, attention_mask)
    selected_cache = LLMSequenceEncoder.reorder_cache(cache, torch.tensor([1, 0, 1]))

    assert len(encoder.model.layers) == 2
    assert hidden.shape == (2, 5, 32)
    assert encoder.cache_seq_length(selected_cache) == 5
    assert selected_cache.layers[0].keys.shape[0] == 3
