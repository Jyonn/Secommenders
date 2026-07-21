import torch
from torch import nn

from core.encoders import LLMSequenceEncoder
from core.model import SequentialRecModel


class BeamSearchHarness(SequentialRecModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = type('Config', (), {'code_beam_width': 2})()
        self.compiled = type(
            'Compiled',
            (),
            {
                'sid_num_quantizers': 3,
                'sid_prefix_to_next': {
                    (): [0, 1],
                    (0,): [2, 3],
                    (1,): [2, 3],
                    (0, 2): [0, 1],
                    (0, 3): [0, 1],
                    (1, 2): [0, 1],
                    (1, 3): [0, 1],
                },
            },
        )()
        self.encoder = type('CacheOps', (), {'reorder_cache': staticmethod(LLMSequenceEncoder.reorder_cache)})()
        self.append_batch_sizes = []
        self.append_position_ids = []

    @property
    def device(self):
        return torch.device('cpu')

    def _sid_kv_cache_supported(self):
        return True

    def _sid_base_logits_and_cache(self, sample):
        logits = torch.tensor([4.0, 3.0, 0.0, 0.0])
        cache = ((torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1)),)
        return logits, cache

    def _sid_base_batch_logits_and_cache(self, batch):
        batch_size = len(batch)
        logits = torch.tensor([[4.0, 3.0, 0.0, 0.0]]).repeat(batch_size, 1)
        cache = ((
            torch.zeros(batch_size, 1, 3, 1),
            torch.zeros(batch_size, 1, 3, 1),
        ),)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)[:batch_size]
        lengths = torch.tensor([3, 2], dtype=torch.long)[:batch_size]
        return logits, cache, attention_mask, lengths

    def _sid_append_cached_tokens(self, past_key_values, codes, attention_mask=None, position_ids=None):
        self.append_batch_sizes.append(len(codes))
        if position_ids is not None:
            self.append_position_ids.append(position_ids.squeeze(1).tolist())
        logits = []
        for code in codes:
            if code in {0, 1}:
                logits.append([0.0, 0.0, 4.0, 3.0])
            else:
                logits.append([4.0, 3.0, 0.0, 0.0])
        batch_size = len(codes)
        next_length = past_key_values[0][0].shape[-2] + 1
        next_cache = ((
            torch.zeros(batch_size, 1, next_length, 1),
            torch.zeros(batch_size, 1, next_length, 1),
        ),)
        return torch.tensor(logits), next_cache


def test_reorder_cache_selects_and_duplicates_beam_rows():
    key = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    value = key + 1_000
    cache = ((key, value),)
    indices = torch.tensor([1, 0, 1], dtype=torch.long)

    reordered = LLMSequenceEncoder.reorder_cache(cache, indices)

    assert reordered[0][0].shape == (3, 3, 4, 5)
    assert torch.equal(reordered[0][0][0], key[1])
    assert torch.equal(reordered[0][0][1], key[0])
    assert torch.equal(reordered[0][0][2], key[1])
    assert torch.equal(reordered[0][1][2], value[1])


def test_reorder_cache_preserves_tuple_structure_and_scalar_entries():
    key = torch.zeros(1, 2, 3, 4)
    scalar = torch.tensor(7)
    cache = ((key, scalar, 'metadata'),)

    reordered = LLMSequenceEncoder.reorder_cache(cache, torch.tensor([0, 0]))

    assert isinstance(reordered, tuple)
    assert isinstance(reordered[0], tuple)
    assert reordered[0][0].shape[0] == 2
    assert reordered[0][1].item() == 7
    assert reordered[0][2] == 'metadata'


def test_reorder_cache_rejects_unknown_cache_container():
    assert LLMSequenceEncoder.reorder_cache(object(), torch.tensor([0])) is None


def test_kv_beam_search_batches_all_active_beams_once_per_sid_slot():
    model = BeamSearchHarness()

    beams = model._beam_search_sid_items_with_kv_cache(sample={})

    assert len(beams) == 2
    assert all(len(prefix) == 3 for prefix, _ in beams)
    assert model.append_batch_sizes == [2, 2]


def test_batch_kv_search_batches_histories_and_respects_unpadded_positions():
    model = BeamSearchHarness()

    beams_by_sample = model._beam_search_sid_items_batch_with_kv_cache([{}, {}])

    assert [len(beams) for beams in beams_by_sample] == [2, 2]
    assert model.append_batch_sizes == [4, 4]
    assert model.append_position_ids[0] == [3, 3, 2, 2]
    assert model.append_position_ids[1] == [4, 4, 3, 3]
