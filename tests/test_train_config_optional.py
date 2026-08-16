from oba import Obj

from core.train_config import _get, _plain_section
from utils.word2vec import WORD2VEC_DEFAULTS


def test_missing_obj_attribute_uses_default():
    section = Obj({})

    assert _get(section, 'missing', None) is None
    assert _get(section, 'missing', 'fallback') == 'fallback'


def test_plain_section_omits_oba_not_found_values():
    section = Obj({'vector_size': 32})

    assert _plain_section(section, WORD2VEC_DEFAULTS) == {'vector_size': 32}
