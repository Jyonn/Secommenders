from typing import Optional

from models.base import BaseBackbone
from models.llm import LLMBackbone
from models.transformer import ScratchTransformerBackbone
from utils import model as model_utils


def build_backbone(model_name: str, item_texts: list[str], max_length_override: Optional[int] = None) -> BaseBackbone:
    normalized_name = str(model_name).strip().lower()
    model_key = model_utils.match(normalized_name)
    if model_key:
        return LLMBackbone(normalized_name, model_key, max_length_override=max_length_override)
    if normalized_name == 'scratch':
        return ScratchTransformerBackbone(normalized_name, item_texts, max_length_override=max_length_override)
    raise ValueError(
        f'Unknown model "{model_name}". '
        f'Use "scratch" for the scratch backbone or configure a mapped model name in .model'
    )
