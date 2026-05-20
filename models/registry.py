from models.base import BaseBackbone
from models.llm import LLMBackbone
from models.transformer import ScratchTransformerBackbone
from utils import model as model_utils


def build_backbone(model_name: str, item_texts: list[str]) -> BaseBackbone:
    model_key = model_utils.match(model_name)
    if model_key:
        return LLMBackbone(model_name, model_key)
    return ScratchTransformerBackbone(model_name, item_texts)
