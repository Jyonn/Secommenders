from abc import ABC, abstractmethod

TYPE_MARKER_TOKENS = {
    'uid': '<uid>',
    'sid': '<sid>',
    'hash': '<hash>',
    'text': '<text>',
    'embedding': '<embedding>',
    'uid+embedding': '<uid+embedding>',
}
TYPE_MARKER_ORDER = list(TYPE_MARKER_TOKENS)


class BaseBackbone(ABC):
    DEFAULT_MAX_LENGTH = 512
    tokenizer = None

    def __init__(self, model_name: str, max_length_override: int | None = None):
        self.model_name = model_name
        self.max_length_override = max_length_override if max_length_override and max_length_override > 0 else None
        self.prompt_spec = None

    @property
    def namespace_name(self):
        return self.model_name

    @property
    def max_length(self):
        return self.max_length_override or self.native_max_length

    @property
    @abstractmethod
    def native_max_length(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def kind(self):
        raise NotImplementedError

    @abstractmethod
    def build_vocab_artifact(self):
        raise NotImplementedError

    @abstractmethod
    def build_prompt_spec(self):
        raise NotImplementedError

    @abstractmethod
    def tokenize_texts(self, texts: list[str], max_tokens: int):
        raise NotImplementedError

    @abstractmethod
    def estimate_main_length(self, history_values: list, target_value, task_type: str):
        raise NotImplementedError
