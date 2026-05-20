from abc import ABC, abstractmethod


class BaseBackbone(ABC):
    DEFAULT_MAX_LENGTH = 512
    tokenizer = None

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_spec = None

    @property
    def namespace_name(self):
        return self.model_name

    @property
    @abstractmethod
    def max_length(self):
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

    @abstractmethod
    def build_alignment_spec(self):
        raise NotImplementedError
