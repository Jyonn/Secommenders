from transformers import AutoConfig, AutoTokenizer

from models.base import BaseBackbone


class LLMBackbone(BaseBackbone):
    HISTORY_PREFIX = 'A user has browsed the following items:'
    ITEM_SEPARATOR = ','
    QUERY_PREFIX = 'Which item would the user probably interact with:'

    ALIGN_PREFIX = 'An item featured'
    ALIGN_BRIDGE = 'can be mapped to'

    def __init__(self, model_name: str, model_key: str, max_length_override: int | None = None):
        super().__init__(model_name, max_length_override=max_length_override)
        self.model_key = model_key
        self.tokenizer = AutoTokenizer.from_pretrained(model_key, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = AutoConfig.from_pretrained(model_key, trust_remote_code=True)
        self._native_max_length = self._resolve_max_length()

    @property
    def kind(self):
        return 'llm'

    @property
    def native_max_length(self):
        return self._native_max_length

    def _resolve_max_length(self):
        tokenizer_max = getattr(self.tokenizer, 'model_max_length', None)
        if tokenizer_max and tokenizer_max < 1_000_000:
            return int(tokenizer_max)

        for attr in ['max_position_embeddings', 'n_positions', 'seq_length', 'max_seq_len', 'model_max_length']:
            value = getattr(self.config, attr, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                return int(value)
        return self.DEFAULT_MAX_LENGTH

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_texts(self, texts: list[str], max_tokens: int):
        return [
            self.tokenizer.encode(
                text or '[Empty Content]',
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
            for text in texts
        ]

    def build_vocab_artifact(self):
        vocab = self.tokenizer.get_vocab()
        size = max(vocab.values()) + 1 if vocab else 0
        tokens = [''] * size
        for token, index in vocab.items():
            tokens[index] = token
        return {
            'tokens': tokens,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'eos_token_id': getattr(self.tokenizer, 'eos_token_id', None),
            'pad_token_id': getattr(self.tokenizer, 'pad_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
            'model_key': self.model_key,
        }

    def build_prompt_spec(self):
        if self.prompt_spec is not None:
            return self.prompt_spec
        self.prompt_spec = {
            'history_prefix_ids': self._encode(self.HISTORY_PREFIX),
            'item_separator_ids': self._encode(self.ITEM_SEPARATOR),
            'query_prefix_ids': self._encode(self.QUERY_PREFIX),
            'max_length': self.max_length,
            'kind': self.kind,
        }
        return self.prompt_spec

    def build_alignment_spec(self):
        return {
            'align_prefix_ids': self._encode(self.ALIGN_PREFIX),
            'align_bridge_ids': self._encode(self.ALIGN_BRIDGE),
            'kind': self.kind,
        }

    @staticmethod
    def _value_length(value, task_type):
        if task_type == 'embedding':
            return 0
        if isinstance(value, list):
            return len(value)
        return 1

    def estimate_main_length(self, history_values: list, target_value, task_type: str):
        prompt = self.build_prompt_spec()
        history_len = sum(len(value) if isinstance(value, list) else 1 for value in history_values)
        separator_len = len(prompt['item_separator_ids']) * max(0, len(history_values) - 1)
        target_len = self._value_length(target_value, task_type)
        return (
            len(prompt['history_prefix_ids'])
            + history_len
            + separator_len
            + len(prompt['query_prefix_ids'])
            + target_len
        )
