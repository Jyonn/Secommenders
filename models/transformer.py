import re
from typing import Optional

from models.base import BaseBackbone


class ScratchTokenizer:
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    HISTORY_TOKEN = '<history>'
    SEP_TOKEN = '<sep>'
    NEXT_TOKEN = '<next>'
    ALIGN_TOKEN = '<align>'
    TO_TOKEN = '<to>'

    def __init__(self, texts: list[str]):
        base_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.HISTORY_TOKEN,
            self.SEP_TOKEN,
            self.NEXT_TOKEN,
            self.ALIGN_TOKEN,
            self.TO_TOKEN,
        ]
        vocab = {}
        tokens = []
        for token in base_tokens:
            vocab[token] = len(tokens)
            tokens.append(token)

        for text in texts:
            for token in self.tokenize(text):
                if token not in vocab:
                    vocab[token] = len(tokens)
                    tokens.append(token)

        self.vocab = vocab
        self.tokens = tokens

    @classmethod
    def tokenize(cls, text: str):
        text = (text or '').lower().strip()
        if not text:
            return []
        return cls.TOKEN_PATTERN.findall(text)

    def encode(self, text: str, max_tokens: Optional[int] = None):
        tokens = [self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in self.tokenize(text)]
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        return tokens


class ScratchTransformerBackbone(BaseBackbone):
    def __init__(self, model_name: str, texts: list[str], max_length_override: int | None = None):
        super().__init__(model_name, max_length_override=max_length_override)
        self.tokenizer = ScratchTokenizer(texts)

    @property
    def kind(self):
        return 'scratch'

    @property
    def native_max_length(self):
        return self.DEFAULT_MAX_LENGTH

    def tokenize_texts(self, texts: list[str], max_tokens: int):
        return [self.tokenizer.encode(text or '[Empty Content]', max_tokens=max_tokens) for text in texts]

    def build_vocab_artifact(self):
        return {
            'tokens': self.tokenizer.tokens,
            'pad_token_id': self.tokenizer.vocab[ScratchTokenizer.PAD_TOKEN],
            'unk_token_id': self.tokenizer.vocab[ScratchTokenizer.UNK_TOKEN],
        }

    def build_prompt_spec(self):
        if self.prompt_spec is not None:
            return self.prompt_spec
        self.prompt_spec = {
            'history_prefix_ids': [self.tokenizer.vocab[ScratchTokenizer.HISTORY_TOKEN]],
            'item_separator_ids': [self.tokenizer.vocab[ScratchTokenizer.SEP_TOKEN]],
            'query_prefix_ids': [self.tokenizer.vocab[ScratchTokenizer.NEXT_TOKEN]],
            'max_length': self.max_length,
            'kind': self.kind,
        }
        return self.prompt_spec

    def build_alignment_spec(self):
        return {
            'align_prefix_ids': [self.tokenizer.vocab[ScratchTokenizer.ALIGN_TOKEN]],
            'align_bridge_ids': [self.tokenizer.vocab[ScratchTokenizer.TO_TOKEN]],
            'kind': self.kind,
        }

    def estimate_main_length(self, history_values: list, target_value, task_type: str):
        prompt = self.build_prompt_spec()
        history_len = sum(len(value) if isinstance(value, list) else 1 for value in history_values)
        separator_len = len(prompt['item_separator_ids']) * max(0, len(history_values) - 1)
        if task_type == 'embedding':
            target_len = 0
        elif isinstance(target_value, list):
            target_len = len(target_value)
        else:
            target_len = 1
        return len(prompt['history_prefix_ids']) + history_len + separator_len + len(prompt['query_prefix_ids']) + target_len
