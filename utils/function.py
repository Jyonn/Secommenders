import sys
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.class_hub import ClassHub


def argparse():
    arguments = sys.argv[1:]
    kwargs = {}

    key = None
    for arg in arguments:
        if key is not None:
            kwargs[key] = arg
            key = None
        else:
            assert arg.startswith('--')
            key = arg[2:]

    for key, value in kwargs.items():
        if value == 'null':
            kwargs[key] = None
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            kwargs[key] = int(value)
        elif value.lower() == 'true':
            kwargs[key] = True
        elif value.lower() == 'false':
            kwargs[key] = False
        else:
            try:
                kwargs[key] = float(value)
            except ValueError:
                pass
    return kwargs


def load_formatter(dataset, data_dir=None):
    formatters = ClassHub.formatters()
    key = dataset.lower()
    if key not in formatters:
        available = ', '.join(sorted(formatters.class_dict))
        raise ValueError(f'Unknown formatter: {dataset}. Available: {available}')
    formatter = formatters[key]
    return formatter(data_dir=data_dir)


def load_processor(dataset, data_dir=None):
    from processors.base_processor import Processor
    return Processor(dataset=dataset)


def load_embedder(model, **kwargs):
    embedders = ClassHub.embedders()
    key = model.lower()
    if key not in embedders:
        compact_key = key.replace('-', '').replace('_', '')
        if compact_key in embedders:
            key = compact_key
    if key not in embedders:
        available = ', '.join(sorted(embedders.class_dict))
        raise ValueError(f'Unknown embedder: {model}. Available: {available}')
    embedder = embedders[key]
    return embedder(**kwargs)


def to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def coerce_bool(value: str, default: bool) -> bool:
    if value == 'auto':
        return default
    return value == 'true'


def build_dataloaders(
        train_dataset,
        test_dataset,
        batch_size: int,
        train_sampler: DistributedSampler | None = None,
        test_sampler: DistributedSampler | None = None,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=lambda batch: batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=lambda batch: batch,
    )
    return train_loader, test_loader


def resolve_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    key = str(dtype_name).lower()
    if key == 'auto':
        return None
    mapping = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f'Unsupported torch dtype: {dtype_name}')
    return mapping[key]


def format_torch_dtype(dtype):
    if dtype is None:
        return 'auto'
    if hasattr(dtype, 'name'):
        return dtype.name
    return str(dtype).replace('torch.', '')


def normalize_optional_string(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'null':
        return None
    return text


def normalize_lora_layers(value):
    text = normalize_optional_string(value)
    if text is None:
        return None
    return re.sub(r'\s+', '', text)


def resolve_hidden_layer_count(config):
    for attr in ('num_hidden_layers', 'n_layer', 'num_layers', 'n_layers'):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, 'text_config', None)
    if text_config is not None:
        for attr in ('num_hidden_layers', 'n_layer', 'num_layers', 'n_layers'):
            value = getattr(text_config, attr, None)
            if value is not None:
                return int(value)
    decoder = getattr(config, 'decoder', None)
    if decoder is not None:
        for attr in ('num_hidden_layers', 'n_layer', 'num_layers', 'n_layers'):
            value = getattr(decoder, attr, None)
            if value is not None:
                return int(value)
    return None


def parse_layer_selection(spec, total_layers: int):
    normalized = normalize_lora_layers(spec)
    if normalized is None:
        return None
    if total_layers <= 0:
        raise ValueError('total_layers must be positive when parsing layer selection')

    chunks = normalized
    if chunks.startswith('[') and chunks.endswith(']'):
        chunks = chunks[1:-1]
    items = [item for item in chunks.split(',') if item]
    if not items:
        raise ValueError(f'Invalid LoRA layer selection: {spec}')

    selected = set()
    for item in items:
        if ':' in item:
            start_text, end_text = item.split(':', 1)
            start = int(start_text) if start_text else None
            end = int(end_text) if end_text else None
            layer_range = range(total_layers)[slice(start, end)]
            selected.update(layer_range)
            continue
        index = int(item)
        if index < 0:
            index += total_layers
        if index < 0 or index >= total_layers:
            raise ValueError(f'Layer index out of range in LoRA layer selection: {item}')
        selected.add(index)

    if not selected:
        raise ValueError(f'LoRA layer selection {spec} resolved to no layers')
    return selected


_LAYER_INDEX_PATTERNS = [
    re.compile(r'(?:^|\.)layers\.(\d+)(?:\.|$)'),
    re.compile(r'(?:^|\.)h\.(\d+)(?:\.|$)'),
    re.compile(r'(?:^|\.)blocks?\.(\d+)(?:\.|$)'),
]


def extract_layer_index(parameter_name: str):
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.search(parameter_name)
        if match:
            return int(match.group(1))
    return None


def summarize_trainable_parameters(named_parameters):
    entries = []
    pattern = re.compile(r'\.(\d+)(?=\.|$)')
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        template = pattern.sub('.{i}', name)
        entries.append((name, template, tuple(param.shape)))

    groups = {}
    order = []
    for name, template, shape in entries:
        key = (template, shape)
        if key not in groups:
            groups[key] = {
                'count': 0,
                'example': name,
            }
            order.append(key)
        groups[key]['count'] += 1

    lines = []
    for template, shape in order:
        info = groups[(template, shape)]
        label = template
        if info['count'] > 1:
            label = f'{template} [x{info["count"]}]'
        lines.append((label, shape, info['count'], info['example']))
    return lines
