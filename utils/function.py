import sys
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def coerce_bool(value: str, default: bool):
    if value == 'auto':
        return default
    return value == 'true'


def build_dataloaders(train_dataset, test_dataset, batch_size: int):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch)
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
