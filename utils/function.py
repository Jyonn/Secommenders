import sys

from processors.base_processor import Processor
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
    formatter = load_formatter(dataset, data_dir=data_dir)
    return Processor(formatter=formatter)


def load_embedder(model, **kwargs):
    embedders = ClassHub.embedders()
    key = model.lower()
    if key not in embedders:
        available = ', '.join(sorted(embedders.class_dict))
        raise ValueError(f'Unknown embedder: {model}. Available: {available}')
    embedder = embedders[key]
    return embedder(**kwargs)
