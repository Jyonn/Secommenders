import sys

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


def load_processor(dataset, data_dir=None):
    processors = ClassHub.processors()
    key = dataset.lower()
    if key not in processors:
        available = ', '.join(sorted(processors.class_dict))
        raise ValueError(f'Unknown processor: {dataset}. Available: {available}')
    processor = processors[key]
    return processor(data_dir=data_dir)


def load_embedder(model, **kwargs):
    embedders = ClassHub.embedders()
    key = model.lower()
    if key not in embedders:
        available = ', '.join(sorted(embedders.class_dict))
        raise ValueError(f'Unknown embedder: {model}. Available: {available}')
    embedder = embedders[key]
    return embedder(**kwargs)
