from utils.data import get_data_dir


def load_formatter(*args, **kwargs):
    from utils.function import load_formatter as _load_formatter
    return _load_formatter(*args, **kwargs)


def load_processor(*args, **kwargs):
    from utils.function import load_processor as _load_processor
    return _load_processor(*args, **kwargs)


def load_embedder(*args, **kwargs):
    from utils.function import load_embedder as _load_embedder
    return _load_embedder(*args, **kwargs)

__all__ = ['get_data_dir', 'load_embedder', 'load_formatter', 'load_processor']
