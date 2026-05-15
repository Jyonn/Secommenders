from pathlib import Path


def _load_models():
    path = Path('.model')
    if not path.exists():
        return {}

    model_map = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        name, value = line.split('=', 1)
        name = name.strip().lower()
        keys = value.strip().split(',')
        keys = [item.split(':', 1) for item in keys]
        keys = {key.strip(): mapped.strip() for key, mapped in keys}
        for key, mapped in keys.items():
            model_map[f'{name}{key}'] = mapped

    return model_map


MODEL_MAP = _load_models()


def match(key):
    return MODEL_MAP.get(key)
