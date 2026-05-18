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
        entries = []
        for item in value.strip().split(','):
            key, mapped = item.split(':', 1)
            key = key.strip()
            mapped = mapped.strip().split('$', 1)[0].strip()
            entries.append((key, mapped))
            model_map[f'{name}{key}'] = mapped

        if name not in model_map and entries:
            default = None
            for key, mapped in entries:
                if key in {'', 'base'}:
                    default = mapped
                    break
            if default is None:
                default = entries[0][1]
            model_map[name] = default

    return model_map


MODEL_MAP = _load_models()


def match(key):
    return MODEL_MAP.get(key)
