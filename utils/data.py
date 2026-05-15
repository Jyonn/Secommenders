from pathlib import Path


def _load_data_dirs():
    path = Path('.data')
    if not path.exists():
        return {}

    data_dirs = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        name, value = line.split('=', 1)
        data_dirs[name.strip()] = value.strip()
    return data_dirs


DATA_DIRS = _load_data_dirs()


def get_data_dir(dataset):
    return DATA_DIRS.get(dataset)
