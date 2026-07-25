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
    path = DATA_DIRS.get(dataset)
    if path is not None:
        return path

    if dataset.startswith('recif') and 'recif' in DATA_DIRS:
        return DATA_DIRS['recif']
    if dataset[:2] in {'rv', 'ra'} and dataset[2:].isdigit() and 'recif' in DATA_DIRS:
        return DATA_DIRS['recif']
    if dataset[:3] in {'rvs', 'rvt', 'ras'} and dataset[3:].isdigit() and 'recif' in DATA_DIRS:
        return DATA_DIRS['recif']
    if dataset.startswith('minds') and dataset[5:].isdigit() and 'mind' in DATA_DIRS:
        return DATA_DIRS['mind']
    return None
