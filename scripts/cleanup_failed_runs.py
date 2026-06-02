import json
import os
import shutil
import socket
from pathlib import Path


ROOT = Path('artifacts/trained')


def pid_alive(pid: int):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def should_delete(run_dir: Path):
    pid_path = run_dir / 'pid.json'
    if not pid_path.exists():
        return False, 'missing pid record'

    pid_record = load_json(pid_path)
    if not pid_record or 'pid' not in pid_record:
        return False, 'invalid pid record'

    hostname = pid_record.get('hostname')
    current_host = socket.gethostname()
    if hostname and hostname != current_host:
        return False, f'pid belongs to another host ({hostname})'

    pid = int(pid_record['pid'])
    if pid_alive(pid):
        return False, f'pid {pid} still alive'

    meta_path = run_dir / 'meta.json'
    meta = load_json(meta_path) if meta_path.exists() else None
    has_test_metrics = isinstance(meta, dict) and 'test_metrics' in meta
    if has_test_metrics:
        return False, 'completed run with test_metrics'

    return True, f'pid {pid} dead and test_metrics missing'


def main():
    if not ROOT.exists():
        print(f'no trained artifacts under {ROOT}')
        return

    deleted = 0
    skipped = 0
    for dataset_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            delete, reason = should_delete(run_dir)
            if delete:
                shutil.rmtree(run_dir)
                deleted += 1
                print(f'DELETE {run_dir} :: {reason}')
            else:
                skipped += 1
                print(f'SKIP   {run_dir} :: {reason}')

    print(f'cleanup finished deleted={deleted} skipped={skipped}')


if __name__ == '__main__':
    main()
