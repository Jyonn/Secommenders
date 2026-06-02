import pigmento
from pigmento import pnt
import sys
from pathlib import Path


_LOGGING_READY = False
_RUN_LOG_PATH = None
_RUN_LOG_FILE = None


class _TeeStream:
    def __init__(self, original, file_handle):
        self.original = original
        self.file_handle = file_handle

    def write(self, data):
        self.original.write(data)
        self.file_handle.write(data)
        return len(data)

    def flush(self):
        self.original.flush()
        self.file_handle.flush()

    def isatty(self):
        return self.original.isatty()

    @property
    def encoding(self):
        return getattr(self.original, 'encoding', 'utf-8')


def setup_logging():
    global _LOGGING_READY
    if _LOGGING_READY:
        return

    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False,
    )
    _LOGGING_READY = True


def attach_run_log(path: str | Path):
    global _RUN_LOG_PATH, _RUN_LOG_FILE
    log_path = str(Path(path))
    if _RUN_LOG_PATH == log_path:
        return
    if _RUN_LOG_FILE is not None:
        _RUN_LOG_FILE.close()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _RUN_LOG_FILE = open(log_path, 'a', encoding='utf-8')
    sys.stdout = _TeeStream(sys.stdout, _RUN_LOG_FILE)
    sys.stderr = _TeeStream(sys.stderr, _RUN_LOG_FILE)
    _RUN_LOG_PATH = log_path
