import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pigmento import pnt


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _pid_is_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ArtifactRunCoordinator:
    """Coordinate one artifact producer and any number of progress-watching consumers."""

    STATE_NAME = 'run_state.json'
    LOCK_NAME = '.run.lock'
    DEFAULT_WAIT_SECONDS = 5.0
    REMOTE_STALE_SECONDS = 60.0 * 60.0

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        kind: str,
        identity: str,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        reporter: Callable[[str], None] = pnt,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.kind = str(kind)
        self.identity = str(identity)
        self.wait_seconds = float(wait_seconds)
        self.reporter = reporter
        self.state_path = self.artifact_dir / self.STATE_NAME
        self.lock_path = self.artifact_dir / self.LOCK_NAME
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.owner = False
        self._state = {}
        self._state_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._last_write_monotonic = 0.0

    def _write_state(self):
        with self._state_lock:
            payload = dict(self._state)
            payload['updated_at'] = _utc_now()
            temporary = self.state_path.with_name(f'.{self.STATE_NAME}.{self.pid}.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(self.state_path)
            self._last_write_monotonic = time.monotonic()

    def _heartbeat(self):
        interval = max(1.0, min(self.wait_seconds, 5.0))
        while not self._heartbeat_stop.wait(interval):
            if not self.owner:
                return
            self._write_state()

    def _start_owner(self):
        self.owner = True
        self._state = {
            'kind': self.kind,
            'identity': self.identity,
            'status': 'running',
            'stage': 'starting',
            'current': 0,
            'total': None,
            'percent': None,
            'message': 'initializing artifact producer',
            'pid': self.pid,
            'hostname': self.hostname,
            'started_at': _utc_now(),
        }
        self._write_state()
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeat_thread.start()
        self.reporter(f'started {self.kind} artifact producer for {self.identity} pid={self.pid}')

    @staticmethod
    def _state_age_seconds(state):
        value = state.get('updated_at')
        if not value:
            return float('inf')
        try:
            updated = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            return float('inf')
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())

    def _owner_is_stale(self, state):
        try:
            lock_age = max(0.0, time.time() - self.lock_path.stat().st_mtime)
        except FileNotFoundError:
            return False
        if lock_age < max(30.0, self.wait_seconds * 2.0):
            return False
        if not state:
            return lock_age > self.REMOTE_STALE_SECONDS
        hostname = state.get('hostname')
        if hostname == self.hostname:
            return not _pid_is_alive(state.get('pid'))
        return self._state_age_seconds(state) > self.REMOTE_STALE_SECONDS

    def _remove_stale_lock(self, state):
        if not self._owner_is_stale(state):
            return False
        try:
            self.lock_path.rmdir()
        except (FileNotFoundError, OSError):
            return False
        self.reporter(
            f'recovered stale {self.kind} artifact lock for {self.identity}; '
            f'previous_pid={state.get("pid", "-")} previous_host={state.get("hostname", "-")}'
        )
        return True

    @staticmethod
    def _format_progress(state):
        stage = state.get('stage') or 'running'
        message = state.get('message') or '-'
        current = state.get('current')
        total = state.get('total')
        percent = state.get('percent')
        if percent is not None:
            progress = f'{float(percent):.1f}%'
        elif current is not None and total:
            progress = f'{current}/{total}'
        elif current is not None:
            progress = str(current)
        else:
            progress = '-'
        return f'stage={stage} progress={progress} message={message}'

    def acquire_or_wait(self, is_ready: Callable[[], bool], *, force_producer: bool = False):
        """Return True to the producer and False to a waiter once the artifact is ready."""
        while True:
            try:
                self.lock_path.mkdir()
            except FileExistsError:
                state = _read_json(self.state_path)
                if self._remove_stale_lock(state):
                    continue
                self.reporter(
                    f'waiting for {self.kind} artifact {self.identity}; '
                    f'{self._format_progress(state)} producer={state.get("hostname", "-")}:{state.get("pid", "-")}'
                )
                time.sleep(self.wait_seconds)
                continue
            if is_ready() and not force_producer:
                self.lock_path.rmdir()
                return False
            self._start_owner()
            return True

    def update(self, *, stage: str, current=None, total=None, message: str = ''):
        if not self.owner:
            return
        percent = None
        if current is not None and total:
            percent = max(0.0, min(100.0, 100.0 * float(current) / float(total)))
        with self._state_lock:
            stage_changed = self._state.get('stage') != stage
            self._state.update(
                {
                    'status': 'running',
                    'stage': stage,
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'message': message,
                }
            )
        if stage_changed or time.monotonic() - self._last_write_monotonic >= 1.0:
            self._write_state()

    def finish(self, *, message='artifact ready'):
        if not self.owner:
            return
        self._stop_heartbeat()
        with self._state_lock:
            self._state.update(
                {
                    'status': 'completed',
                    'stage': 'completed',
                    'percent': 100.0,
                    'message': message,
                    'finished_at': _utc_now(),
                }
            )
        self._write_state()
        self._release_lock()
        self.reporter(f'completed {self.kind} artifact for {self.identity}')

    def fail(self, error):
        if not self.owner:
            return
        self._stop_heartbeat()
        with self._state_lock:
            self._state.update(
                {
                    'status': 'failed',
                    'stage': 'failed',
                    'message': repr(error),
                    'failed_at': _utc_now(),
                }
            )
        self._write_state()
        self._release_lock()

    def _stop_heartbeat(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.wait_seconds + 1.0))
            self._heartbeat_thread = None

    def _release_lock(self):
        try:
            self.lock_path.rmdir()
        except FileNotFoundError:
            pass
        self.owner = False
