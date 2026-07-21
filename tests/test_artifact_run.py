import threading
import time

from utils.artifact_run import ArtifactRunCoordinator


def test_only_one_producer_runs_and_waiter_reuses_result(tmp_path):
    ready_path = tmp_path / 'ready'
    producers = []
    reports = []
    results = []

    def execute(name):
        coordinator = ArtifactRunCoordinator(
            tmp_path,
            kind='test',
            identity='shared/artifact',
            wait_seconds=0.02,
            reporter=reports.append,
        )
        owner = coordinator.acquire_or_wait(ready_path.exists)
        results.append((name, owner))
        if not owner:
            return
        producers.append(name)
        for index in range(1, 4):
            coordinator.update(stage='working', current=index, total=3, message=f'step {index}')
            time.sleep(0.03)
        ready_path.write_text('ready')
        coordinator.finish()

    first = threading.Thread(target=execute, args=('first',))
    second = threading.Thread(target=execute, args=('second',))
    first.start()
    time.sleep(0.01)
    second.start()
    first.join()
    second.join()

    assert len(producers) == 1
    assert sorted(owner for _, owner in results) == [False, True]
    assert any('waiting for test artifact' in report for report in reports)
    state = (tmp_path / 'run_state.json').read_text()
    assert '"status": "completed"' in state
    assert not (tmp_path / '.run.lock').exists()


def test_failed_producer_releases_lock(tmp_path):
    coordinator = ArtifactRunCoordinator(
        tmp_path,
        kind='test',
        identity='failed/artifact',
        wait_seconds=0.01,
        reporter=lambda _: None,
    )
    assert coordinator.acquire_or_wait(lambda: False)
    coordinator.fail(RuntimeError('boom'))

    assert not (tmp_path / '.run.lock').exists()
    assert '"status": "failed"' in (tmp_path / 'run_state.json').read_text()


def test_force_producer_runs_even_when_artifact_is_ready(tmp_path):
    ready_path = tmp_path / 'ready'
    ready_path.write_text('old')
    coordinator = ArtifactRunCoordinator(
        tmp_path,
        kind='test',
        identity='forced/artifact',
        reporter=lambda _: None,
    )

    assert coordinator.acquire_or_wait(ready_path.exists, force_producer=True)
    coordinator.finish()


def test_waiter_observes_active_overwrite_even_when_old_artifact_is_ready(tmp_path):
    ready_path = tmp_path / 'ready'
    ready_path.write_text('old')
    owner = ArtifactRunCoordinator(
        tmp_path,
        kind='test',
        identity='overwrite/artifact',
        wait_seconds=0.01,
        reporter=lambda _: None,
    )
    assert owner.acquire_or_wait(ready_path.exists, force_producer=True)

    waited = []

    def consume():
        waiter = ArtifactRunCoordinator(
            tmp_path,
            kind='test',
            identity='overwrite/artifact',
            wait_seconds=0.01,
            reporter=lambda _: None,
        )
        started = time.monotonic()
        assert not waiter.acquire_or_wait(ready_path.exists)
        waited.append(time.monotonic() - started)

    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(0.05)
    owner.finish()
    thread.join()

    assert waited[0] >= 0.04
