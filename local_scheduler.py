from pathlib import Path

from utils import Schedule


def build_simple_schedule():
    return (
        Schedule(
            name='simple',
            effective_batch_size=64,
        )
        .main_metric('ndcg@10')
        .grid(
            'simple',
            datasets=['mind'],
            models=['scratch'],
            targets=['uid'],
            histories=['uid'],
        )
    ).export(Path('config/simple_scheduler.yaml'))


build_simple_schedule()
