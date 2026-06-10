from pathlib import Path

from utils import Schedule


DATASETS = ['mind', 'recifvideo', 'recifvideolarge', 'recifvideoxlarge']
LLM_MODELS = ['llama3', 'qwen35th9b', 'qwen35th4b', 'qwen35th08b']
SCRATCH_MODELS = ['scratch']
SOURCE_MODELS = ['llama3', 'qwen3embedding06b']
LLM_HISTORIES = ['uid', 'text', 'sid', 'embedding']
SCRATCH_HISTORIES = ['uid', 'sid', 'embedding']
SID_VARIANTS = [('rqvae', 'recon')]
UID_VARIANTS = ['flat', ('hierarchical', '20', '3,20')]


def build_basic_schedule(dataset: str):
    dataset = str(dataset).strip().lower()
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    return (
        Schedule(
            name=f'basic_{dataset}',
            effective_batch_size=64,
        )
        .main_metric('ndcg@10')
        .source_models(*SOURCE_MODELS)
        .sid_variants(*SID_VARIANTS)
        .uid_variants(*UID_VARIANTS)
        .grid(
            'basic_llm',
            datasets=[dataset],
            models=LLM_MODELS,
            targets=['uid', 'sid'],
            histories=LLM_HISTORIES,
        )
        .grid(
            'basic_scratch',
            datasets=[dataset],
            models=SCRATCH_MODELS,
            targets=['uid', 'sid'],
            histories=SCRATCH_HISTORIES,
        )
    ).export(Path(f'config/basic_{dataset}_scheduler.yaml'))


def build_basic_schedules():
    return [build_basic_schedule(dataset) for dataset in DATASETS]


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


if __name__ == '__main__':
    build_simple_schedule()
