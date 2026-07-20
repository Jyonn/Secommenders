from pathlib import Path

from utils import Schedule


DATASETS = [
    'mind',
    'recifvideo',
    'recifvideolarge',
    'recifvideoxlarge',
    'recifvideoxlargeall',
    'recifadsall',
    'recifadslargeall',
    'recifadsxlargeall',
]
LLM_MODELS = ['llama3', 'qwen35th9b', 'qwen35th4b', 'qwen35th08b']
SCRATCH_MODELS = ['scratch']
SOURCE_MODELS = ['llama3', 'qwen3embedding06b']
LLM_HISTORIES = ['uid', 'text', 'sid', 'embedding']
SCRATCH_HISTORIES = ['uid', 'sid', 'embedding']
SID_VARIANTS = [('rqvae', 'recon')]
UID_VARIANTS = ['flat', ('hierarchical', '20', '3,20')]
RECIF_SCALE_PERCENTS = [20, 40]
RECIF_SCALE_SOURCE_MODEL = 'pretrain-multimodal'
RECIF_SCALE_SID_VARIANTS = [('rqvae', 'coll')]
RECIF_SCALE_REPRESENTATIONS = [
    ('uid', 'uid'),
    ('sid', 'sid'),
    # ('uid+embedding', ('uid', 'embedding')),
    # ('sid+embedding', ('sid', 'embedding')),
    # ('sid+text', ('sid', 'text')),
    # ('uid+text', ('uid', 'text')),
]
RECIF_SCALE_SCRATCH_REPRESENTATIONS = [
    representation
    for representation in RECIF_SCALE_REPRESENTATIONS
    if 'text' not in representation[0].split('+')
]


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
        .sid_variants(*SID_VARIANTS)
        .uid_variants(*UID_VARIANTS)
        .grid(
            'basic_llm',
            datasets=[dataset],
            models= ['qwen35th08b'],
            targets=['uid'],
            histories=LLM_HISTORIES,
            source_models=SOURCE_MODELS
        )
        .grid(
            'basic_llm',
            datasets=[dataset],
            models=['qwen35th08b'],
            targets=['sid'],
            histories=LLM_HISTORIES,
            source_models=['llama3']
        )
        .grid(
            'basic_scratch',
            datasets=[dataset],
            models=SCRATCH_MODELS,
            targets=['uid'],
            histories=SCRATCH_HISTORIES,
            source_models=SOURCE_MODELS
        )
        .grid(
            'basic_scratch',
            datasets=[dataset],
            models=SCRATCH_MODELS,
            targets=['sid'],
            histories=SCRATCH_HISTORIES,
            source_models=['llama3']
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


def _recif_scale_datasets(scales=None, prefixes=('ra',)):
    scales = RECIF_SCALE_PERCENTS if scales is None else list(scales)
    return [f'{prefix}{scale}' for prefix in prefixes for scale in scales]


def _group_representations(representations):
    grouped = {}
    for label, history in representations:
        target = label.split('+', 1)[0]
        grouped.setdefault(target, []).append(history)
    return grouped


def build_recif_scaling_schedule(scales=None):
    datasets = _recif_scale_datasets(scales)
    schedule = (
        Schedule(
            name='recif_scaling',
            effective_batch_size=64,
        )
        .main_metric('ndcg@10')
        .source_models(RECIF_SCALE_SOURCE_MODEL)
        .sid_variants(*RECIF_SCALE_SID_VARIANTS)
        .uid_variants('flat')
    )

    def sid_args(prefix: str):
        return {
            'sid_embedding_model': RECIF_SCALE_SOURCE_MODEL,
            'sid_codebook_size': 128 if prefix == 'ra' else 512,
        }

    for target, histories in _group_representations(RECIF_SCALE_SCRATCH_REPRESENTATIONS).items():
        if target == 'sid':
            for prefix in ('ra',):
                schedule.grid(
                    f'recif_scaling_scratch_{target}_{prefix}',
                    datasets=_recif_scale_datasets(scales, prefixes=(prefix,)),
                    models=['scratch'],
                    targets=[target],
                    histories=histories,
                    args=sid_args(prefix),
                )
        else:
            schedule.grid(
                f'recif_scaling_scratch_{target}',
                datasets=datasets,
                models=['scratch'],
                targets=[target],
                histories=histories,
            )

    for target, histories in _group_representations(RECIF_SCALE_REPRESENTATIONS).items():
        if target == 'sid':
            for prefix in ('ra',):
                schedule.grid(
                    f'recif_scaling_qwen35th08b_{target}_{prefix}',
                    datasets=_recif_scale_datasets(scales, prefixes=(prefix,)),
                    models=['qwen35th08b'],
                    targets=[target],
                    histories=histories,
                    args=sid_args(prefix),
                )
        else:
            schedule.grid(
                f'recif_scaling_qwen35th08b_{target}',
                datasets=datasets,
                models=['qwen35th08b'],
                targets=[target],
                histories=histories,
            )

    return schedule.export(Path('config/recif_scaling_ra_20_40.yaml'))


if __name__ == '__main__':
    # build_simple_schedule()
    # build_basic_schedules()
    # build_basic_schedule('recifvideoxlargeall')
    build_recif_scaling_schedule()
