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


def build_schedule():
    return (
        Schedule(
            name='basic',
            backend_auth='5ec0mmender5',
            backend_host='https://api.seco.qijiong.work',
            effective_batch_size=64,
        )
        .main_metric('ndcg@10')
        .source_models(*SOURCE_MODELS)
        .sid_variants(*SID_VARIANTS)
        .uid_variants(*UID_VARIANTS)
        .grid(
            'basic_llm',
            datasets=DATASETS,
            models=LLM_MODELS,
            targets=['uid', 'sid'],
            histories=LLM_HISTORIES,
        )
        .grid(
            'basic_scratch',
            datasets=DATASETS,
            models=SCRATCH_MODELS,
            targets=['uid', 'sid'],
            histories=SCRATCH_HISTORIES,
        )
    )


def main():
    output_path = Path('config/local_scheduler.yaml')
    build_schedule().export(output_path)
    print(output_path)


if __name__ == '__main__':
    main()
