from types import SimpleNamespace

from pigmento import pnt

from utils.compile import CompileConfig
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_formatter, load_processor
from utils.logging import setup_logging
from utils.artifact_identity import compiled_signature_from_config


def ensure_formatted(data: str):
    setup_logging()
    pnt(f'auto preparing formatted artifacts for {data}')
    formatter = load_formatter(data, data_dir=get_data_dir(data))
    formatter.load()
    return formatter


def ensure_processed(data: str):
    setup_logging()
    pnt(f'auto preparing processed artifacts for {data}')
    processor = load_processor(data)
    processor.load()
    return processor


def ensure_embedded(
        data: str,
        model: str,
        device=None,
        batch_size=32,
        normalize=False,
        overwrite=False,
        embedding_spec: dict | None = None,
):
    setup_logging()
    pnt(f'auto preparing embedded artifacts for {data}/{model}')
    from embedder import Embedder

    conf = SimpleNamespace(
        data=data,
        model=model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        overwrite=overwrite,
        word2vec_config=embedding_spec,
    )
    embedder = Embedder(conf)
    embedder.embed()
    return embedder


def ensure_quantized(data: str, model: str, quantizer_name: str | None = None, quantizer_spec: dict | None = None):
    setup_logging()
    suffix = f'/{quantizer_name}' if quantizer_name else ''
    pnt(f'auto preparing quantized artifacts for {data}/{model}{suffix}')
    from quantizer import Quantizer

    quantizer_spec = quantizer_spec or {}
    quantizer = quantizer_spec.get('quantizer') or {}
    encoder = quantizer_spec.get('encoder') or {}
    trainer = quantizer_spec.get('trainer') or {}
    hash_config = quantizer.get('config') or {}
    quantizer_config = quantizer.get('config') or {}
    if (quantizer.get('name') or quantizer_name) in {'lsh', 'simhash', 'pcahash', 'itq'}:
        hash_config = quantizer.get('config') or {}

    kwargs = {
        'data': data,
        'model': model,
        'quantizer_name': quantizer.get('name') or quantizer_name,
        'config': 'config/quantizer.yaml',
        'hidden_dims': (encoder.get('config') or {}).get('hidden_dims'),
        'latent_dim': quantizer_config.get('latent_dim'),
        'reconstruction_loss': quantizer_config.get('reconstruction_loss'),
        'codebook_size': quantizer_config.get('codebook_size'),
        'commitment_weight': quantizer_config.get('commitment_weight'),
        'codebook_weight': quantizer_config.get('codebook_weight'),
        'use_ema_codebook': quantizer_config.get('use_ema_codebook'),
        'ema_decay': quantizer_config.get('ema_decay'),
        'ema_epsilon': quantizer_config.get('ema_epsilon'),
        'dead_code_reset': quantizer_config.get('dead_code_reset'),
        'dead_code_threshold': quantizer_config.get('dead_code_threshold'),
        'num_quantizers': quantizer_config.get('num_quantizers'),
        'num_codebooks': quantizer_config.get('num_codebooks'),
        'assignment_strategy': quantizer_config.get('assignment_strategy'),
        'sinkhorn_epsilon': quantizer_config.get('sinkhorn_epsilon'),
        'sinkhorn_iters': quantizer_config.get('sinkhorn_iters'),
        'kmeans_init': quantizer_config.get('kmeans_init'),
        'kmeans_iters': quantizer_config.get('kmeans_iters'),
        'num_bits': hash_config.get('num_bits'),
        'num_tables': hash_config.get('num_tables'),
        'projection_distribution': hash_config.get('projection_distribution'),
        'use_median_thresholds': hash_config.get('use_median_thresholds'),
        'num_iterations': hash_config.get('num_iterations'),
        'normalize_inputs': hash_config.get('normalize_inputs'),
        'validation_ratio': trainer.get('validation_ratio'),
        'test_ratio': trainer.get('test_ratio'),
        'epoch': trainer.get('epochs'),
        'batch_size': trainer.get('batch_size'),
        'lr': trainer.get('learning_rate'),
        'patience': trainer.get('patience'),
        'save_best_by': trainer.get('save_best_by'),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    configurations = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            config='config/quantizer.yaml',
        ),
        makedirs=[],
    ).parse_kwargs(kwargs)
    quantizer = Quantizer(configurations.data, configurations.model, configurations.config)
    quantizer.run()
    return quantizer


def ensure_clustered(data: str, uid_cluster_levels: str, clusterer_spec: dict | None = None):
    setup_logging()
    from clusterer import Clusterer, ClustererConfig

    clusterer_spec = clusterer_spec or {}
    embedding = clusterer_spec.get('embedding') or {}
    word2vec = clusterer_spec.get('word2vec') or {}
    cluster = clusterer_spec.get('cluster') or {}
    source = embedding.get('source') or 'collaborative'
    content_model = embedding.get('content_model')
    pnt(
        f'auto preparing clustered artifacts for {data}/{uid_cluster_levels} '
        f'source={source}'
        + (f' content_model={content_model}' if content_model else '')
    )

    kwargs = {
        'data': data,
        'uid_cluster_levels': uid_cluster_levels,
        'cluster_embedding_source': embedding.get('source'),
        'cluster_content_model': embedding.get('content_model'),
        'cluster_content_reduce_dim': embedding.get('content_reduce_dim'),
        'cluster_normalize_blocks': embedding.get('normalize_blocks'),
        'cluster_mix_alpha': embedding.get('mix_alpha'),
        'cluster_vector_size': word2vec.get('vector_size'),
        'cluster_window': word2vec.get('window'),
        'cluster_patience': word2vec.get('patience'),
        'cluster_sg': word2vec.get('sg'),
        'cluster_negative': word2vec.get('negative'),
        'cluster_min_count': word2vec.get('min_count'),
        'cluster_workers': word2vec.get('workers'),
        'cluster_seed': word2vec.get('seed'),
        'cluster_max_epochs': word2vec.get('max_epochs'),
        'cluster_learning_rate': word2vec.get('learning_rate'),
        'cluster_word2vec_batch_size': word2vec.get('batch_size'),
        'cluster_valid_batch_size': word2vec.get('valid_batch_size'),
        'cluster_min_delta': word2vec.get('min_delta'),
        'cluster_batch_size': cluster.get('batch_size'),
        'cluster_max_iter': cluster.get('max_iter'),
        'cluster_n_init': cluster.get('n_init'),
        'config': 'config/clusterer.yaml',
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    configurations = ConfigInit(
        required_args=['data'],
        default_args=dict(
            config='config/clusterer.yaml',
        ),
        makedirs=[],
    ).parse_kwargs(kwargs)
    clusterer = Clusterer(ClustererConfig.from_refconfig(configurations))
    clusterer.run()
    return clusterer


def ensure_compiled(config: CompileConfig):
    setup_logging()
    signature = compiled_signature_from_config(config)
    pnt(f'auto preparing compiled artifacts for {config.data}/{signature}')
    from compiler import Compiler

    compiler = Compiler(config)
    compiler.run()
    return compiler
