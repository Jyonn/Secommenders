from types import SimpleNamespace

from pigmento import pnt

from utils.compile import CompileConfig
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_formatter, load_processor
from utils.logging import setup_logging


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


def ensure_embedded(data: str, model: str, device=None, batch_size=32, normalize=False, overwrite=False):
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
    )
    embedder = Embedder(conf)
    embedder.embed()
    return embedder


def ensure_quantized(data: str, model: str, quantizer_name: str | None = None):
    setup_logging()
    suffix = f'/{quantizer_name}' if quantizer_name else ''
    pnt(f'auto preparing quantized artifacts for {data}/{model}{suffix}')
    from quantizer import Quantizer

    configurations = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            config='config/quantizer.yaml',
        ),
        makedirs=[],
    ).parse_kwargs(
        {
            'data': data,
            'model': model,
            'quantizer_name': quantizer_name,
            'config': 'config/quantizer.yaml',
        }
    )
    quantizer = Quantizer(configurations.data, configurations.model, configurations.config)
    quantizer.run()
    return quantizer


def ensure_clustered(data: str, uid_cluster_levels: str, clusterer_spec: dict | None = None):
    setup_logging()
    pnt(f'auto preparing clustered artifacts for {data}/{uid_cluster_levels}')
    from clusterer import Clusterer, ClustererConfig

    clusterer_spec = clusterer_spec or {}
    word2vec = clusterer_spec.get('word2vec') or {}
    cluster = clusterer_spec.get('cluster') or {}

    kwargs = {
        'data': data,
        'uid_cluster_levels': uid_cluster_levels,
        'cluster_vector_size': word2vec.get('vector_size'),
        'cluster_window': word2vec.get('window'),
        'cluster_patience': word2vec.get('patience'),
        'cluster_sg': word2vec.get('sg'),
        'cluster_negative': word2vec.get('negative'),
        'cluster_min_count': word2vec.get('min_count'),
        'cluster_workers': word2vec.get('workers'),
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
    pnt(f'auto preparing compiled artifacts for {config.data}/{config.prepare_id}')
    from compiler import Compiler

    compiler = Compiler(config)
    compiler.run()
    return compiler
