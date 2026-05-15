from processors.base_amazon_processor import AmazonProcessor
from processors.cds_clean_processor import CDsCleanProcessor
from processors.cds_processor import CDsProcessor
from processors.goodreads_processor import GoodreadsProcessor
from processors.hm_processor import HMProcessor
from processors.microlens_processor import MicroLensProcessor
from processors.mind_processor import MINDProcessor
from processors.movielens_processor import MovieLensProcessor
from processors.pens_processor import PENSProcessor
from processors.yelp_processor import YelpProcessor


PROCESSOR_REGISTRY = {
    'amazon': AmazonProcessor,
    'cds': CDsProcessor,
    'cdsclean': CDsCleanProcessor,
    'goodreads': GoodreadsProcessor,
    'hm': HMProcessor,
    'microlens': MicroLensProcessor,
    'mind': MINDProcessor,
    'movielens': MovieLensProcessor,
    'pens': PENSProcessor,
    'yelp': YelpProcessor,
}


def get_processor(name: str):
    key = name.lower()
    if key not in PROCESSOR_REGISTRY:
        available = ', '.join(sorted(PROCESSOR_REGISTRY))
        raise ValueError(f'Unknown processor: {name}. Available: {available}')
    return PROCESSOR_REGISTRY[key]


def list_processors():
    return sorted(PROCESSOR_REGISTRY)
