from class_hub import ClassHub
from processors.base_amazon_processor import AmazonProcessor
from processors.base_processor import BaseProcessor
from processors.base_uict_processor import UICTProcessor
from processors.cds_clean_processor import CDsCleanProcessor
from processors.cds_processor import CDsProcessor
from processors.goodreads_processor import GoodreadsProcessor
from processors.hm_processor import HMProcessor
from processors.microlens_processor import MicroLensProcessor
from processors.mind_processor import MINDProcessor
from processors.movielens_processor import MovieLensProcessor
from processors.pens_processor import PENSProcessor
from processors.yelp_processor import YelpProcessor


def get_processor(name: str):
    processors = ClassHub.processors()
    key = name.lower()
    if key not in processors:
        available = ', '.join(sorted(processors.class_dict))
        raise ValueError(f'Unknown processor: {name}. Available: {available}')
    return processors[key]


def list_processors():
    return sorted(ClassHub.processors().class_dict)


__all__ = [
    'AmazonProcessor',
    'BaseProcessor',
    'CDsCleanProcessor',
    'CDsProcessor',
    'GoodreadsProcessor',
    'HMProcessor',
    'MINDProcessor',
    'MicroLensProcessor',
    'MovieLensProcessor',
    'PENSProcessor',
    'UICTProcessor',
    'YelpProcessor',
    'get_processor',
    'list_processors',
]
