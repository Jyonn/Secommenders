from seq_process.base_amazon_seqprocessor import AmazonSeqProcessor
from seq_process.cds_clean_seqprocessor import CDsCleanSeqProcessor
from seq_process.cds_seqprocessor import CDsSeqProcessor
from seq_process.goodreads_seqprocessor import GoodreadsSeqProcessor
from seq_process.hm_seqprocessor import HMSeqProcessor
from seq_process.microlens_seqprocessor import MicroLensSeqProcessor
from seq_process.mind_seqprocessor import MINDSeqProcessor
from seq_process.movielens_seqprocessor import MovieLensSeqProcessor
from seq_process.pens_seqprocessor import PENSSeqProcessor
from seq_process.yelp_seqprocessor import YelpSeqProcessor


PROCESSOR_REGISTRY = {
    'amazon': AmazonSeqProcessor,
    'cds': CDsSeqProcessor,
    'cdsclean': CDsCleanSeqProcessor,
    'goodreads': GoodreadsSeqProcessor,
    'hm': HMSeqProcessor,
    'microlens': MicroLensSeqProcessor,
    'mind': MINDSeqProcessor,
    'movielens': MovieLensSeqProcessor,
    'pens': PENSSeqProcessor,
    'yelp': YelpSeqProcessor,
}


def get_processor(name: str):
    key = name.lower()
    if key not in PROCESSOR_REGISTRY:
        available = ', '.join(sorted(PROCESSOR_REGISTRY))
        raise ValueError(f'Unknown processor: {name}. Available: {available}')
    return PROCESSOR_REGISTRY[key]


def list_processors():
    return sorted(PROCESSOR_REGISTRY)
