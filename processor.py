from pigmento import pnt

from utils.function import load_processor
from utils.logging import setup_logging


if __name__ == '__main__':
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description='Build or load processed dataset artifacts from formatted data.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    args = parser.parse_args()

    data = args.data.lower()
    processor = load_processor(data)
    processor.load()

    pnt(f'Dataset: {data}')
    pnt(f'Processed items: {len(processor.items)}')
    pnt(f'Test users: {0 if processor.test_set is None else len(processor.test_set)}')
    pnt(f'Finetune users: {0 if processor.finetune_set is None else len(processor.finetune_set)}')
