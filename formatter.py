from utils.data import get_data_dir
from utils.function import load_formatter
from utils.logging import setup_logging


if __name__ == '__main__':
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description='Format a raw dataset into standardized parquet artifacts.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    args = parser.parse_args()

    data = args.data.lower()
    data_dir = get_data_dir(data)
    formatter = load_formatter(data, data_dir=data_dir)
    formatter.load()
