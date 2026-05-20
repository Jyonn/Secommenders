from utils.function import load_processor


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build or load processed dataset artifacts from formatted data.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    args = parser.parse_args()

    data = args.data.lower()
    processor = load_processor(data)
    processor.load()

    print(f'Dataset: {data}')
    print(f'Processed items: {len(processor.items)}')
    print(f'Test users: {0 if processor.test_set is None else len(processor.test_set)}')
    print(f'Finetune users: {0 if processor.finetune_set is None else len(processor.finetune_set)}')
