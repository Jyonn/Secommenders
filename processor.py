from utils.data import get_data_dir
from utils.function import load_processor


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build or load processed dataset artifacts from formatted data.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--source', default=None, choices=['original', 'test', 'finetune'])
    parser.add_argument('--slicer', type=int, default=-20, help='History slicer for optional preview.')
    parser.add_argument('--limit', type=int, default=3, help='Preview sample count.')
    args = parser.parse_args()

    data = args.data.lower()
    processor = load_processor(data, data_dir=get_data_dir(data))
    processor.load()

    print(f'Dataset: {data}')
    print(f'Processed items: {len(processor.items)}')
    print(f'Test users: {0 if processor.test_set is None else len(processor.test_set)}')
    print(f'Finetune users: {0 if processor.finetune_set is None else len(processor.finetune_set)}')

    if args.source:
        for index, (uid, history) in enumerate(
            processor.generate(slicer=args.slicer, source=args.source)
        ):
            print(f'User: {uid}')
            print('History:')
            for item_index, item in enumerate(history):
                print(f'  {item_index:2d}: {item}')
            print()

            if index + 1 >= args.limit:
                break
