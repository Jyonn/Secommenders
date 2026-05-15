from utils.data import get_data_dir
from utils.function import load_processor


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preview a sequential dataset processor.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--source', default='original', choices=['original', 'test', 'finetune'])
    parser.add_argument('--slicer', type=int, default=-20)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--data_dir', default=None, help='Optional raw data directory override.')
    args = parser.parse_args()

    data = args.data.lower()
    data_dir = args.data_dir or get_data_dir(data)
    processor = load_processor(data, data_dir=data_dir)
    processor.load()

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
