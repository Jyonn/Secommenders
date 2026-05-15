from processors.registry import get_processor


def load_processor(dataset, data_dir=None):
    processor = get_processor(dataset)
    return processor(data_dir=data_dir)
