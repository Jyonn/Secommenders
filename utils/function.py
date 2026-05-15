from utils.class_hub import ClassHub


def load_processor(dataset, data_dir=None):
    processors = ClassHub.processors()
    key = dataset.lower()
    if key not in processors:
        available = ', '.join(sorted(processors.class_dict))
        raise ValueError(f'Unknown processor: {dataset}. Available: {available}')
    processor = processors[key]
    return processor(data_dir=data_dir)
