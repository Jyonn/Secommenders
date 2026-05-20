import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


class CompiledSampleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.rows = []
        for row in dataframe.to_dict('records'):
            self.rows.append(
                {
                    'uid': row['uid'],
                    'history_uids': [int(value) for value in to_list(row['history_uids'])],
                    'target_uid': int(row['target_uid']),
                    'history_item_count': int(row['history_item_count']),
                    'total_input_length': int(row['total_input_length']),
                    'target_pos': int(row['target_pos']),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
