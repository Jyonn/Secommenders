import numpy as np
import pandas as pd
from tqdm import tqdm

from processors.cds_processor import CDsProcessor


class CDsCleanProcessor(CDsProcessor):
    def load_items(self) -> pd.DataFrame:
        items: pd.DataFrame = super().load_items()
        flags = []
        title_set = set()
        for _, item in tqdm(items.iterrows()):
            title = item['title']
            if title.lower() in title_set:
                flags.append(0)
            else:
                flags.append(1)
                title_set.add(title.lower())
        flags = np.array(flags)
        return items[flags == 1]
