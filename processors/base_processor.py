import abc
import json
import os.path
import random
from typing import Callable, Optional, Union

import pandas as pd
from oba import Obj
from pigmento import pnt
from tqdm import tqdm


class Meta:
    VER = 'v1.0'

    def __init__(self, path):
        self.path = path

        if not os.path.exists(path):
            self.compressed = False
            self.version = self.VER
        else:
            data = json.load(open(path, 'r'))
            data = Obj(data)
            self.compressed = data.fully_compressed
            self.version = data.version

    def save(self):
        data = {
            'fully_compressed': self.compressed,
            'version': self.version,
        }
        json.dump(data, open(self.path, 'w'))


class BaseProcessor(abc.ABC):
    IID_COL: str
    UID_COL: str
    HIS_COL: str

    NUM_TEST: int
    NUM_FINETUNE: int

    MAX_HISTORY_PER_USER: int = 100
    REQUIRE_STRINGIFY: bool
    BASE_STORE_DIR = 'data'

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.store_dir = os.path.join(self.BASE_STORE_DIR, self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.meta = Meta(os.path.join(self.store_dir, 'meta.json'))
        self._loaded = False

        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.test_set: Optional[pd.DataFrame] = None
        self.finetune_set: Optional[pd.DataFrame] = None

    @property
    @abc.abstractmethod
    def default_attrs(self):
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Processor', '').lower()

    @abc.abstractmethod
    def load_items(self) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_users(self) -> pd.DataFrame:
        raise NotImplementedError

    def _stringify(self, df: pd.DataFrame):
        if not self.REQUIRE_STRINGIFY:
            return df
        if self.IID_COL in df.columns:
            df[self.IID_COL] = df[self.IID_COL].astype(str)
        if self.UID_COL in df.columns:
            df[self.UID_COL] = df[self.UID_COL].astype(str)
        return df

    def compress(self):
        if self.meta.compressed:
            return False

        item_set = set()
        old_item_size = len(self.items)
        self.users[self.HIS_COL].apply(lambda x: [item_set.add(i) for i in x])
        self.items = self.items[self.items[self.IID_COL].isin(item_set)].reset_index(drop=True)
        pnt(f'compressed items from {old_item_size} to {len(self.items)}')

        self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))

        self.meta.compressed = True
        self.meta.save()
        return True

    def load(self):
        items_path = os.path.join(self.store_dir, 'items.parquet')
        users_path = os.path.join(self.store_dir, 'users.parquet')

        if os.path.exists(items_path) and os.path.exists(users_path):
            pnt(f'loading {self.get_name()} from cache')
            self.items = pd.read_parquet(items_path)
            pnt(f'loaded {len(self.items)} items')
            self.users = pd.read_parquet(users_path)
            pnt(f'loaded {len(self.users)} users')

            self.items = self._stringify(self.items)
            self.users = self._stringify(self.users)
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self.load_items()
            self.items = self._stringify(self.items)
            pnt(f'loaded {len(self.items)} items')

            self.users = self.load_users()
            self.users = self._stringify(self.users)
            pnt(f'loaded {len(self.users)} users')

            self.items.to_parquet(items_path)
            self.users.to_parquet(users_path)

        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(lambda x: [str(item) for item in x])

        self.item_vocab = dict(zip(self.items[self.IID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.UID_COL], range(len(self.users))))

        if self.compress():
            pnt(f'compressed {self.get_name()} data, re-run to load compressed data')
            return self.load()

        self.load_public_sets()
        return self

    def organize_item(self, iid, item_attrs: list, as_dict=False, item_self=False):
        item = iid if item_self else self.items.iloc[self.item_vocab[iid]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}
        if len(item_attrs) == 1:
            return item[item_attrs[0]]
        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])

    @staticmethod
    def _build_slicer(slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]

        return _slicer

    def _iterate(self, dataframe: pd.DataFrame, slicer: Union[int, Callable], **kwargs):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)

        for _, row in dataframe.iterrows():
            uid = row[self.UID_COL]
            history = slicer(row[self.HIS_COL])
            yield uid, history

    def get_source_set(self, source):
        assert source in ['test', 'finetune', 'original'], 'source must be test, finetune, or original'
        return self.users if source == 'original' else getattr(self, f'{source}_set')

    def generate(self, slicer: Union[int, Callable], source='test', **kwargs):
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)
        return self._iterate(source_set, slicer)

    def iterate(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='original')

    def test(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='test')

    def finetune(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='finetune')

    def _iterator(self, user_order, users):
        for uid in user_order:
            user = users[users[self.UID_COL] == uid]
            yield user.iloc[0]

    def _split(self, iterator, count):
        users = []
        for user in tqdm(iterator, total=count):
            users.append(user)
            if len(users) >= count:
                break
        return pd.DataFrame(users)

    def _load_user_order(self):
        path = os.path.join(self.store_dir, 'user_order.txt')
        if os.path.exists(path):
            with open(path, 'r') as file:
                return [line.strip() for line in file]

        users = self.users[self.UID_COL].unique().tolist()
        random.shuffle(users)
        with open(path, 'w') as file:
            for user in users:
                file.write(f'{user}\n')

        return users

    @property
    def test_set_required(self):
        return self.NUM_TEST > 0

    @property
    def finetune_set_required(self):
        return self.NUM_FINETUNE > 0

    @property
    def test_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'test.parquet')) or not self.test_set_required

    @property
    def finetune_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'finetune.parquet')) or not self.finetune_set_required

    def load_public_sets(self):
        if self.test_set_valid and self.finetune_set_valid:
            pnt(f'loading {self.get_name()} public splits from cache')

            if self.NUM_TEST:
                self.test_set = pd.read_parquet(os.path.join(self.store_dir, 'test.parquet'))
                self.test_set = self._stringify(self.test_set)
                pnt('loaded test set')

            if self.NUM_FINETUNE:
                self.finetune_set = pd.read_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
                self.finetune_set = self._stringify(self.finetune_set)
                pnt('loaded finetune set')

            self._loaded = True
            return

        pnt(f'processing {self.get_name()} public splits from users')
        users_order = self._load_user_order()
        iterator = self._iterator(users_order, self.users)

        if self.NUM_TEST:
            self.test_set = self._split(iterator, self.NUM_TEST)
            self.test_set.reset_index(drop=True, inplace=True)
            self.test_set.to_parquet(os.path.join(self.store_dir, 'test.parquet'))
            pnt(f'generated test set with {len(self.test_set)}/{self.NUM_TEST} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self._split(iterator, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.finetune_set.to_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
            pnt(f'generated finetune set with {len(self.finetune_set)}/{self.NUM_FINETUNE} samples')

        self._loaded = True
