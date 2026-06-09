import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.uid_hierarchy import parse_uid_cluster_topk


class UIDHierarchyArtifacts:
    def __init__(self, directory: str | Path, compiled, topk_spec: str):
        self.directory = Path(directory)
        self.compiled = compiled

        self.meta = json.loads((self.directory / 'meta.json').read_text())
        item_col = self.meta.get('item_col', 'item_id')
        item_ids = pd.read_parquet(self.directory / 'item_ids.parquet')[item_col]
        self.item_ids = [str(item_id) for item_id in item_ids.tolist()]
        compiled_item_ids = [str(item_id) for item_id in compiled.uid_raw_items]
        if self.item_ids != compiled_item_ids:
            raise ValueError('uid hierarchy item order does not match compiled uid order')

        self.item_node_ids = torch.tensor(np.load(self.directory / 'item_node_ids.npy'), dtype=torch.long)
        self.item_labels = torch.tensor(np.load(self.directory / 'item_labels.npy'), dtype=torch.long)
        node_meta = pd.read_parquet(self.directory / 'node_meta.parquet').sort_values('node_id').reset_index(drop=True)
        self.node_child_counts = [int(value) for value in node_meta['child_count'].tolist()]
        self.node_levels = [int(value) for value in node_meta['level'].tolist()]
        self.depth = int(self.meta['hierarchy_depth'])
        self.resolved_levels = [int(value) for value in self.meta['resolved_levels']]
        self.topk_per_level = parse_uid_cluster_topk(topk_spec, self.depth)

        child_nodes_frame = pd.read_parquet(self.directory / 'child_nodes.parquet')
        self.child_nodes = {
            int(parent_node_id): [int(value) for value in frame.sort_values('child_label')['child_node_id'].tolist()]
            for parent_node_id, frame in child_nodes_frame.groupby('parent_node_id', sort=False)
        }
        leaf_items_frame = pd.read_parquet(self.directory / 'leaf_items.parquet')
        self.leaf_items = {
            int(parent_node_id): [int(value) for value in frame.sort_values('local_label')['item_uid'].tolist()]
            for parent_node_id, frame in leaf_items_frame.groupby('parent_node_id', sort=False)
        }
