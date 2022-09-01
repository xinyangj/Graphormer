# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import torch
import numpy as np

from ..wrapper import preprocess_item
from .. import algos

import copy
from functools import lru_cache


class GraphormerSVGDataset(Dataset):
    def __init__(
        self,
        train_set,
        valid_set,
        test_set,
    ):
        
        self.num_data = len(train_set) + len(valid_set) + len(test_set)
        self.train_data = self.create_subset(train_set)
        self.valid_data = self.create_subset(valid_set)
        self.test_data = self.create_subset(test_set)
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        
        self.__indices__ = None
        

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            #item.y = item.y.reshape(-1)
            return self.dataset.preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data
