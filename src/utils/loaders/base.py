# Copyright (c) latataro (jchanxtarov). All rights reserved.
# Licensed under the MIT License.

import pickle
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..types import UserItems


@dataclass
class BaseDataset(metaclass=ABCMeta):
    n_train: int
    n_test: int
    n_users: int
    n_items: int
    dict_train_cf: UserItems
    dict_test_cf: UserItems
    arr_train_cf: np.ndarray
    arr_test_cf: np.ndarray
    uniq_users: List[int]

    def __init__(self, dataset: str) -> None:
        # read files
        path_train_cf = f"./src/datasets/{dataset}/train.txt"
        path_test_cf = f"./src/datasets/{dataset}/test.txt"

        self.arr_train_cf, self.dict_train_cf = self._load_cf_txt(path_train_cf)
        self.arr_test_cf, self.dict_test_cf = self._load_cf_txt(path_test_cf)
        self.n_train = len(self.arr_train_cf)
        self.n_test = len(self.arr_test_cf)

        # get statics
        self.uniq_users = sorted(list(self.dict_train_cf.keys()))
        self.n_users = max(self.uniq_users) + 1
        self.n_items = max(max(self.arr_train_cf[:, 1]), max(self.arr_test_cf[:, 1])) + 1

    def _load_cf_dict(self, path_dataset: str) -> Tuple[np.ndarray, UserItems]:
        with open(path_dataset, mode='rb') as f:
            dict_uid_iids = pickle.load(f)
        list_uid_iid = [[uid, iid] for (uid, iids) in dict_uid_iids.items() for iid in list(set(iids))]
        return np.array(list_uid_iid), dict_uid_iids

    def _load_cf_txt(self, path_dataset: str) -> Tuple[np.ndarray, UserItems]:
        dict_uid_iids = dict()
        list_uid_iid = list()

        lines = open(path_dataset, "r").readlines()
        for line in lines:
            tmps = line.strip()
            inters = [int(i) for i in tmps.split(" ")]

            user, items = inters[0], inters[1:]
            items = list(set(items))

            for item in items:
                list_uid_iid.append([user, item])

            if len(items) > 0:
                dict_uid_iids[user] = items
        return np.array(list_uid_iid), dict_uid_iids


    @abstractmethod
    def logging_statistics(self) -> None:
        raise NotImplementedError()
