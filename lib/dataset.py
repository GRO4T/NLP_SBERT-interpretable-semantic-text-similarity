import csv

import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.utils import types_to_int, TYPES_MAP


class SBERTDataset(Dataset):
    def __init__(self, file_path: str):
        self._data = pd.read_csv(
            file_path, sep="\t", keep_default_na=False, quoting=csv.QUOTE_NONE
        )
        
        self._x1 = self._data["x1"].tolist()
        self._x2 = self._data["x2"].tolist()

        self._types = self._get_encoded_types()
        self._scores = torch.tensor(self._data["y_score"]).float()

    def _get_encoded_types(self):
        types_as_int = types_to_int(self._data["y_type"].tolist())
        encoded_types = torch.nn.functional.one_hot(
            torch.tensor(types_as_int),
            num_classes=len(TYPES_MAP)
        ).float()
        return encoded_types
    
    def __getitem__(self, index):
        a, b = self._x1[index], self._x2[index]
        x = (a, b)
        y = (self._types[index], self._scores[index])
        return x, y
    
    def __len__(self):
        return self._types.shape[0]
