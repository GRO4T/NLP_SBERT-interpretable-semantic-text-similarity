import unittest

from torch import tensor

import numpy as np

from lib.dataset import SBERTDataset
from lib.data_module import SBERTDataModule


class Test_SBERT_iSTS(unittest.TestCase):
    def test_dataset(self):
        dataset = SBERTDataset("./data/sem_eval_2016/answers-students/test.tsv")

        item = dataset[0]

        expected_item = (
            ("both", "Bulbs A and C"),
            (tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), tensor(5.0)),
        )

        self.assertEqual(item[0], expected_item[0])
        self.assertEqual(item[1][1], expected_item[1][1])

    def test_data_module(self):
        test_path = "./data/sem_eval_2016/answers-students/test.tsv"
        train_path = "./data/sem_eval_2016/answers-students/train.tsv"

        data_module = SBERTDataModule(
            train_path, test_path, batch_size=16, train_batch_size=16, num_workers=2
        )
    
    def test_numpy_concatenate(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        
        c = np.concatenate((a, b))

        expected_c = np.array([1, 2, 3, 4, 5, 6])

        self.assertEqual(c.all(), expected_c.all())
