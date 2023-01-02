import unittest

from torch import tensor

from lib.dataset import SBERTDataset


class Test_SBERT_iSTS(unittest.TestCase):
    def test_dataset(self):
        dataset = SBERTDataset("./data/sem_eval_2016/answers-students/test.tsv")

        item = dataset[0]

        expected_item = (('both', 'Bulbs A and C'), (tensor([0., 0., 0., 0., 0., 0., 0., 1.]), tensor(5.)))

        self.assertEqual(item[0], expected_item[0])
        self.assertEqual(item[1][1], expected_item[1][1])
