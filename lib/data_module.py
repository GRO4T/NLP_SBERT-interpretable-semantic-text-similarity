from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from lib.dataset import SBERTDataset


TRAIN_VAL_SPLIT = 0.8


class SBERTDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: int,
        train_batch_size: int,
        num_workers: int,
        persistent_workers: bool,
    ):
        super().__init__()

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._batch_size = batch_size
        self._train_batch_size = train_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._train_path = train_path
        self._test_path = test_path
        self._preparte_data_per_node = True

    def _split(self, dataset, proportion):
        a = int(len(dataset) * proportion)
        b = len(dataset) - a
        return random_split(dataset, (a, b))

    def prepare_data(self):
        pass

    def setup(self, stage):
        if self._train_dataset is not None:
            return

        self._train_dataset, self._val_dataset = self._split(
            SBERTDataset(self._train_path), TRAIN_VAL_SPLIT
        )
        self._test_dataset = SBERTDataset(self._test_path)

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            persistent_workers=self._persistent_workers,
        )
