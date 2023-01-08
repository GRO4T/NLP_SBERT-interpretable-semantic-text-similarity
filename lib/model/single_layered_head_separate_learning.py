from pytorch_lightning import LightningModule
import torch

from lib.utils import TYPES_MAP
from lib.params import SBERT_EMBEDDING_WIDTH


class SingleLayeredClassHeadSeparateLearning(LightningModule):
    def __init__(
        self, sbert_model: str = "all-mpnet-base-v2", learning_rate: float = 0.001
    ):
        super().__init__()
        self._class_head = torch.nn.Linear(
            in_features=SBERT_EMBEDDING_WIDTH * 2, out_features=len(TYPES_MAP)
        )
        self._learning_rate = learning_rate
        self.save_hyperparameters()

    def _step(self, batch, batch_idx, id: str):
        x, y = batch
        y_hat = self.forward(x)
        return self.loss(y[0], y_hat, id)

    def forward(self, x):
        cls = torch.nn.functional.softmax(self._class_head(x), dim=1)

        return cls

    def loss(self, y, y_hat, id):
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_hat, y
        )
        return class_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        types = self.forward(x)

        return torch.argmax(types, dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)


class SingleLayeredScoringHeadSeparateLearning(LightningModule):
    def __init__(
        self, sbert_model: str = "all-mpnet-base-v2", learning_rate: float = 0.001
    ):
        super().__init__()
        self._scoring_head = torch.nn.Linear(
            in_features=SBERT_EMBEDDING_WIDTH * 2, out_features=1
        )
        self._learning_rate = learning_rate
        self.save_hyperparameters()

    def _step(self, batch, batch_idx, id: str):
        x, y = batch
        y_hat = self.forward(x)
        return self.loss(y[1], y_hat, id)

    def forward(self, x):
        score = torch.reshape(self._scoring_head(x), (-1,))
        return score

    def loss(self, y, y_hat, id):
        scoring_loss = torch.nn.functional.mse_loss(y_hat, y)
        return scoring_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        scores = self.forward(x)

        return torch.clamp(torch.round(scores).int(), min=0, max=5)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)