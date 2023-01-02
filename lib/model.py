from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from lib.utils import TYPES_MAP


class SBERT_iSTS_Model(LightningModule):
    def __init__(
        self, sbert_model: str = "all-mpnet-base-v2", learning_rate: float = 0.001
    ):
        super().__init__()
        self._sbert = SentenceTransformer(sbert_model)
        # kolasdam(TODO) sparametryzować
        self._scoring_head = torch.nn.Linear(in_features=384 * 2, out_features=1)
        self._class_head = torch.nn.Linear(
            in_features=384 * 2, out_features=len(TYPES_MAP)
        )
        self._learning_rate = learning_rate
        self.save_hyperparameters()

    def _step(self, batch, batch_idx, id: str):
        x, y = batch
        y_hat = self.forward(x)
        return self.loss(y, y_hat, id)

    def forward(self, x):
        embedding_a = self._sbert.encode(x[0])
        embedding_b = self._sbert.encode(x[1])

        ists_head_input = np.concatenate((embedding_a, embedding_b))

        score = torch.reshape(self._scoring_head(ists_head_input), (-1,))
        cls = torch.nn.functional.softmax(self._class_head(ists_head_input), dim=1)

        return cls, score

    def loss(self, y, y_hat, id):
        # Klasa i ocena uczone razem
        scoring_loss = torch.nn.functional.mse_loss(y_hat[1], y[1])
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_hat[0], y[0]
        )
        return scoring_loss + class_loss

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
        x, y = batch
        types, scores = self.forward(x)

        return torch.argmax(types, dim=1), torch.clamp(
            torch.round(scores).int(), min=0, max=5
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)
