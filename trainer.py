""" Trainer script """

import click

from pytorch_lightning import Trainer

from lib.model import SingleLayeredHeadJointLearningWithSBERTFrozen
from lib.data_module import SBERTDataModule
from lib.params import (
    DATA_DIR,
    BATCH_SIZE,
    TRAIN_BATCH_SIZE,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    ACCELERATOR,
    EPOCHS,
)

from evaluate import evaluate

MODELS = {0: SingleLayeredHeadJointLearningWithSBERTFrozen}


@click.command()
@click.option("--model_id", default=0)
@click.option("--strategy", default=None)
@click.option("--workers", default=NUM_WORKERS)
@click.option("--persistent_workers", default=PERSISTENT_WORKERS)
def train(model_id, strategy, workers, persistent_workers):
    model = MODELS[model_id]()
    data = SBERTDataModule(
        f"{DATA_DIR}/train.tsv",
        f"{DATA_DIR}/test.tsv",
        batch_size=BATCH_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        num_workers=workers,
        persistent_workers=persistent_workers,
    )

    trainer = Trainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, strategy=strategy)

    trainer.fit(model, data)

    trainer.test(model, data)
    predictions = trainer.predict(model, data)
    evaluate(predictions)


if __name__ == "__main__":
    train()
