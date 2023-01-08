""" Trainer script """

import click

from pytorch_lightning import Trainer

from lib.model import (
    SingleLayeredHeadJointLearningWithSBERTFrozen, 
    SingleLayeredScoringHeadSeparateLearning, 
    SingleLayeredClassHeadSeparateLearning,
)
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

MODELS = {
        0: SingleLayeredHeadJointLearningWithSBERTFrozen,
        1: (SingleLayeredScoringHeadSeparateLearning, SingleLayeredClassHeadSeparateLearning)}


@click.command()
@click.option("--model_id", default=0)
@click.option("--strategy", default=None)
@click.option("--workers", default=NUM_WORKERS)
@click.option("--persistent_workers", default=PERSISTENT_WORKERS)
def train(model_id, strategy, workers, persistent_workers):
    data = SBERTDataModule(
        f"{DATA_DIR}/train.tsv",
        f"{DATA_DIR}/test.tsv",
        batch_size=BATCH_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        num_workers=workers,
        persistent_workers=persistent_workers,
    )
    if model_id == 1:
        model_scoring = MODELS[model_id][0]()
        model_class = MODELS[model_id][1]()
        trainer_scoring = Trainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, strategy=strategy)
        trainer_class = Trainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, strategy=strategy)
        trainer_scoring.fit(model_scoring, data)
        trainer_class.fit(model_class, data)
        
        trainer_scoring.test(model_scoring, data)
        trainer_class.test(model_class, data)

        predictions_scoring = trainer_scoring.predict(model_scoring, data)
        predictions_class = trainer_class.predict(model_class, data)
        evaluate([(i,j) for i,j in zip(predictions_class, predictions_scoring)])
    else:
        model = MODELS[model_id]()

        trainer = Trainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, strategy=strategy)

        trainer.fit(model, data)

        trainer.test(model, data)
        predictions = trainer.predict(model, data)
        evaluate(predictions)


if __name__ == "__main__":
    train()
