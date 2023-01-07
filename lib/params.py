import os

PROJECT_DIR = "./"  # @param {type: "string"}
DATASET = "headlines"  # @param {type: "string"}
DATA_DIR = os.path.join(PROJECT_DIR, "data/sem_eval_2016/", DATASET)

SBERT_EMBEDDING_WIDTH = 768

TRAIN_VAL_SPLIT = 0.8  # @param {type: "slider", min:0, max: 1}

TRAIN_BATCH_SIZE = 16  # @param {type: "slider", min:1, max:128}
BATCH_SIZE = 16  # @param {type: "slider", min:1, max:128}
NUM_WORKERS = 0  # @param {type: "slider", min:1, max:16}
PERSISTENT_WORKERS = False  # TODO param

EPOCHS = 1  # @param {type: "slider", min:1, max:128}
ACCELERATOR = "auto"  # @param ["auto", "gpu", "tpu", "cpu"]
