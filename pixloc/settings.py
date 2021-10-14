from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = root / 'datasets/'  # datasets and pretrained weights
TRAINING_PATH = root / 'outputs/training/'  # training checkpoints
LOC_PATH = root / 'outputs/hloc/'  # localization logs
EVAL_PATH = root / 'outputs/results/'  # evaluation results
