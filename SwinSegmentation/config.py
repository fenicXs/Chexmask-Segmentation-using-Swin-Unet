"""
Configuration file for CheXmask Segmentation with Swin Transformer
"""

import os

# ========================
# Dataset Configuration
# ========================
CODE_DIR = r"/home/mmohan21/IPA/Segmentation/Pradeep/Swin2/SwinSegmentation"

BASE_PATH = r"/scratch/pkrish52"
DATASET_DIR = os.path.join(BASE_PATH, "CheXmask Dataset")
DATABASE_DIR = os.path.join(BASE_PATH, "CheXmask-Database")

DATASET_NAME = 'VinDr-CXR'
CSV_PATH = os.path.join(DATASET_DIR, "Preprocessed", f"{DATASET_NAME}.csv")

USE_COMPOSITE_DATASET = False
COMPOSITE_DATASETS = [
    'VinDr-CXR',
    'MIMIC-CXR-JPG',
    'CheXpert',
]

IMAGE_ROOTS = {
    "VinDr-CXR": r"/scratch/tprasad6/my_mimic_project/vinDR/vinbigdata",
    "MIMIC-CXR-JPG": r"/scratch/ischultz/Final_MIMIC",
    "CheXpert": r"/scratch/smanika3/chexpert/full_uncompressed",
}

if DATASET_NAME not in IMAGE_ROOTS:
    raise KeyError(f"Missing image root for dataset '{DATASET_NAME}'. Update IMAGE_ROOTS in config.py.")

IMAGE_BASE_PATH = IMAGE_ROOTS[DATASET_NAME]

SPLIT_CONFIG = {
    "VinDr-CXR": {
        "train_split": os.path.join(DATABASE_DIR, "VinDR", "VinDrCXR_train_pe_global_one.txt"),
        "test_split": os.path.join(DATABASE_DIR, "VinDR", "VinDrCXR_test_pe_global_one.txt"),
        "val_ratio": 0.1,
    },
    "MIMIC-CXR-JPG": {
        "split_csv": os.path.join(BASE_PATH, "MIMIC files", "mimic-cxr-2.0.0-split.csv"),
        "filename_map": os.path.join(BASE_PATH, "MIMIC files", "IMAGE_FILENAMES"),
    },
    "CheXpert": {
        "train_csv": os.path.join(IMAGE_ROOTS["CheXpert"], "CheXpert-v1.0_train.csv"),
        "valid_csv": os.path.join(IMAGE_ROOTS["CheXpert"], "CheXpert-v1.0_valid.csv"),
    },
}

QUALITY_THRESHOLD = 0.75

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# ========================
# Model Configuration
# ========================
NUM_CLASSES = 4

IMAGE_SIZE = 1024

# Backbone selection -------------------------------------------------
# Set to "tiny" (default) or "base" to switch Swin backbone width/depth.
# You can also override via environment variable SWIN_VARIANT.
SWIN_VARIANT = os.environ.get('SWIN_VARIANT', 'tiny').lower()

SWIN_VARIANTS = {
    'tiny': {
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'drop_path_rate': 0.2,
        'checkpoint': 'swin_tiny_patch4_window7_224.pth',
    },
    'base': {
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size': 8,
        'drop_path_rate': 0.3,
        'checkpoint': 'swin_base_patch4_window7_224.pth',
    },
}

if SWIN_VARIANT not in SWIN_VARIANTS:
    raise ValueError(
        f"Unknown SWIN_VARIANT '{SWIN_VARIANT}'. Choose from {sorted(SWIN_VARIANTS)}."
    )

_swin_settings = SWIN_VARIANTS[SWIN_VARIANT]

SWIN_CONFIG = {
    'img_size': IMAGE_SIZE,
    'patch_size': 4,
    'in_chans': 1,
    'num_classes': NUM_CLASSES,
    'embed_dim': _swin_settings['embed_dim'],
    'depths': _swin_settings['depths'],
    'num_heads': _swin_settings['num_heads'],
    'window_size': _swin_settings['window_size'],
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'drop_path_rate': _swin_settings['drop_path_rate'],
    'ape': False,
    'patch_norm': True,
}

USE_PRETRAINED = True
PRETRAINED_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "weights",
    _swin_settings['checkpoint'],
)

FREEZE_ENCODER_EPOCHS = 3
PRETRAINED_NORMALIZE_INPUT = True

# ========================
# Training Configuration
# ========================
BATCH_SIZE = 16
NUM_EPOCHS = 50

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

OPTIMIZER = 'adamw'

USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'
SCHEDULER_PATIENCE = 15
SCHEDULER_FACTOR = 0.5
WARMUP_EPOCHS = 8

GRADIENT_CLIP_NORM = 0.5

GRADIENT_ACCUMULATION_STEPS = 1

CLASS_WEIGHTS = [0.2, 1.0, 1.0, 1.5]

LOSS_TYPE = 'combined'
DICE_WEIGHT = 0.7
CE_WEIGHT = 0.3

USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20

# ========================
# Data Augmentation
# ========================
USE_AUGMENTATION = True
AUG_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 10,
    'brightness': 0.2,
    'contrast': 0.2,
    'gamma_range': (85, 115),
    'noise_std': 0.01,
}

# ========================
# Checkpoint & Logging
# ========================
CHECKPOINT_DIR = os.path.join(CODE_DIR, "checkpoints")

OUTPUT_DIR = os.path.join(CODE_DIR, "outputs")

SAVE_EVERY = 10

KEEP_BEST_N = 3

if USE_COMPOSITE_DATASET:
    EXPERIMENT_NAME = "swin_seg_composite"
else:
    EXPERIMENT_NAME = f"swin_seg_{DATASET_NAME}"

LOG_FREQ = 10

VAL_FREQ = 1

# ========================
# Hardware Configuration
# ========================
DEVICE = 'cuda'

NUM_WORKERS = 8

USE_AMP = True

USE_MULTI_GPU = False

# ========================
# Inference Configuration
# ========================
PRED_THRESHOLD = 0.5

USE_CRF = False
USE_MORPHOLOGY = True

SAVE_VISUALIZATIONS = True
VIS_OVERLAY_ALPHA = 0.4
VIS_FREQ = 5
VIS_ON_IMPROVEMENT = True

# ========================
# Evaluation Metrics
# ========================
METRICS = ['dice', 'iou', 'precision', 'recall', 'hausdorff']
COMPUTE_HAUSDORFF = False

# ========================
# Create necessary directories
# ========================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, EXPERIMENT_NAME), exist_ok=True)
