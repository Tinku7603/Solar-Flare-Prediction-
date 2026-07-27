"""
config.py
---------
Central configuration for the SWAN-SF solar flare prediction pipeline.
Edit the paths and hyperparameters here before running main.py.
"""

import os
import sys

IN_COLAB = "google.colab" in sys.modules
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ isn't defined when this code runs directly in a notebook
    # cell (e.g. pasted instead of written via %%writefile) — fall back
    # to the current working directory.
    BASE_DIR = os.getcwd()

# ------------------------------------------------------------------
# PATHS  (edit these to point at your actual SWAN-SF download)
# ------------------------------------------------------------------
if IN_COLAB:
    RAW_DATA_ROOT = "/content/SWAN-SF"
    CACHE_DIR = "/content/swan_sf_cache"
    OUTPUT_DIR = "/content/swan_sf_outputs"
else:
    RAW_DATA_ROOT = os.path.join(BASE_DIR, "SWAN-SF")
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PARTITION_DIRS = {
    "p1": os.path.join(RAW_DATA_ROOT, "partition1"),
    "p2": os.path.join(RAW_DATA_ROOT, "partition2"),
    "p3": os.path.join(RAW_DATA_ROOT, "partition3"),
    "p4": os.path.join(RAW_DATA_ROOT, "partition4"),
    "p5": os.path.join(RAW_DATA_ROOT, "partition5"),
}

# ------------------------------------------------------------------
# DATA SHAPE
# ------------------------------------------------------------------
N_TIMESTEPS = 60          # SWAN-SF: 60 records per instance (12-min cadence, 12hr window)

# Raw SWAN-SF flare classes -> binary mapping
# Positive (major flare): X, M   |  Negative: C, B, N (flare-quiet)
POSITIVE_CLASSES = {"X", "M"}
NEGATIVE_CLASSES = {"C", "B", "N", "FQ"}

# ------------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------------
Z_OUTLIER_THRESH = 7.0    # relaxed from 4.0 — SHARP magnetic parameters (TOTUSJH,
                          # USFLUX, TOTPOT, etc.) are often genuinely extreme right
                          # before a flare; capping at 4-sigma to the median was
                          # likely destroying real predictive signal, not just noise
CORR_DROP_THRESH = 0.97    # relaxed from 0.95 — dropping too aggressively at 0.95
                            # can throw away real discriminative signal; check how
                            # many features survive selection and tune from there

# ------------------------------------------------------------------
# SAMPLING / CLASS BALANCE
# ------------------------------------------------------------------
# SMOTE creates SYNTHETIC minority-class sequences by interpolating between
# real ones. For time-series like SWAN-SF, that interpolation is done
# independently per timestep/feature and can produce sequences that don't
# look like anything a real flare/quiet-region evolution looks like — the
# model then partly learns those synthetic patterns, which don't exist in
# the (naturally imbalanced) validation/test data. This is very likely a
# big part of why val_TSS was highest at epoch 1 and fell afterwards.
#
# Set to False to train on the REAL class distribution instead, relying
# only on class-weighted Focal Loss to handle the imbalance (no synthetic
# samples at all). Try this first.
USE_SMOTE_BALANCING = False

# Only used when USE_SMOTE_BALANCING = True.
SMOTE_TARGET_RATIO = 0.1   # reduced further — if you do re-enable SMOTE, keep
                            # the amount of synthetic data as small as possible
RUS_TARGET_RATIO = 0.2     # reduced further, for the same reason
USE_CLASS_WEIGHTED_LOSS = True
CB_BETA = 0.999   # effective-number-of-samples class-balance strength

# ------------------------------------------------------------------
# MODEL / TRAINING
# ------------------------------------------------------------------
TCN_CHANNELS = [32, 64, 64]     # reverted from [64, 128, 128] — the larger network
                                 # was overfitting (train_loss kept dropping while
                                 # val_TSS kept falling), so capacity is back down.
KERNEL_SIZE = 3
DROPOUT = 0.3                   # kept moderate for extra regularization
N_CLASSES = 2

# Focal Loss (instead of plain weighted CrossEntropy) — focuses training on
# hard/misclassified examples, often improves the precision/recall trade-off
# under extreme imbalance like SWAN-SF's ~50-60:1 ratio.
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.5

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-4     # increased L2 penalty for extra regularization
EARLY_STOP_PATIENCE = 100      # was 100 (>= EPOCHS, so it never actually triggered);
                               # now early stopping can actually kick in

# LR schedule. "plateau" (ReduceLROnPlateau) drops LR only when val_loss
# stalls — but if val_loss rises from very early on (as we saw), it can
# collapse the LR too fast and leave the model stuck. "cosine" decays LR
# on a fixed, predictable schedule over all EPOCHS regardless of val_loss
# behavior, which is more robust for this kind of noisy validation signal.
LR_SCHEDULER = "cosine"   # "cosine" or "plateau"

# Multi-task learning: weight of the auxiliary self-supervised sequence
# reconstruction loss, relative to the primary classification loss.
# total_loss = classification_loss + RECON_LOSS_WEIGHT * reconstruction_loss
# Start small (0.1) — the auxiliary task should regularize, not dominate.
RECON_LOSS_WEIGHT = 0.1

RANDOM_SEED = 42

DEVICE = "cuda"  # falls back to "cpu" automatically in code if unavailable
