"""
sampling.py
-----------
Class-balancing for SWAN-SF's extreme imbalance (major flares are rare).

Strategy (memory-safe order):
  1. Pre-undersample the majority class down to a reasonable multiple of
     the minority count (PRE_SMOTE_MAJORITY_MULTIPLIER). Doing this BEFORE
     SMOTE is critical: with SWAN-SF's ~50-60:1 imbalance, running SMOTE
     straight on the full majority class would require synthesizing tens
     of thousands of minority samples to reach SMOTE_TARGET_RATIO, which
     can exhaust RAM on a standard Colab runtime.
  2. SMOTE-oversample the (now smaller) minority class up to SMOTE_TARGET_RATIO.
  3. Random-undersample the majority class down to RUS_TARGET_RATIO.

Only ever apply this to the TRAINING set. Never oversample/undersample
validation or test data.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import SMOTE_TARGET_RATIO, RUS_TARGET_RATIO, RANDOM_SEED
import config

# Cap on how large the majority class is allowed to be BEFORE SMOTE runs,
# expressed as a multiple of the minority count. Lower = less RAM used,
# but too low risks discarding too much real majority-class signal before
# SMOTE/RUS get a chance to work. 15-20x is a reasonable default for
# SWAN-SF's scale.
PRE_SMOTE_MAJORITY_MULTIPLIER = 8


def balance_mvts(X, y, smote_ratio=SMOTE_TARGET_RATIO, rus_ratio=RUS_TARGET_RATIO,
                  pre_smote_multiplier=PRE_SMOTE_MAJORITY_MULTIPLIER):
    """
    X: ndarray (n_instances, n_timesteps, n_features)
    y: ndarray (n_instances,) binary labels
    Returns balanced (X_bal, y_bal), same shape convention.
    """
    n_inst, n_time, n_feat = X.shape
    X_flat = X.reshape(n_inst, -1)

    pos = int(y.sum())
    neg = len(y) - pos
    print(f"Before balancing: {pos} positive / {neg} negative "
          f"(ratio={pos / max(neg, 1):.4f})")

    if pos == 0:
        raise ValueError("No positive samples in training set — cannot balance.")

    # ---- Step 1: pre-undersample majority BEFORE SMOTE (memory guard) ----
    target_majority = min(neg, pos * pre_smote_multiplier)
    if target_majority < neg:
        pre_rus = RandomUnderSampler(
            sampling_strategy={0: target_majority, 1: pos},
            random_state=RANDOM_SEED,
        )
        X_flat, y = pre_rus.fit_resample(X_flat, y)
        neg = target_majority
        print(f"Pre-undersampled majority to {neg} (x{pre_smote_multiplier} minority) "
              f"before SMOTE to control memory use.")

    # Guard: SMOTE requires n_neighbors < n_minority_samples
    k_neighbors = min(5, max(1, pos - 1))

    try:
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_SEED,
                       k_neighbors=k_neighbors)
        X_over, y_over = smote.fit_resample(X_flat, y)
    except ValueError as e:
        print(f"SMOTE skipped ({e}); proceeding without oversampling.")
        X_over, y_over = X_flat, y

    del X_flat

    rus = RandomUnderSampler(sampling_strategy=rus_ratio, random_state=RANDOM_SEED)
    X_bal, y_bal = rus.fit_resample(X_over, y_over)

    del X_over, y_over

    X_bal = X_bal.reshape(-1, n_time, n_feat).astype(np.float32)

    pos_b = int(y_bal.sum())
    neg_b = len(y_bal) - pos_b
    print(f"After balancing:  {pos_b} positive / {neg_b} negative "
          f"(ratio={pos_b / max(neg_b, 1):.4f})")

    return X_bal, y_bal


def compute_class_weights(y, beta=None):
    """
    Class-Balanced weights based on Effective Number of Samples
    (Cui et al., 2019 - https://arxiv.org/abs/1901.05555).

    Replaces plain inverse-frequency weighting. Effective number of
    samples for a class: E_n = (1 - beta^n) / (1 - beta), where n is
    the class count. Weight = 1 / E_n, then normalized so weights
    sum to n_classes (keeps loss scale comparable to before).

    beta controls how strongly re-weighting kicks in:
      beta=0        -> no re-weighting (same as uniform weights)
      beta->1       -> approaches inverse-frequency weighting
      0.99 - 0.9999 -> the useful range for extreme imbalance; higher
                       beta = more aggressive re-weighting toward rare class
    Default here (0.999) is a reasonable starting point for SWAN-SF's
    ~50-60:1 ratio -- gentler than raw inverse-frequency, less likely
    to cause unstable/high-variance gradient updates than the old
    compute_class_weights did.
    """
    if beta is None:
        beta = getattr(config, "CB_BETA", 0.999)

    classes, counts = np.unique(y, return_counts=True)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(classes)  # normalize, sum = n_classes

    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Class-balanced weights (beta={beta}): {weight_dict}")
    return weight_dict