"""
preprocessing.py
-----------------
Imputation, outlier handling, correlation-based feature selection,
and normalization for SWAN-SF multivariate time series.

Memory-safe version:
  - impute_missing / cap_outliers mutate arrays IN PLACE (no defensive
    .copy()).
  - cap_outliers computes stats ONE FEATURE COLUMN AT A TIME instead of
    vectorizing over the whole (n_samples, n_features) matrix at once —
    this avoids allocating a full-size temporary array during mean/std
    computation, which was a major peak-memory contributor given
    SWAN-SF's large concatenated training sets.

All "fit" steps (feature selection, scaler) must be fit ONLY on the
training partition and then applied to val/test to avoid temporal leakage.

NOTE: functions here mutate their input in place where noted. Callers
(main.py) should not rely on the original array remaining unchanged, and
should process train/val/test one at a time, deleting each stale
reference as soon as it's no longer needed, in the SAME scope that holds
the original reference (deleting inside a helper function does NOT free
memory if the caller still holds its own reference to the same object).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Z_OUTLIER_THRESH, CORR_DROP_THRESH


def impute_missing(X):
    """
    Linear interpolation along the time axis for each (instance, feature),
    then fill any still-missing values (e.g. all-NaN series) with the
    global per-feature mean.

    VECTORIZED VERSION: uses pandas' vectorized interpolate() instead of
    Python-level nested loops, since the nested-loop version became too
    slow once rate-of-change features doubled the feature count.

    Returns a NEW array (does not mutate in place). Caller should
    reassign: X = impute_missing(X)
    X: ndarray (n_instances, n_timesteps, n_features)
    """
    import pandas as pd

    n_inst, n_time, n_feat = X.shape

    X_rows = X.transpose(0, 2, 1).reshape(n_inst * n_feat, n_time)

    df = pd.DataFrame(X_rows)
    df = df.interpolate(axis=1, limit_direction="both")
    X_rows = df.values.astype(X.dtype)

    X_out = X_rows.reshape(n_inst, n_feat, n_time).transpose(0, 2, 1)

    flat_view = X_out.reshape(-1, n_feat)
    with np.errstate(invalid="ignore"):
        col_has_data = ~np.isnan(flat_view).all(axis=0)
        global_means = np.zeros(n_feat, dtype=X_out.dtype)
        if col_has_data.any():
            global_means[col_has_data] = np.nanmean(flat_view[:, col_has_data], axis=0)

    for f in range(n_feat):
        mask = np.isnan(X_out[:, :, f])
        if mask.any():
            X_out[:, :, f][mask] = global_means[f]

    return X_out

def cap_outliers(X, z_thresh=Z_OUTLIER_THRESH):
    """
    Cap per-feature outliers beyond z_thresh standard deviations to the
    median of that feature. Mutates X in place and returns it.

    Processes ONE FEATURE COLUMN AT A TIME (not the whole matrix at once)
    to keep temporary array sizes small — SWAN-SF features like TOTPOT
    can reach ~1e24 in magnitude, and squaring a full (n_samples, n_feat)
    matrix at once both wastes memory and risks numeric overflow.
    """
    n_inst, n_time, n_feat = X.shape
    flat = X.reshape(-1, n_feat)  # view, not a copy

    for f in range(n_feat):
        col = flat[:, f].astype(np.float64)  # small: one column only
        mean = col.mean()
        std = col.std()
        if std == 0 or np.isnan(std):
            std = 1.0
        median = np.median(col)

        z = (col - mean) / std
        mask = np.abs(z) > z_thresh
        if mask.any():
            flat[mask, f] = median
        del col, z, mask

    return X


def select_features(X_train, feature_names, corr_thresh=CORR_DROP_THRESH):
    """
    Drop one feature from any pair with |correlation| > corr_thresh,
    computed on the TRAIN set only.

    Also prints:
      - dropped feature
      - correlated retained feature
      - correlation coefficient
    """

    n_inst, n_time, n_feat = X_train.shape
    flat = X_train.reshape(-1, n_feat)

    # Compute correlation matrix
    corr = np.corrcoef(flat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    to_drop = set()
    drop_reason = {}

    for i in range(n_feat):

        if i in to_drop:
            continue

        for j in range(i + 1, n_feat):

            if j in to_drop:
                continue

            corr_value = corr[i, j]

            if abs(corr_value) > corr_thresh:

                to_drop.add(j)

                drop_reason[j] = {
                    "dropped": feature_names[j],
                    "correlated_with": feature_names[i],
                    "correlation": corr_value
                }

    keep_idx = [
        i for i in range(n_feat)
        if i not in to_drop
    ]

    kept_names = [
        feature_names[i]
        for i in keep_idx
    ]

    dropped_names = [
        feature_names[i]
        for i in sorted(to_drop)
    ]

    print(
        f"\nFeature selection: kept {len(keep_idx)}/{n_feat} features "
        f"(dropped {len(to_drop)} highly correlated)."
    )

    if dropped_names:

        print("\nHighly correlated features:\n")

        for idx in sorted(to_drop):

            info = drop_reason[idx]

            print(
                f"  {info['dropped']}  <-->  "
                f"{info['correlated_with']}  |  "
                f"Correlation = {info['correlation']:.4f}"
            )

        print("\nDropped features:")
        print(dropped_names)

    return keep_idx, kept_names


def apply_feature_selection(X, keep_idx):
    """Returns a new (smaller) array — caller should drop the input
    reference right after calling this."""
    return np.ascontiguousarray(X[:, :, keep_idx])


def fit_scaler(X_train):
    """Fit a StandardScaler on the training partition only."""
    n_inst, n_time, n_feat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    return scaler


def apply_scaler(X, scaler):
    """Returns a new scaled array — caller should drop the input
    reference right after calling this."""
    n_inst, n_time, n_feat = X.shape
    X_scaled = scaler.transform(X.reshape(-1, n_feat)).astype(np.float32)
    return X_scaled.reshape(n_inst, n_time, n_feat)

def add_rate_of_change_features(X):
    """
    Appends first-difference (rate-of-change) channels to the input MVTS.
    For each feature f and timestep t: roc[t, f] = X[t, f] - X[t-1, f]
    (roc[0, f] is set to 0, since there's no t-1 to diff against).

    Doubles the feature dimension: (n_inst, n_time, n_feat) ->
    (n_inst, n_time, 2*n_feat), with the original values in the first
    half of the feature axis and their derivatives in the second half.

    Call this AFTER impute_missing/cap_outliers (so derivatives are
    computed on cleaned data) and BEFORE select_features/fit_scaler
    (so correlation-based selection and scaling see the derivative
    channels too, on the same footing as the original features).
    """
    n_inst, n_time, n_feat = X.shape

    roc = np.zeros_like(X)
    roc[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    # roc[:, 0, :] stays 0 -- no prior timestep to diff against

    return np.concatenate([X, roc], axis=2)

