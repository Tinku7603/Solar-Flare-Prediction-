"""
swan_sf_loader.py
------------------
Loader for the ACTUAL SWAN-SF raw data format, as released by GSU's DMLab
on Harvard Dataverse (doi:10.7910/DVN/EBCFKM).

REAL observed format (confirmed from actual downloaded files):
  - Each partition directory contains TWO subfolders: FL (flaring) and NF
    (non-flaring / flare-quiet).
  - Flaring instance filenames look like:
        M1.1@1436:Secondary_ar401_s2011-03-10T06:36:00_e2011-03-10T18:24:00.csv
        B4.3@1963:Primary_ar667_s2011-06-24T19:48:00_e2011-06-25T07:36:00.csv
    -> the flare class is the LEADING LETTERS of the filename (e.g. "M", "B").
  - Non-flaring instance filenames look like:
        FQ_ar1043_s2011-11-06T17:24:00_e2011-11-07T05:12:00.csv
    -> label is "FQ" (flare-quiet).
  - The active region id follows "ar" (e.g. ar401, ar1043).
  - Each instance file itself is TAB-delimited, first column "Timestamp",
    remaining columns are SHARP magnetic-field parameters.

This loader:
  1. Recursively finds every instance file under a partition directory
     (including the FL/ and NF/ subfolders).
  2. Extracts the flare-class label and active-region id directly from
     the filename (no bracket-tag parsing needed for this release).
  3. Reads the tab-delimited file contents.
  4. Pads/truncates every instance to config.N_TIMESTEPS rows.
  5. Caches the assembled (X, y, ids, feature_names) per partition as a
     pickle, since re-parsing tens of thousands of raw files is slow.
"""

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import N_TIMESTEPS, POSITIVE_CLASSES, NEGATIVE_CLASSES


# ------------------------------------------------------------------
# Filename parsing (real SWAN-SF format: label is a leading prefix,
# not a tag[value] pair)
# ------------------------------------------------------------------
def extract_label_from_filename(filename):
    """
    Extract the raw flare-class label from the leading letters of the
    filename, e.g.:
      'M1.1@1436:Secondary_ar401_s..._e....csv' -> 'M'
      'FQ_ar1043_s..._e....csv'                  -> 'FQ'
      'B4.3@1963:Primary_ar667_s..._e....csv'    -> 'B'
    Returns None if no leading letters are found.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^([A-Za-z]+)', name)
    return m.group(1) if m else None


def extract_ar_id_from_filename(filename):
    """Extract the active-region id following 'ar' in the filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'ar(\d+)', name)
    return m.group(1) if m else name


def binarize_label(raw_label):
    """Map a raw SWAN-SF flare-class string to binary: 1 = major flare (X/M), 0 = otherwise."""
    v = str(raw_label).upper().strip()
    if v in POSITIVE_CLASSES:
        return 1
    if v in NEGATIVE_CLASSES:
        return 0
    # Some releases encode flare class as e.g. 'M1.2' -> take leading letter
    if v and v[0] in POSITIVE_CLASSES:
        return 1
    if v and v[0] in NEGATIVE_CLASSES:
        return 0
    return 0  # conservative fallback for unrecognized labels


# ------------------------------------------------------------------
# Reading a single instance file
# ------------------------------------------------------------------
def read_instance_file(filepath, timestamp_col_candidates=('Timestamp', 'timestamp', 'TIME')):
    """
    Read one tab-delimited SWAN-SF instance file.
    Returns (values: ndarray (t, n_features), feature_names: list[str])
    """
    df = pd.read_csv(filepath, sep='\t')

    ts_col = None
    for cand in timestamp_col_candidates:
        if cand in df.columns:
            ts_col = cand
            break

    feature_cols = [c for c in df.columns if c != ts_col]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df[feature_cols].values.astype(np.float32), feature_cols


# ------------------------------------------------------------------
# Loading a full partition directory
# ------------------------------------------------------------------
def find_instance_files(partition_dir, extensions=('.csv', '.tab', '.txt')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(partition_dir, '**', f'*{ext}'), recursive=True))
    return sorted(set(files))


def load_raw_partition(partition_dir, n_timesteps=N_TIMESTEPS, show_progress=True):
    """
    Load every instance file in `partition_dir` (searched recursively,
    including FL/ and NF/ subfolders).

    Returns:
        X: ndarray (n_instances, n_timesteps, n_features)
        y: ndarray (n_instances,) binary labels
        ids: list[str] active-region / instance identifiers (best-effort)
        feature_names: list[str] common feature columns kept across all instances
    """
    files = find_instance_files(partition_dir)
    if not files:
        raise FileNotFoundError(
            f"No instance files found under {partition_dir}. "
            f"Expected tab-delimited .csv/.tab/.txt files under FL/ and NF/ subfolders."
        )

    all_values = []
    all_labels = []
    all_ids = []
    reference_features = None
    skipped = 0

    iterator = tqdm(files, desc=f"Loading {os.path.basename(partition_dir.rstrip('/'))}") \
        if show_progress else files

    for fpath in iterator:
        try:
            raw_label = extract_label_from_filename(fpath)
            if raw_label is None:
                skipped += 1
                continue

            values, feat_names = read_instance_file(fpath)

            if reference_features is None:
                reference_features = feat_names
            elif feat_names != reference_features:
                common = [f for f in reference_features if f in feat_names]
                if not common:
                    skipped += 1
                    continue
                idx = [feat_names.index(f) for f in common]
                values = values[:, idx]
                reference_features = common

            t = values.shape[0]
            if t < n_timesteps:
                pad = np.full((n_timesteps - t, values.shape[1]), np.nan, dtype=np.float32)
                values = np.vstack([values, pad])
            elif t > n_timesteps:
                values = values[:n_timesteps, :]

            all_values.append(values)
            all_labels.append(binarize_label(raw_label))
            all_ids.append(extract_ar_id_from_filename(fpath))

        except Exception as e:
            skipped += 1
            continue

    if not all_values:
        raise RuntimeError(f"Failed to load any valid instances from {partition_dir}.")

    n_features = len(reference_features)
    X = np.stack([
        v if v.shape[1] == n_features else v[:, :n_features]
        for v in all_values
    ]).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    if skipped:
        print(f"  ({skipped} files skipped: unparseable filename, missing label, or read error)")
    pos = int(y.sum())
    print(f"  Loaded {X.shape[0]} instances, {X.shape[2]} features "
          f"[{pos} positive / {len(y) - pos} negative]")

    return X, y, all_ids, reference_features


# ------------------------------------------------------------------
# Caching so you don't re-parse tens of thousands of raw files every run
# ------------------------------------------------------------------
def load_raw_partition_cached(partition_dir, cache_path, n_timesteps=N_TIMESTEPS,
                               force_reload=False, show_progress=True):
    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached partition -> {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y'], data['ids'], data['feature_names']

    X, y, ids, feature_names = load_raw_partition(
        partition_dir, n_timesteps=n_timesteps, show_progress=show_progress
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'ids': ids, 'feature_names': feature_names}, f)
    print(f"  Cached -> {cache_path}")

    return X, y, ids, feature_names


def load_all_raw_partitions(partition_dirs, cache_dir, n_timesteps=N_TIMESTEPS,
                             force_reload=False):
    """
    partition_dirs: dict {partition_name: path_to_partition_directory}
    Returns: dict {partition_name: (X, y, feature_names)}
    """
    data = {}
    for name, path in partition_dirs.items():
        cache_path = os.path.join(cache_dir, f"{name}.pkl")
        X, y, ids, feature_names = load_raw_partition_cached(
            path, cache_path, n_timesteps=n_timesteps, force_reload=force_reload
        )
        data[name] = (X, y, feature_names)
    return data
