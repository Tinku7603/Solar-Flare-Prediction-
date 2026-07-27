"""
main.py
-------
End-to-end SWAN-SF pipeline: preprocessing -> sampling -> TCN+Attention
model -> evaluation, using the standard rotating-partition protocol
(train on 4 partitions, test on the held-out one, repeat for all 5).

TSS-focused version:
  - During training, the "best" checkpoint is selected by VALIDATION TSS
    (not validation loss). Loss and TSS don't always move together under
    heavy class imbalance, and TSS is the metric that actually matters
    for this task.
  - After training, a decision threshold is tuned on the validation set
    to maximize TSS there (TSS = TPR - FPR = Youden's J, which is
    threshold-dependent), and that same threshold is then applied when
    evaluating on the held-out TEST partition. No leakage: threshold is
    chosen using validation data only.

Memory-safe: partitions are loaded from cache fresh per fold, and large
arrays are freed as soon as they're no longer needed.

Run:
    python main.py
"""

import os
import gc
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

import config
from swan_sf_loader import load_raw_partition_cached
from preprocessing import (
    impute_missing, cap_outliers, select_features,
    apply_feature_selection, fit_scaler, apply_scaler,
    add_rate_of_change_features,
)
from sampling import balance_mvts, compute_class_weights
from model import build_model
from evaluate import (
    evaluate, summarize_cross_partition_results,
    true_skill_statistic, run_inference, find_best_threshold,
)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017), adapted for extreme class imbalance.
    Down-weights easy, already-well-classified examples and focuses
    training on hard/misclassified ones — often gives a better
    precision/recall trade-off than plain class-weighted cross-entropy
    on severely imbalanced data like SWAN-SF.

    loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # tensor of per-class weights, or None
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_partition(name):
    cache_path = os.path.join(config.CACHE_DIR, f"{name}.pkl")
    X, y, ids, feature_names = load_raw_partition_cached(
        config.PARTITION_DIRS[name], cache_path, n_timesteps=config.N_TIMESTEPS
    )
    return X, y, feature_names


def preprocess_one(X, keep_idx, scaler):
    X = impute_missing(X)
    X = cap_outliers(X)
    X = add_rate_of_change_features(X)
    X_sel = apply_feature_selection(X, keep_idx)
    del X
    gc.collect()
    X_final = apply_scaler(X_sel, scaler)
    del X_sel
    gc.collect()
    return X_final


def train_one_fold(X_train, y_train, X_val, y_val, n_features, device, class_weight_dict=None, fold_name="fold"):
    n_timesteps = X_train.shape[1]
    model = build_model(n_features, n_timesteps, device)

    if config.USE_CLASS_WEIGHTED_LOSS and class_weight_dict is not None:
        weight_tensor = torch.tensor(
            [class_weight_dict.get(0, 1.0), class_weight_dict.get(1, 1.0)],
            dtype=torch.float32,
        ).to(device)
    else:
        weight_tensor = None

    if getattr(config, "USE_FOCAL_LOSS", False):
        ce_criterion = FocalLoss(alpha=weight_tensor, gamma=config.FOCAL_GAMMA)
    elif weight_tensor is not None:
        ce_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        ce_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    scheduler_type = getattr(config, "LR_SCHEDULER", "plateau")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    train_loader = make_loader(X_train, y_train, config.BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, config.BATCH_SIZE, shuffle=False)

    best_val_tss = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss, train_ce, train_recon = 0.0, 0.0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _, reconstruction = model(X_batch)
            ce_loss = ce_criterion(logits, y_batch)
            recon_loss = recon_criterion(reconstruction, X_batch)
            loss = ce_loss + config.RECON_LOSS_WEIGHT * recon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_ce += ce_loss.item() * X_batch.size(0)
            train_recon += recon_loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_ce /= len(train_loader.dataset)
        train_recon /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_val_probs, all_val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _, reconstruction = model(X_batch)
                ce_loss = ce_criterion(logits, y_batch)
                recon_loss = recon_criterion(reconstruction, X_batch)
                loss = ce_loss + config.RECON_LOSS_WEIGHT * recon_loss
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(y_batch.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_labels_arr = np.array(all_val_labels)
        val_probs_arr = np.array(all_val_probs)
        # NOTE: train set is SMOTE/RUS-balanced but val set keeps the natural
        # imbalance, so a fixed 0.5 argmax threshold under-represents the
        # model's real separating power and drifts as training progresses.
        # Tune the threshold on validation probabilities each epoch instead,
        # so checkpoint selection reflects the model's underlying skill.
        _, val_tss = find_best_threshold(val_labels_arr, val_probs_arr)
        # AUC is threshold-independent — track it alongside TSS. If AUC keeps
        # improving while val_TSS falls, that points to a calibration/
        # threshold issue rather than the model actually getting worse.
        try:
            val_auc = roc_auc_score(val_labels_arr, val_probs_arr)
        except ValueError:
            val_auc = float("nan")  # only one class present in this val batch set

        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(f"  Epoch {epoch:3d}/{config.EPOCHS}  "
              f"train_loss={train_loss:.4f} (ce={train_ce:.4f} recon={train_recon:.4f})  "
              f"val_loss={val_loss:.4f}  val_TSS={val_tss:.4f}  val_AUC={val_auc:.4f}")

        if val_tss > best_val_tss:
            best_val_tss = val_tss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no val TSS improvement "
                      f"for {config.EARLY_STOP_PATIENCE} epochs). Best val_TSS={best_val_tss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    del train_loader, optimizer, scheduler, ce_criterion, recon_criterion
    gc.collect()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["val_tss"], label="val_TSS", color="green")
    axes[1].plot(history["epoch"], history["val_auc"], label="val_AUC", color="purple")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, f"curves_fold_{fold_name}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves -> {plot_path}")

    return model, val_loader


def run_rotating_partition_cv():
    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    partition_names = list(config.PARTITION_DIRS.keys())

    print("\n=== Pre-caching all partitions (raw SWAN-SF format) ===")
    for name in partition_names:
        X, y, _ = load_partition(name)
        del X, y
    gc.collect()

    all_results = []

    for test_name in partition_names:
        ckpt_path = os.path.join(config.OUTPUT_DIR, f"model_fold_{test_name}.pt")
        if os.path.exists(ckpt_path):
            print(f"\nFold '{test_name}' already has a saved checkpoint "
                  f"({ckpt_path}) — skipping (delete the file to re-run this fold).")
            continue

        print(f"\n============================================")
        print(f"  Fold: hold out '{test_name}' as TEST partition")
        print(f"============================================")

        train_names = [p for p in partition_names if p != test_name]
        val_name = train_names[0]
        fit_train_names = train_names[1:]

        print("\n--- Loading + preprocessing TRAIN partitions (from cache) ---")
        train_Xs, train_ys = [], []
        feature_names = None
        for name in fit_train_names:
            X, y, feature_names = load_partition(name)
            train_Xs.append(X)
            train_ys.append(y)

        X_train_raw = np.concatenate(train_Xs, axis=0)
        y_train_raw = np.concatenate(train_ys, axis=0)
        del train_Xs, train_ys
        gc.collect()

        X_train_raw = impute_missing(X_train_raw)
        X_train_raw = cap_outliers(X_train_raw)
        X_train_raw = add_rate_of_change_features(X_train_raw)
        feature_names = feature_names + [f"{f}_roc" for f in feature_names]

        keep_idx, kept_feats = select_features(X_train_raw, feature_names)
        X_train_sel = apply_feature_selection(X_train_raw, keep_idx)
        del X_train_raw
        gc.collect()

        scaler = fit_scaler(X_train_sel)
        X_train_p = apply_scaler(X_train_sel, scaler)
        del X_train_sel
        gc.collect()

        print("\n--- Loading + preprocessing VAL partition ---")
        X_val_raw, y_val_raw, _ = load_partition(val_name)
        X_val_p = preprocess_one(X_val_raw, keep_idx, scaler)
        del X_val_raw
        gc.collect()

        print("--- Loading + preprocessing TEST partition ---")
        X_test_raw, y_test_raw, _ = load_partition(test_name)
        X_test_p = preprocess_one(X_test_raw, keep_idx, scaler)
        del X_test_raw
        gc.collect()

        if getattr(config, "USE_SMOTE_BALANCING", True):
            print("\n--- Balancing training set (SMOTE + undersampling) ---")
            X_train_bal, y_train_bal = balance_mvts(X_train_p, y_train_raw)
        else:
            # Train on the REAL class distribution — no synthetic samples.
            # Class imbalance is instead handled purely through
            # class-weighted Focal Loss (see USE_CLASS_WEIGHTED_LOSS /
            # USE_FOCAL_LOSS in config.py). This keeps the train distribution
            # aligned with val/test, avoiding the synthetic-pattern mismatch
            # SMOTE can introduce on multivariate time series.
            print("\n--- Skipping SMOTE/undersampling (USE_SMOTE_BALANCING=False) "
                  "— training on real class distribution ---")
            X_train_bal, y_train_bal = X_train_p, y_train_raw
        del X_train_p, y_train_raw
        gc.collect()
        class_weights = compute_class_weights(y_train_bal)

        print("\n--- Training (model selection by validation TSS) ---")
        n_features = X_train_bal.shape[2]
        model, val_loader = train_one_fold(
            X_train_bal, y_train_bal, X_val_p, y_val_raw,
            n_features, device, class_weight_dict=class_weights, fold_name=test_name,
        )
        del X_train_bal, y_train_bal
        gc.collect()

        # ---- Tune decision threshold on VALIDATION set (maximize TSS) ----
        val_true, _, val_prob = run_inference(model, val_loader, device)
        best_threshold, best_val_tss = find_best_threshold(val_true, val_prob)
        print(f"\nValidation-tuned threshold: {best_threshold:.3f} "
              f"(val TSS at this threshold = {best_val_tss:.4f})")
        del val_loader, X_val_p, y_val_raw
        gc.collect()

        # ---- Evaluation on held-out test partition, using tuned threshold ----
        test_loader = make_loader(X_test_p, y_test_raw, config.BATCH_SIZE, shuffle=False)
        result = evaluate(model, test_loader, device, partition_name=test_name,
                           threshold=best_threshold)
        all_results.append(result)

        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint -> {ckpt_path}")

        del X_test_p, y_test_raw, test_loader, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_cross_partition_results(all_results)

    results_path = os.path.join(config.OUTPUT_DIR, "cv_results.json")
    with open(results_path, "w") as f:
        json.dump({"folds": all_results, "summary": summary}, f, indent=2)
    print(f"\nSaved full results -> {results_path}")

    return all_results, summary


# NOTE: the `if __name__ == "__main__":` auto-run guard was removed on
# purpose. If this file is pasted straight into a notebook cell, that
# guard evaluates to True (a notebook cell's __name__ is "__main__") and
# the whole pipeline fires immediately when you run the cell — which is
# what was happening. Call it explicitly, in its own cell, instead:
#
#   from main import run_rotating_partition_cv
#   all_results, summary = run_rotating_partition_cv()
#
# (If you later use %%writefile main.py + import main, this file's
# __name__ becomes "main", not "__main__", so the guard would have been
# safe in that flow too — but leaving it out avoids the footgun either way.)

if __name__ == "__main__":
    all_results, summary = run_rotating_partition_cv()
