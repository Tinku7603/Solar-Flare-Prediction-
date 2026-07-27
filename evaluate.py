"""
evaluate.py
-----------
Solar-flare-specific evaluation metrics (TSS, HSS) plus standard
classification metrics (accuracy, precision, recall, F1, AUC), robust
to SWAN-SF's extreme class imbalance.

Key addition: since TSS = TPR - FPR (the Youden's J statistic), it is
threshold-dependent. Rather than always using the default 0.5 decision
threshold (argmax), `find_best_threshold` scans thresholds on the
VALIDATION set to find the one that maximizes TSS there, and that
threshold is then applied to the held-out TEST set. This is a purely
post-hoc decision-boundary adjustment — no leakage, since the threshold
is chosen using only validation data.
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)


def true_skill_statistic(y_true, y_pred):
    """TSS = TPR - FPR. Standard skill metric in solar flare forecasting.
    Range: [-1, 1], 0 = no skill, 1 = perfect."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr - fpr


def heidke_skill_score(y_true, y_pred):
    """HSS: skill relative to random chance forecast. Range: (-inf, 1]."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    expected_correct = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n
    denom = n - expected_correct
    if denom == 0:
        return 0.0
    return (tp + tn - expected_correct) / denom


@torch.no_grad()
def run_inference(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for X, y in data_loader:
        X = X.to(device)
        logits, _, _ = model(X)   # (logits, attention_weights, reconstruction)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))


def find_best_threshold(y_true, y_prob, thresholds=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    if thresholds is None:
        thresholds = np.arange(0.01, 0.501, 0.005)

    records = []

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        recall = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        tss = recall - fpr
        precision = tp / (tp + fp + 1e-8)

        records.append({
            "threshold": th,
            "tss": tss,
            "fpr": fpr,
            "recall": recall,
            "precision": precision,
            "fp": fp,
            "fn": fn
        })

    best_tss = max(r["tss"] for r in records)

    # keep thresholds within 0.01 of best TSS,
    # then choose the one with lowest false positive rate
    candidates = [
        r for r in records
        if r["tss"] >= best_tss - 0.01
    ]

    best = sorted(
        candidates,
        key=lambda r: (r["fpr"], -r["precision"], -r["tss"])
    )[0]

    print(
        f"Best val TSS = {best_tss:.4f} | "
        f"Selected threshold = {best['threshold']:.3f} | "
        f"Selected val TSS = {best['tss']:.4f} | "
        f"FPR = {best['fpr']:.4f} | "
        f"Recall = {best['recall']:.4f} | "
        f"Precision = {best['precision']:.4f}"
    )

    return best["threshold"], best["tss"]

def evaluate(model, data_loader, device, partition_name="test", verbose=True, threshold=0.5):
    """
    threshold: decision threshold applied to P(positive). Default 0.5
    reproduces plain argmax behavior. Pass a validation-tuned threshold
    (from find_best_threshold) to evaluate with a TSS-optimized cutoff.
    """
    y_true, _, y_prob = run_inference(model, data_loader, device)
    y_pred = (y_prob >= threshold).astype(int)

    tss = true_skill_statistic(y_true, y_pred)
    hss = heidke_skill_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    results = {
        "partition": partition_name,
        "threshold": threshold,
        "TSS": tss,
        "HSS": hss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }

    if verbose:
        print(f"\n--- Evaluation on {partition_name} (threshold={threshold:.3f}) ---")
        print(f"TSS: {tss:.4f}  |  HSS: {hss:.4f}  |  AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}  |  Precision: {precision:.4f}  "
              f"|  Recall: {recall:.4f}  |  F1: {f1:.4f}")
        print(f"Confusion matrix [[TN, FP],[FN, TP]]:\n{cm}")

    return results


def summarize_cross_partition_results(all_results):
    """Aggregate all metrics across the rotating-partition CV folds."""
    metric_names = ["TSS", "HSS", "accuracy", "precision", "recall", "f1", "auc"]
    metric_vals = {m: [r[m] for r in all_results] for m in metric_names}

    print("\n===== Cross-Partition Summary =====")
    header = f"  {'Fold':>6s} | " + " | ".join(f"{m:>9s}" for m in metric_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        row = f"  {r['partition']:>6s} | " + " | ".join(f"{r[m]:9.4f}" for m in metric_names)
        print(row)

    print("  " + "-" * (len(header) - 2))
    summary = {}
    for m in metric_names:
        vals = [v for v in metric_vals[m] if not np.isnan(v)]
        mean_v = float(np.mean(vals)) if vals else float("nan")
        std_v = float(np.std(vals)) if vals else float("nan")
        summary[f"mean_{m}"] = mean_v
        summary[f"std_{m}"] = std_v
        print(f"  Mean {m:>9s}: {mean_v:.4f} +/- {std_v:.4f}")

    return summary
