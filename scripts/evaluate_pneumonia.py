"""One-pass evaluation on the publisher-defined, held-out test split."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

from visionx.findings import load_findings_model
from scripts.train_pneumonia import load_split


@torch.inference_mode()
def main():
    data = np.load("data/pneumoniamnist_224.npz")
    loader = DataLoader(load_split(data, "test"), batch_size=64, shuffle=False)
    model, checkpoint = load_findings_model()
    labels, scores = [], []
    for x, y in loader:
        scores.extend(torch.sigmoid(model(x)).numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())
    y, p = np.asarray(labels), np.asarray(scores)
    pred = (p >= checkpoint["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "dataset": "PneumoniaMNIST+ 224", "split": "official held-out test",
        "n": int(len(y)), "threshold": checkpoint["threshold"],
        "accuracy": float(accuracy_score(y, pred)), "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "sensitivity": float(recall_score(y, pred)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred)), "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": {"true_normal": int(tn), "false_pneumonia": int(fp),
                             "missed_pneumonia": int(fn), "true_pneumonia": int(tp)},
        "warning": "Research benchmark only. Pediatric dataset; not a diagnostic device or for clinical use.",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/pneumonia_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
