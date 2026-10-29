import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from preprocess_data import create_data_generators


def roc_auc(y_true, scores):
    # Rank-based (Mann-Whitney U) AUC with average ranks for ties.
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def safe_div(a, b):
    return float(a / b) if b else 0.0


def compute_metrics(y_true, prob_positive, threshold=0.5):
    y_pred = (prob_positive >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    f1_neg = safe_div(2 * npv * specificity, npv + specificity)

    return {
        "threshold": threshold,
        "accuracy": safe_div(tp + tn, len(y_true)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1_score": f1,
        "macro_f1": (f1 + f1_neg) / 2,
        "roc_auc": roc_auc(y_true, prob_positive),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_split(model, generator, positive_label="fractured"):
    probs = model.predict(generator, verbose=1).ravel()
    labels = generator.classes
    pos_index = generator.class_indices[positive_label]

    # The sigmoid output is P(class index 1); convert it to P(positive_label).
    prob_positive = probs if pos_index == 1 else 1.0 - probs
    y_true = (labels == pos_index).astype(int)

    result = compute_metrics(y_true, prob_positive)
    result["samples"] = int(len(labels))
    result["class_counts"] = {
        name: int((labels == idx).sum()) for name, idx in generator.class_indices.items()
    }
    return result


if __name__ == "__main__":
    model_dir = Path(__file__).resolve().parent
    dataset_folder = model_dir.parent / "dataset"
    _, val_generator, test_generator = create_data_generators(dataset_folder)

    model = load_model(model_dir / "bone_fracture_model.h5")

    metrics = {
        "model": "MobileNetV2 transfer learning (frozen base, 128-64-1 head)",
        "model_file": "bone_fracture_model.h5",
        "input_size": [224, 224, 3],
        "positive_class": "fractured",
        "class_indices": test_generator.class_indices,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation": evaluate_split(model, val_generator),
        "test": evaluate_split(model, test_generator),
    }

    output_path = model_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {output_path}")
