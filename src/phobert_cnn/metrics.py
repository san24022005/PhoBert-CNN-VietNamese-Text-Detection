from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "report": classification_report(labels, predictions, zero_division=0),
    }
