"""Lesson 4: validation chooses settings; the locked test partition reports quality."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

from .common import REASON_LABELS, synthetic, write_csv, write_json


def choose_threshold(expected, probabilities, classes, target_accuracy=.9, known_mask=None):
    expected = np.asarray(expected)
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    scores = probabilities.max(axis=1)
    known = np.ones(len(expected), dtype=bool) if known_mask is None else np.asarray(known_mask, dtype=bool)
    curve = []
    for threshold in (.35, .45, .55, .65, .75, .85, .95):
        used = (scores >= threshold) & known
        curve.append(dict(threshold=threshold, accepted=int(used.sum()), coverage=float(used.mean()),
                          accepted_accuracy=float((predictions[used] == expected[used]).mean()) if used.any() else None))
    eligible = [point for point in curve if point["accepted"] >= 5 and point["accepted_accuracy"] >= target_accuracy]
    selected = max(eligible, key=lambda point: point["coverage"])["threshold"] if eligible else 1.0
    return selected, curve


def evaluate(rows, probabilities, classes, threshold, known_mask=None):
    expected = [row["reason"] for row in rows]
    predicted = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
    known = np.ones(len(rows), dtype=bool) if known_mask is None else np.asarray(known_mask, dtype=bool)
    predicted[~known] = "__no_prediction__"
    used = (probabilities.max(axis=1) >= threshold) & known
    matrix_labels = list(classes) + (["__no_prediction__"] if not known.all() else [])
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped["synthetic" if synthetic(row) else "real"].append(i)
        grouped["channel:" + row.get("source", "unknown")].append(i)
    result = dict(examples=len(rows), accuracy=float(accuracy_score(expected, predicted)),
                  macro_f1=float(f1_score(expected, predicted, labels=REASON_LABELS, average="macro", zero_division=0)),
                  log_loss=float(log_loss(np.asarray(expected)[known], probabilities[known], labels=classes)) if known.any() else None,
                  prediction_missing=int((~known).sum()),
                  per_class=classification_report(expected, predicted, labels=list(classes), output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(expected, predicted, labels=matrix_labels).tolist(),
                  class_order=matrix_labels, threshold=threshold, accepted=int(used.sum()), coverage=float(used.mean()),
                  accepted_accuracy=float(accuracy_score(np.asarray(expected)[used], predicted[used])) if used.any() else None,
                  slices={key: dict(examples=len(ids), accuracy=float(accuracy_score(np.asarray(expected)[ids], predicted[ids])))
                          for key, ids in grouped.items()},
                  warning="Scores are uncalibrated. Synthetic/test-set performance does not establish production quality.")
    return result


def save_report(output, report, rows, probabilities, classes, known_mask=None):
    from pathlib import Path
    output = Path(output)
    write_json(output / "evaluation.json", report)
    predictions = []
    known = np.ones(len(rows), dtype=bool) if known_mask is None else np.asarray(known_mask, dtype=bool)
    for row, scores, has_features in zip(rows, probabilities, known):
        index = int(scores.argmax())
        predicted = classes[index] if has_features else "__no_prediction__"
        predictions.append(dict(id=row["id"], expected=row["reason"], predicted=predicted,
                                score=round(float(scores[index]), 6) if has_features else 0., correct=int(row["reason"] == predicted)))
    write_csv(output / "test_predictions.csv", predictions, ("id", "expected", "predicted", "score", "correct"))
    test = report["test"]
    lines = ["# Reason-model experiment", "", "Synthetic starter results are a pipeline exercise, not real-customer accuracy.", "",
             f"- Selected settings: `{report['selected']}` (validation only).",
             f"- Untouched test: {test['examples']} rows; macro F1 {test['macro_f1']:.3f}; accuracy {test['accuracy']:.3f}.",
             f"- Coverage at score threshold {test['threshold']}: {test['coverage']:.1%} ({test['accepted']} accepted).",
             f"- Accuracy among accepted predictions: {test['accepted_accuracy']} (null means none accepted).",
             "- The saved candidate contains only training rows. Validation and test rows were never refitted into it.",
             "- Selective accuracy is a small validation-sample observation, not a performance guarantee.", "",
             "## Per-class test results", "", "| Reason | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in classes:
        row = test["per_class"][label]
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |")
    lines += ["", "See evaluation.json for the confusion matrix, threshold curve, provenance and channel/origin slices.",
              "Review errors by joining test_predictions.csv IDs to the local test CSV. Do not publish private customer text."]
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
