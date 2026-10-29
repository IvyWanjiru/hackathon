"""Lesson 5: train a baseline, tune on validation, test once, export app-compatible JSON."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from demand_ml import ReasonClassifier, VECTORIZER_PARAMS
from .common import load_partitions, provenance, synthetic, write_json
from .evaluation import choose_threshold, evaluate, save_report


def train(directory, output):
    parts = load_partitions(directory)
    train, validation, test = [parts[name] for name in ("train", "validation", "test")]
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    x_train = vectorizer.fit_transform([row["text"] for row in train])
    x_validation = vectorizer.transform([row["text"] for row in validation])
    y_train = [row["reason"] for row in train]
    y_validation = [row["reason"] for row in validation]
    candidates = []
    best = None
    for c in (.5, 2.0, 8.0):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
        score = float(f1_score(y_validation, model.predict(x_validation), average="macro", zero_division=0))
        candidates.append(dict(C=c, validation_macro_f1=score))
        if best is None or score > best[0]:
            best = score, c, model
    _, c, model = best
    classes = model.classes_.tolist()
    validation_probabilities = model.predict_proba(x_validation)
    threshold, curve = choose_threshold(y_validation, validation_probabilities, classes,
                                         known_mask=x_validation.getnnz(axis=1) > 0)
    test_features = vectorizer.transform([row["text"] for row in test])
    probabilities = model.predict_proba(test_features)
    known_test = test_features.getnnz(axis=1) > 0
    metrics = evaluate(test, probabilities, classes, threshold, known_mask=known_test)
    sources = [Path(directory)/f"{name}.csv" for name in parts]
    record = provenance(sources)
    experiment = json.dumps(dict(files=record["sha256"], C=c, threshold=threshold), sort_keys=True)
    version = "tutorial-" + hashlib.sha256(experiment.encode()).hexdigest()[:12]
    train_synthetic = sum(synthetic(row) for row in train)
    test_synthetic = sum(synthetic(row) for row in test)
    metadata = dict(version=version, trained_at=datetime.now(timezone.utc).isoformat(),
                    examples=len(train), starter_examples=train_synthetic, reviewed_examples=len(train)-train_synthetic,
                    reviewed_through="", class_counts=dict(Counter(y_train)),
                    selected_threshold=threshold, provenance=record,
                    evaluation=dict(examples=len(test), starter_examples=test_synthetic, reviewed_examples=len(test)-test_synthetic,
                                    accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                                    method="Untouched grouped test; training partition only; synthetic limitations apply"))
    artifact = dict(format_version=1, metadata=metadata, classes=classes,
                    vectorizer_params={**VECTORIZER_PARAMS, "ngram_range": list(VECTORIZER_PARAMS["ngram_range"])},
                    vocabulary={key: int(value) for key, value in vectorizer.vocabulary_.items()},
                    idf=vectorizer.idf_.tolist(), weights=model.coef_.tolist(), intercept=model.intercept_.tolist())
    output = Path(output)
    target = output / "reason_classifier.json"
    # Validate before saving and check serving math equals sklearn's training math.
    checker = ReasonClassifier(target, threshold=threshold)
    checker._decode(artifact)
    write_json(target, artifact)
    for row, scores, feature in zip(test, probabilities, test_features):
        prediction = checker.predict(row["text"])
        if feature.nnz and (prediction["reason"] != classes[int(scores.argmax())]
                           or abs(prediction["confidence"]-float(scores.max())) > .00006):
            raise RuntimeError("Training/serving parity failed.")
    majority = Counter(y_train).most_common(1)[0][0]
    report = dict(algorithm="character TF-IDF + logistic regression", selected=dict(C=c, threshold=threshold),
                  candidates=candidates, validation_threshold_curve=curve, test=metrics,
                  dummy_majority_test_macro_f1=float(f1_score([r["reason"] for r in test], [majority]*len(test), average="macro", zero_division=0)),
                  training_examples=len(train), provenance=record)
    save_report(output, report, test, probabilities, classes, known_mask=known_test)
    write_json(output / "serving.json", dict(kind="tfidf", path=target.name, threshold=threshold, version=version))
    print(f"Candidate saved to {target}; test macro F1={metrics['macro_f1']:.3f}. Active app unchanged.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="ml_workflow/work/splits")
    parser.add_argument("--out", default="ml_workflow/work/baseline")
    args = parser.parse_args()
    train(args.splits, args.out)
