"""Lesson 9: inspect a new labeled batch without retraining or promoting a model."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_csv, validate_reasons, write_json
from .service import create_app


def monitor(config, batch, output):
    rows = read_csv(batch)
    validate_reasons(rows)
    client = create_app(config).test_client()
    accepted = correct = accepted_correct = missing = 0
    predicted, sources = Counter(), Counter()
    for row in rows:
        response = client.post("/predict", json={"text": row["text"]})
        if response.status_code != 200:
            raise RuntimeError("Prediction failed; inspect service configuration/authentication.")
        prediction = response.get_json()["reason"]
        sources[row["source"]] += 1
        if prediction is None:
            missing += 1
            continue
        predicted[prediction["reason"]] += 1
        correct += prediction["reason"] == row["reason"]
        accepted += prediction["used"]
        accepted_correct += prediction["used"] and prediction["reason"] == row["reason"]
    result = dict(rows=len(rows), accuracy_including_missing_as_errors=correct/len(rows),
                  prediction_missing=missing, accepted=accepted, coverage=accepted/len(rows),
                  accepted_accuracy=accepted_correct/accepted if accepted else None,
                  predicted_labels=dict(predicted), expected_labels=dict(Counter(row["reason"] for row in rows)),
                  channels=dict(sources), note="Compare with a representative real holdout. Label/channel mix changes alone are not proof of degraded accuracy.")
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason-config", default="ml_workflow/work/baseline/serving.json")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out", default="ml_workflow/work/monitoring.json")
    args = parser.parse_args()
    monitor(args.reason_config, args.batch, args.out)
