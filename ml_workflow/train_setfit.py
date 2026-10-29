"""Optional lesson 6: fine-tune MiniLM with SetFit on the SAME frozen reason splits."""
import argparse
from pathlib import Path

from .common import REASON_LABELS, load_partitions, provenance, write_json
from .evaluation import choose_threshold, evaluate, save_report


def train(directory, output, base_model, revision=None):
    import numpy as np
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from transformers import set_seed

    set_seed(42)
    parts = load_partitions(directory)
    labels = sorted(REASON_LABELS)
    dataset = Dataset.from_dict({"text": [row["text"] for row in parts["train"]],
                                 "label": [labels.index(row["reason"]) for row in parts["train"]]})
    kwargs = {"revision": revision} if revision else {}
    model = SetFitModel.from_pretrained(base_model, labels=labels, **kwargs)
    output = Path(output)
    args = TrainingArguments(output_dir=str(output/"checkpoints"), batch_size=8, num_epochs=1,
                             num_iterations=10, seed=42, report_to="none")
    Trainer(model=model, args=args, train_dataset=dataset).train()
    validation = np.asarray(model.predict_proba([row["text"] for row in parts["validation"]], as_numpy=True))
    threshold, curve = choose_threshold([row["reason"] for row in parts["validation"]], validation, labels)
    probabilities = np.asarray(model.predict_proba([row["text"] for row in parts["test"]], as_numpy=True))
    metrics = evaluate(parts["test"], probabilities, labels, threshold)
    target = output / "model"
    model.save_pretrained(str(target))
    record = provenance([Path(directory)/f"{name}.csv" for name in parts])
    record.update(base_model=base_model, revision=revision or "main (record resolved Hub commit before production)")
    report = dict(algorithm="SetFit + MiniLM", selected=dict(epochs=1, threshold=threshold),
                  validation_threshold_curve=curve, test=metrics, provenance=record)
    save_report(output, report, parts["test"], probabilities, labels)
    write_json(output / "serving.json", dict(kind="setfit", path="model", labels=labels,
               threshold=threshold, version="setfit-" + record["sha256"][str(Path(directory)/"train.csv")][:10]))
    print("SetFit candidate saved. Compare both evaluation.json files; no automatic promotion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="ml_workflow/work/splits")
    parser.add_argument("--out", default="ml_workflow/work/setfit")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision", help="Use a Hugging Face commit hash for a repeatable download.")
    args = parser.parse_args()
    train(args.splits, args.out, args.base_model, args.revision)
