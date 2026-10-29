"""Optional later lessons: retail entities, 14-day recovery and weekly sales.

These require additional annotations/outcomes/history. Schema files are blank
templates, not datasets that can produce a useful trained model.
"""
import argparse
import json
from pathlib import Path

from .common import digest, normalized, write_json

ENTITY_TYPES = ("PRODUCT", "QUANTITY", "BUDGET", "CURRENCY", "LOCATION")
TAGS = ["O"] + [prefix + name for name in ENTITY_TYPES for prefix in ("B-", "I-")]
RECOVERY_CATEGORICAL = ["source", "reason", "product_id", "currency"]
RECOVERY_NUMERIC = ["qty", "budget", "urgency"]
RECOVERY_FEATURES = RECOVERY_CATEGORICAL + RECOVERY_NUMERIC
FORECAST_FEATURES = ["sku", "week_of_year", "lag_1", "lag_4", "lag_52", "mean_4"]


def entity_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen, ids = {}, set()
    for row in rows:
        if not row.get("id") or row["id"] in ids:
            raise ValueError("Entity records require unique ids.")
        ids.add(row["id"])
        if row.get("split") not in ("train", "validation", "test") or not row.get("group_id"):
            raise ValueError("Entities need a reviewed group_id and frozen split.")
        if not row.get("tokens") or len(row["tokens"]) != len(row.get("ner_tags", [])):
            raise ValueError("One BIO tag is required for each whitespace-delimited token.")
        previous = "O"
        for tag in row["ner_tags"]:
            if tag not in TAGS or (tag.startswith("I-") and previous not in ("B-"+tag[2:], tag)):
                raise ValueError("Invalid BIO label or I- tag without a matching previous entity.")
            previous = tag
        for key in (("group", row["group_id"]), ("text", normalized(" ".join(row["tokens"])))):
            if key in seen and seen[key] != row["split"]:
                raise ValueError("Entity data leakage: group/text crosses splits.")
            seen[key] = row["split"]
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Need labeled train, validation and test records; the schema example is not a training dataset.")
    return rows


def entities(path, output, checkpoint, revision):
    import numpy as np
    from collections import Counter
    from datasets import Dataset
    from seqeval.metrics import classification_report, f1_score
    from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(42)
    rows = entity_rows(path)
    output = Path(output)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, use_fast=True)
    tag_to_id = {label: i for i, label in enumerate(TAGS)}

    def tokenize(batch):
        encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=False)
        labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            if len(encoded["input_ids"][i]) > 256:
                raise ValueError("Split long records into non-overlapping labeled segments before training; do not silently truncate entities.")
            previous, aligned = None, []
            for word in encoded.word_ids(batch_index=i):
                aligned.append(-100 if word is None or word == previous else tag_to_id[tags[word]])
                previous = word
            labels.append(aligned)
        encoded["labels"] = labels
        return encoded

    data = {name: Dataset.from_list([row for row in rows if row["split"] == name]).map(tokenize, batched=True,
                    remove_columns=list(rows[0])) for name in ("train", "validation", "test")}
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, revision=revision, num_labels=len(TAGS),
                    id2label=dict(enumerate(TAGS)), label2id=tag_to_id)

    def metrics(prediction):
        expected, predicted = [], []
        for labels, scores in zip(prediction.label_ids, prediction.predictions.argmax(axis=-1)):
            expected.append([TAGS[int(label)] for label in labels if label != -100])
            predicted.append([TAGS[int(score)] for label, score in zip(labels, scores) if label != -100])
        return {"span_f1": float(f1_score(expected, predicted, zero_division=0))}

    args = TrainingArguments(output_dir=str(output/"checkpoints"), num_train_epochs=3,
              per_device_train_batch_size=8, per_device_eval_batch_size=8, eval_strategy="epoch", save_strategy="epoch",
              load_best_model_at_end=True, metric_for_best_model="span_f1", greater_is_better=True,
              save_total_limit=1, learning_rate=2e-5, seed=42, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["validation"],
                      processing_class=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=metrics)
    trainer.train()
    result = trainer.predict(data["test"])
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output/"model"))
    tokenizer.save_pretrained(str(output/"model"))
    truth, guesses = [], []
    for labels, predictions in zip(result.label_ids, np.argmax(result.predictions, axis=-1)):
        truth.append([TAGS[int(label)] for label in labels if label != -100])
        guesses.append([TAGS[int(value)] for label, value in zip(labels, predictions) if label != -100])
    write_json(output/"evaluation.json", dict(metrics=result.metrics,
               per_entity=classification_report(truth, guesses, output_dict=True, zero_division=0),
               rows={name: len(dataset) for name, dataset in data.items()},
               tag_counts=dict(Counter(tag for row in rows for tag in row["ner_tags"])),
               input_sha256=digest(path), base_model=checkpoint, revision=revision,
               warning="Token span F1 does not validate numeric parsing or currency. Check exact quantity/budget/currency agreement separately."))


def recovery_frame(path, as_of):
    import pandas as pd
    frame = pd.read_csv(path, dtype={column: str for column in RECOVERY_CATEGORICAL + ["id", "group_id", "split"]})
    required = set(RECOVERY_FEATURES + ["id", "group_id", "split", "captured_at", "features_at", "observed_through", "converted_at"])
    if not required <= set(frame) or frame.empty:
        raise ValueError("Supply completed outcome data using data/recovery_template.csv.")
    if frame.id.isna().any() or frame.id.duplicated().any() or frame.group_id.isna().any():
        raise ValueError("Recovery rows need unique ids and nonempty group_id values.")
    if not set(frame["split"]) <= {"train", "validation", "test"}:
        raise ValueError("Unknown partition name.")
    for column in ("captured_at", "features_at", "observed_through", "converted_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame[["captured_at", "features_at", "observed_through"]].isna().any().any():
        raise ValueError("Prediction/snapshot/observation timestamps are required.")
    if (frame.features_at > frame.captured_at).any():
        raise ValueError("Future feature snapshot detected.")
    if (frame.converted_at.notna() & (frame.converted_at < frame.captured_at)).any():
        raise ValueError("An order predates the request; review linkage.")
    cutoff = pd.to_datetime(as_of, utc=True)
    maturity = frame.captured_at + pd.Timedelta(days=14)
    # The negative label needs a fully observed window; recent open requests are excluded.
    frame = frame[(maturity <= cutoff) & (frame.observed_through >= maturity)].copy()
    frame["target"] = (frame.converted_at.notna() & (frame.converted_at <= frame.captured_at + pd.Timedelta(days=14))).astype(int)
    if frame.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("Recovery conversation/customer group crosses splits.")
    for column in RECOVERY_CATEGORICAL:
        frame[column] = frame[column].fillna("UNKNOWN")
    for column in RECOVERY_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isna().any() or not frame[column].between(0, 1e15).all():
            raise ValueError("Numeric features must be finite, nonnegative reviewed values.")
    parts = {name: frame[frame["split"] == name] for name in ("train", "validation", "test")}
    if any(part.empty or part.target.nunique() != 2 for part in parts.values()):
        raise ValueError("Each time partition needs mature positive and negative outcomes.")
    for left, right in zip(list(parts.values()), list(parts.values())[1:]):
        if (left.captured_at + pd.Timedelta(days=14)).max() >= right.captured_at.min():
            raise ValueError("Labels were not mature at the next partition's prediction time. Purge a 14-day gap and spanning groups.")
    return parts


def recovery(path, output, as_of):
    from catboost import CatBoostClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss
    parts = recovery_frame(path, as_of)
    train, validation, test = [parts[name] for name in ("train", "validation", "test")]
    model = CatBoostClassifier(iterations=300, depth=5, learning_rate=.05, loss_function="Logloss", random_seed=42,
                               verbose=False, allow_writing_files=False)
    model.fit(train[RECOVERY_FEATURES], train.target, cat_features=RECOVERY_CATEGORICAL,
              eval_set=(validation[RECOVERY_FEATURES], validation.target), early_stopping_rounds=30)
    scores = model.predict_proba(test[RECOVERY_FEATURES])[:, 1]
    top = max(1, int(len(test)*.2))
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output/"model.cbm"))
    write_json(output/"evaluation.json", dict(test_examples=len(test), average_precision=float(average_precision_score(test.target, scores)),
               prevalence_baseline=float(test.target.mean()), brier=float(brier_score_loss(test.target, scores)),
               constant_train_prior_brier=float(brier_score_loss(test.target, [float(train.target.mean())]*len(test))),
               precision_top_20_percent=float(test.target.iloc[np.argsort(-scores)[:top]].mean()),
               counts={name: dict(rows=len(part), positives=int(part.target.sum())) for name, part in parts.items()},
               input_sha256=digest(path), as_of=as_of, features=RECOVERY_FEATURES,
               warning="Predicts observed conversion under the historical follow-up policy, not causal lift or guaranteed recovery. Scores need separate calibration."))


def forecast_frame(path):
    import pandas as pd
    frame = pd.read_csv(path, dtype={"sku": str})
    if not {"sku", "week", "quantity"} <= set(frame) or frame.empty:
        raise ValueError("Supply weekly completed-sales data using data/weekly_sales_template.csv.")
    frame["week"] = pd.to_datetime(frame.week, errors="raise")
    frame["quantity"] = pd.to_numeric(frame.quantity, errors="raise")
    if (frame.sku.isna().any() or frame.week.isna().any() or frame.duplicated(["sku", "week"]).any()
            or not frame.quantity.between(0, 1e15).all() or not frame.week.dt.dayofweek.eq(0).all()):
        raise ValueError("Weekly sales need unique SKU/Monday keys and finite nonnegative quantities.")
    for _, part in frame.groupby("sku"):
        if not part.sort_values("week").week.diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError("Missing calendar weeks. Review listing/availability; fill zero only for observed active-selling weeks.")
    frame = frame.sort_values(["sku", "week"]).copy()
    grouped = frame.groupby("sku")["quantity"]
    for lag in (1, 4, 52):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    frame["mean_4"] = grouped.transform(lambda series: series.shift(1).rolling(4).mean())
    frame["week_of_year"] = frame.week.dt.isocalendar().week.astype(int)
    return frame.dropna(subset=FORECAST_FEATURES)


def forecast(path, output):
    from lightgbm import LGBMRegressor, early_stopping
    from sklearn.metrics import mean_absolute_error
    frame = forecast_frame(path)
    weeks = sorted(frame.week.unique())
    if len(weeks) < 15:
        raise ValueError("Need at least 15 evaluation weeks after 52-week lag history (roughly 67+ weeks per SKU).")
    a, b = int(len(weeks)*.6), int(len(weeks)*.8)
    parts = [frame[frame.week.isin(weeks[start:end])].copy() for start, end in ((0, a), (a, b), (b, len(weeks)))]
    train, validation, test = parts
    categories = sorted(train.sku.unique())
    for part in parts:
        if not set(part.sku) <= set(categories):
            raise ValueError("Cold-start SKU requires a separate policy/model.")
        import pandas as pd
        part["sku"] = pd.Categorical(part.sku, categories=categories)
    model = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=.05, random_state=42, verbosity=-1)
    model.fit(train[FORECAST_FEATURES], train.quantity, categorical_feature=["sku"],
              eval_set=[(validation[FORECAST_FEATURES], validation.quantity)], callbacks=[early_stopping(30, verbose=False)])
    predictions = model.predict(test[FORECAST_FEATURES]).clip(min=0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output/"model.txt"))
    write_json(output/"serving.json", dict(categories=categories, features=FORECAST_FEATURES))
    write_json(output/"evaluation.json", dict(test_rows=len(test), mae=float(mean_absolute_error(test.quantity, predictions)),
               lag1_naive_mae=float(mean_absolute_error(test.quantity, test.lag_1)),
               annual_seasonal_naive_mae=float(mean_absolute_error(test.quantity, test.lag_52)),
               input_sha256=digest(path), train_through=str(train.week.max()), validation_through=str(validation.week.max()),
               test_through=str(test.week.max()),
               warning="Rolling one-week-ahead backtest: previous test-week actuals are available for next week's features. Not a multiweek batch forecast or an unmet-demand estimate."))
    test[["sku", "week", "quantity"]].assign(predicted_quantity=predictions).to_csv(output/"test_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("entities", "recovery", "forecast"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--as-of", help="Recovery: UTC observation cutoff, e.g. 2026-10-28T00:00:00Z")
    parser.add_argument("--checkpoint", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.task == "entities":
        entities(args.data, args.out, args.checkpoint, args.revision)
    elif args.task == "recovery":
        if not args.as_of:
            parser.error("recovery requires --as-of")
        recovery(args.data, args.out, args.as_of)
    else:
        forecast(args.data, args.out)
