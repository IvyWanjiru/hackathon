"""Lesson 3: freeze train/validation/test data without conversation or text leakage."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import counts, digest, linked_groups, load_partitions, read_csv, validate_reasons, write_csv, write_json


def split_data(path, output, chronological=False):
    from sklearn.model_selection import StratifiedGroupKFold
    rows = read_csv(path)
    validate_reasons(rows)
    original_rows = rows
    if any("REVIEW-ME" in row.get(field, "") for row in rows for field in ("conversation_id", "group_id")):
        raise ValueError("Replace REVIEW-ME groups with true conversation/customer/template groups before splitting.")
    # Keep only one copy of each normalized text, after retaining all group links.
    from .common import normalized
    original_groups = linked_groups(rows)
    clean, seen = [], set()
    for row, group in zip(rows, original_groups):
        if normalized(row["text"]) not in seen:
            clean.append({**row, "group_id": f"linked-{group}"})
            seen.add(normalized(row["text"]))
    rows = clean
    groups = linked_groups(rows)
    if chronological:
        original_dates = defaultdict(list)
        for row, original_group in zip(original_rows, original_groups):
            parsed = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Real-data dates must include a timezone.")
            original_dates[f"linked-{original_group}"].append(parsed.astimezone(timezone.utc))
        by_group = defaultdict(list)
        for row, group in zip(rows, groups):
            # Deduplication must not erase a later repeated message's timestamp.
            by_group[group].extend(original_dates[row["group_id"]])
        ordered = sorted(by_group, key=lambda group: max(by_group[group]))
        a, b = int(len(ordered)*.6), int(len(ordered)*.8)
        buckets = [set(ordered[:a]), set(ordered[a:b]), set(ordered[b:])]
        parts = {name: [row for row, group in zip(rows, groups) if group in bucket]
                 for name, bucket in zip(("train", "validation", "test"), buckets)}
        for left, right in zip(buckets, buckets[1:]):
            if not left or not right or max(max(by_group[g]) for g in left) >= min(min(by_group[g]) for g in right):
                raise ValueError("Conversations span time boundaries. Choose explicit cutoffs and purge spanning groups first.")
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = [test for _, test in splitter.split(rows, [row["reason"] for row in rows], groups)]
        ids = {"train": [i for fold in folds[:3] for i in fold], "validation": folds[3], "test": folds[4]}
        parts = {name: [rows[int(i)] for i in indices] for name, indices in ids.items()}
    output = Path(output)
    for name, part in parts.items():
        write_csv(output / f"{name}.csv", part)
    load_partitions(output, verify_manifest=False)  # Check new files before replacing an older manifest.
    write_json(output / "manifest.json", dict(input_sha256=digest(path), seed=42,
               method="chronological grouped 60/20/20" if chronological else "three/two held-out folds of stratified group 5-fold; approximately 60/20/20",
               input_rows=len(original_groups), deduplicated_rows=len(rows),
               partitions={name: dict(rows=len(part), classes=counts(part), sha256=digest(output/f"{name}.csv")) for name, part in parts.items()},
               limitation="Grouping only protects relationships that are recorded. Unknown synthetic paraphrase families can still inflate results."))
    print({name: len(part) for name, part in parts.items()})
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml_workflow/work/data/reasons.csv")
    parser.add_argument("--out", default="ml_workflow/work/splits")
    parser.add_argument("--chronological", action="store_true")
    args = parser.parse_args()
    split_data(args.data, args.out, args.chronological)
