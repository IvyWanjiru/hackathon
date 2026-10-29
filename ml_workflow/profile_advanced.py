"""EDA for later-model data contracts, before spending time on training."""
import argparse
from collections import Counter

from .common import digest, read_csv, write_json


def profile(task, path, output, as_of=None):
    if task == "entities":
        from .advanced import entity_rows
        rows = entity_rows(path)
        result = dict(rows=len(rows), partitions=dict(Counter(row["split"] for row in rows)),
                      groups=len({row["group_id"] for row in rows}),
                      tags=dict(Counter(tag for row in rows for tag in row["ner_tags"])),
                      max_tokens=max(len(row["tokens"]) for row in rows),
                      warning="Inspect tokenization, annotation disagreements, omitted fields and per-entity coverage in each split. No raw text exported.")
    elif task == "recovery":
        from .advanced import recovery_frame
        if not as_of:
            raise ValueError("Recovery profiling requires --as-of.")
        original = read_csv(path)
        parts = recovery_frame(path, as_of)
        result = dict(raw_rows=len(original), mature_rows=sum(len(part) for part in parts.values()),
                      partitions={name: dict(rows=len(part), positive_rate=float(part.target.mean()),
                          start=str(part.captured_at.min()), end=str(part.captured_at.max()),
                          channels=part.source.value_counts().to_dict()) for name, part in parts.items()},
                      warning="Recent/insufficiently observed requests are excluded. Historical policy and order-linkage bias need review.")
    else:
        from .advanced import forecast_frame
        frame = forecast_frame(path)
        if frame.empty:
            raise ValueError("No rows remain after 52-week lag history; collect longer series.")
        result = dict(raw_rows=len(read_csv(path)), eligible_rows=len(frame), skus=int(frame.sku.nunique()),
                      start=str(frame.week.min()), end=str(frame.week.max()),
                      zero_sales_share=float(frame.quantity.eq(0).mean()), quantities=frame.quantity.describe().to_dict(),
                      warning="Check launch/retirement dates, returns, partial weeks and stockouts. Completed sales are a censored proxy for demand.")
    result.update(task=task, input_sha256=digest(path))
    write_json(output, result)
    print(f"EDA saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("entities", "recovery", "forecast"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    profile(args.task, args.data, args.out, args.as_of)
