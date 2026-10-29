"""Lesson 1: create an explicitly synthetic practice CSV; optionally export reviewed labels."""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .common import ROOT, FIELDS, digest, write_csv, write_json


def scrub(text):
    # Basic minimization only. Review names, addresses and order references manually.
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return re.sub(r"(?<!\w)\+\d[\d ()-]{8,}\d", "[PHONE]", text)


def prepare(output, database=None):
    output = Path(output)
    rows = []
    if database:
        # Read-only URI: never run a migration or update the live database.
        uri = Path(database).resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            raw = connection.execute("SELECT id,raw,source,ts,reviewed_reason FROM signals "
                                     "WHERE reviewed_reason IS NOT NULL").fetchall()
        for key, text, source, ts, reason in raw:
            anonymous = hashlib.sha256(f"signal-{key}".encode()).hexdigest()[:16]
            rows.append(dict(id=anonymous, text=scrub(text), reason=reason, source=source,
                             conversation_id=f"REVIEW-ME-{anonymous}", group_id=f"REVIEW-ME-{anonymous}",
                             captured_at=ts, origin="reviewed_business_export", is_synthetic="0", verified="1"))
        target = output / "reviewed.csv"
        caveat = "Conversation/customer grouping is absent from the current DB: replace REVIEW-ME IDs before splitting."
    else:
        raw = json.loads((ROOT / "training_data.json").read_text())
        for i, item in enumerate(raw, 1):
            rows.append(dict(id=f"starter-{i:03}", text=item["text"], reason=item["reason"],
                             source="synthetic", conversation_id=f"starter-{i:03}", group_id=f"starter-{i:03}",
                             captured_at="", origin="ghostdemand_synthetic_starter", is_synthetic="1", verified="1"))
        target = output / "reasons.csv"
        caveat = "Synthetic exercise only. Template families are unknown; manually group paraphrases before a stronger evaluation."
    write_csv(target, rows, FIELDS)
    write_json(target.with_suffix(".manifest.json"), dict(rows=len(rows), path=str(target), sha256=digest(target),
               source=str(Path(database).name if database else "training_data.json"), warning=caveat))
    print(f"Wrote {len(rows)} rows to {target}. {caveat}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ml_workflow/work/data")
    parser.add_argument("--reviewed-db", help="Optional local SQLite file, opened read-only. Never exported automatically.")
    args = parser.parse_args()
    prepare(args.out, args.reviewed_db)
