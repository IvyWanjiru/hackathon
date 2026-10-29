"""Small, inspectable data contracts shared by the lessons."""
import csv
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from demand_ml import REASON_LABELS

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "text", "reason", "source", "conversation_id", "group_id",
          "captured_at", "origin", "is_synthetic", "verified")


def normalized(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic(row):
    return str(row.get("is_synthetic", "")).lower() in ("1", "true", "yes")


def validate_reasons(rows):
    if not rows:
        raise ValueError("No examples. Prepare and label a dataset first.")
    ids, labels = set(), {}
    for row in rows:
        if not row.get("id") or row["id"] in ids:
            raise ValueError("Each row needs a unique id.")
        ids.add(row["id"])
        if not row.get("text", "").strip() or len(row["text"]) > 10000:
            raise ValueError(f"{row['id']}: text must contain 1–10,000 characters.")
        if row.get("reason") not in REASON_LABELS:
            raise ValueError(f"{row['id']}: reason must be one of {REASON_LABELS}.")
        if row.get("verified") != "1":
            raise ValueError(f"{row['id']}: human verification is required (verified=1).")
        if row.get("is_synthetic") not in ("0", "1") or not row.get("origin"):
            raise ValueError(f"{row['id']}: origin and is_synthetic=0/1 are required.")
        if not row.get("conversation_id") or not row.get("group_id"):
            raise ValueError(f"{row['id']}: conversation_id and group_id are required.")
        key = normalized(row["text"])
        if not key:
            raise ValueError(f"{row['id']}: no usable text.")
        if key in labels and labels[key] != row["reason"]:
            raise ValueError("Conflicting labels for the same normalized text; resolve with a reviewer.")
        labels[key] = row["reason"]


def linked_groups(rows):
    """Union conversation IDs, template/customer groups, and duplicate text.

    Transitive linking matters: A shares a conversation with B, and B repeats C.
    All three must stay together, even if A and C have different supplied groups.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [("text", normalized(row["text"]))]
        keys += [(field, row[field]) for field in ("conversation_id", "group_id") if row.get(field)]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return [find(i) for i in range(len(rows))]


def load_partitions(directory, verify_manifest=True):
    manifest_path = Path(directory)/"manifest.json"
    if verify_manifest and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name in ("train", "validation", "test"):
            if digest(Path(directory)/f"{name}.csv") != manifest["partitions"][name]["sha256"]:
                raise ValueError("A frozen split was edited after its manifest was written; regenerate and review the split.")
    rows = {name: read_csv(Path(directory) / f"{name}.csv") for name in ("train", "validation", "test")}
    flat = [row for part in rows.values() for row in part]
    validate_reasons(flat)
    memberships = defaultdict(set)
    groups = linked_groups(flat)
    offset = 0
    for name, part in rows.items():
        if set(row["reason"] for row in part) != set(REASON_LABELS):
            raise ValueError(f"{name}: every reason must be represented; collect more independent groups.")
        for group in groups[offset:offset + len(part)]:
            memberships[group].add(name)
        offset += len(part)
    if any(len(names) > 1 for names in memberships.values()):
        raise ValueError("Data leakage: linked groups or duplicate texts cross partitions.")
    return rows


def provenance(paths):
    import sklearn
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "sha256": {str(path): digest(path) for path in paths}, "seed": 42}


def counts(rows):
    return dict(sorted(Counter(row["reason"] for row in rows).items()))
