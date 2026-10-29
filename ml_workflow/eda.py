"""Lesson 2: inspect grain, missingness, labels, origin, duplicates and likely paraphrases."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .common import FIELDS, REASON_LABELS, digest, linked_groups, normalized, read_csv, write_json


def profile(path, output):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    rows = read_csv(path)
    if not rows:
        raise ValueError("Cannot profile an empty CSV.")
    lengths = [len(row.get("text", "")) for row in rows]
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[normalized(row.get("text", ""))].append(row)
    conflicts = sum(len({r.get("reason") for r in values}) > 1 for values in duplicates.values())
    candidates = []
    texts = [row.get("text", "") for row in rows]
    if any(normalized(text) for text in texts) and len(rows) > 1:
        features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=6000).fit_transform(texts)
        # EDA only; these fitted features NEVER enter the model-training stage.
        neighbors = NearestNeighbors(n_neighbors=min(4, len(rows)), metric="cosine").fit(features)
        distances, indices = neighbors.kneighbors(features)
        seen = set()
        for i, (ds, js) in enumerate(zip(distances, indices)):
            for distance, j in zip(ds, js):
                pair = tuple(sorted((i, int(j))))
                if i != j and pair not in seen and distance <= .2 and normalized(texts[i]) != normalized(texts[j]):
                    seen.add(pair)
                    candidates.append(dict(ids=[rows[p].get("id", str(p)) for p in pair],
                                           similarity=round(1 - float(distance), 3)))
    result = dict(source=str(path), sha256=digest(path), grain="one labeled customer-request context",
                  rows=len(rows), columns=list(rows[0]),
                  missing={field: sum(not row.get(field, "").strip() for row in rows) for field in FIELDS},
                  labels=dict(Counter(row.get("reason", "") for row in rows)),
                  unknown_labels=sum(row.get("reason") not in REASON_LABELS for row in rows),
                  sources=dict(Counter(row.get("source", "") for row in rows)),
                  origins=dict(Counter(row.get("origin", "") for row in rows)),
                  synthetic_rows=sum(row.get("is_synthetic") == "1" for row in rows),
                  duplicate_ids=len(rows)-len({row.get("id") for row in rows}),
                  duplicate_text_rows=sum(len(values)-1 for values in duplicates.values()),
                  conflicting_text_groups=conflicts, linked_groups=len(set(linked_groups(rows))),
                  text_characters={str(p): float(np.percentile(lengths, p)) for p in (0, 25, 50, 75, 95, 100)},
                  possible_paraphrases=candidates[:50],
                  note="No raw messages included in this report. Near-duplicate candidates require manual review; this is not exhaustive.")
    output = Path(output)
    write_json(output / "eda.json", result)
    lines = ["# Dataset exploration", "", f"Input: `{path}`. Rows: **{len(rows)}**. Grain: one labeled request context.",
             "", "## Label distribution", "", "| Reason | Rows | Share |", "|---|---:|---:|"]
    lines += [f"| {label} | {count} | {count/len(rows):.1%} |" for label, count in sorted(result["labels"].items())]
    lines += ["", "## Checks and interpretation", "",
              f"- Synthetic rows: {result['synthetic_rows']}/{len(rows)}. Synthetic evaluation does not measure customer accuracy.",
              f"- Duplicate text rows: {result['duplicate_text_rows']}; conflicting text groups: {conflicts}; duplicate IDs: {result['duplicate_ids']}.",
              f"- Unknown reasons: {result['unknown_labels']}. Training must stop on unknown or conflicting labels.",
              f"- Linked groups: {result['linked_groups']}. Keep conversations, customer/template groups and duplicates in one partition.",
              f"- Possible paraphrase pairs: {len(candidates)} (nearest-neighbor screen, cosine ≥0.80). Review before splitting.",
              f"- Missing values by field: `{result['missing']}`.",
              f"- Character-length percentiles: `{result['text_characters']}`.",
              f"- Sources: `{result['sources']}`. Compare future performance by channel and language.",
              "- Missing dates prevent a later-period deployment evaluation. Missing customer/template groups can leave leakage undetected.",
              "- Class balance alone does not prove coverage: include difficult, ambiguous, multilingual and ordinary non-loss conversations.",
              "", "Inspect the paired JSON for the checks and candidate row IDs. Resolve conflicts; supply grouping and dates for real data."]
    (output / "eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"EDA saved to {output / 'eda.md'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml_workflow/work/data/reasons.csv")
    parser.add_argument("--out", default="ml_workflow/work/reports")
    args = parser.parse_args()
    profile(args.data, args.out)
