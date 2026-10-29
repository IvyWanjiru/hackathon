"""Optional dataset adapters. No public support intent is relabeled as a lost sale.

Download files from the publisher links in DATASETS.md, then run these adapters.
The original files stay unchanged; output hashes and sampling rules are retained.
"""
import argparse
import hashlib
from pathlib import Path

from .common import digest, normalized, read_csv, write_csv, write_json


def bitext(path, output, limit):
    rows = read_csv(path)
    required = {"instruction", "intent", "category"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Expected the publisher's instruction/intent/category CSV columns.")
    # Spread the labeling sample across intents instead of taking a sorted head.
    buckets = {}
    for i, row in enumerate(rows):
        buckets.setdefault(row["intent"], []).append((i, row))
    selected = []
    depth = 0
    while len(selected) < min(limit, len(rows)):
        for intent in sorted(buckets):
            if depth < len(buckets[intent]) and len(selected) < limit:
                i, row = buckets[intent][depth]
                selected.append(dict(id=f"bitext-{i}", text=row["instruction"], original_intent=intent,
                                     category=row["category"], eligibility="", reason="", verified="0",
                                     group_id="REVIEW-TEMPLATE-FAMILY", origin="bitext_hybrid_synthetic", is_synthetic="1"))
        depth += 1
    output = Path(output)
    write_csv(output/"annotation_queue.csv", selected,
              ("id", "text", "original_intent", "category", "eligibility", "reason", "verified", "group_id", "origin", "is_synthetic"))
    write_json(output/"manifest.json", dict(source="https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
               input_sha256=digest(path), input_rows=len(rows), sampled_rows=len(selected), license="CDLA-Sharing-1.0",
               sample_rule="Round-robin through sorted intent names; original file order within each intent.",
               warning="Annotation queue only. Most texts cannot establish an unmet request or a GhostDemand blocker without more context."))
    print("Prepared annotation_queue.csv. No blocker labels were inferred.")


def esci(examples, products, output, limit):
    import pandas as pd

    frame = pd.read_parquet(examples, columns=["query", "query_id", "product_id", "product_locale", "esci_label", "small_version", "split"])
    frame = frame[(frame.product_locale == "us") & (frame.small_version == 1)].copy()
    frame["query_key"] = frame["query"].map(normalized)
    # An identical normalized query in official test is NEVER included in train.
    official_test_keys = set(frame.loc[frame["split"] == "test", "query_key"])
    frame = frame[(frame["split"] == "test") | ~frame.query_key.isin(official_test_keys)]
    frame["query_group"] = frame.query_key.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:16])
    frame["partition"] = frame.apply(lambda row: "test" if row["split"] == "test" else
                    ("validation" if int(row.query_group[:8], 16) % 5 == 0 else "train"), axis=1)
    selected = []
    for partition, proportion in (("train", .6), ("validation", .2), ("test", .2)):
        subset = frame[frame.partition == partition]
        positives = subset.loc[subset.esci_label == "E", "query_group"].drop_duplicates().sort_values()
        keys = set(positives.head(max(1, int(limit*proportion))))
        selected.append(subset[subset.query_group.isin(keys)])
    frame = pd.concat(selected)
    metadata = pd.read_parquet(products, columns=["product_id", "product_locale", "product_title", "product_description"])
    metadata = metadata[(metadata.product_locale == "us") & metadata.product_id.isin(frame.product_id)]
    if metadata.duplicated(["product_locale", "product_id"]).any():
        raise ValueError("Product keys are duplicated; stop before an expanding join.")
    joined = frame.merge(metadata, how="left", on=["product_locale", "product_id"], validate="many_to_one")
    if joined.product_title.isna().any():
        raise ValueError("Missing product metadata. Do not silently drop incomplete judgments.")
    catalog = [dict(product_id=str(row.product_id), title=row.product_title,
                    description="" if pd.isna(row.product_description) else row.product_description, origin="amazon_esci")
               for row in metadata.itertuples()]
    queries, pairs = [], []
    for key, group in joined.groupby("query_group", sort=True):
        row = group.iloc[0]
        queries.append(dict(query_id=key, query=row["query"], relevant_ids="|".join(sorted(set(group.loc[group.esci_label == "E", "product_id"].astype(str)))),
                            split=row["partition"], origin="amazon_esci"))
        for product_id, label in group[["product_id", "esci_label"]].drop_duplicates().itertuples(index=False, name=None):
            if label in ("E", "I"):
                pairs.append(dict(query_id=key, product_id=str(product_id), label="1" if label == "E" else "0", origin="amazon_esci"))
    output = Path(output)
    write_csv(output/"catalog.csv", catalog, ("product_id", "title", "description", "origin"))
    write_csv(output/"queries.csv", queries, ("query_id", "query", "relevant_ids", "split", "origin"))
    write_csv(output/"pairs.csv", pairs, ("query_id", "product_id", "label", "origin"))
    write_json(output/"manifest.json", dict(source="https://github.com/amazon-science/esci-data", license="Apache-2.0",
               input_sha256={"examples": digest(examples), "products": digest(products)},
               queries=len(queries), products=len(catalog), pairs=len(pairs),
               sample_rule="English US, small version; official test preserved; 20% normalized-query hash validation from official train; bounded sorted hash sample.",
               warning="Local sampled catalog evaluation, not an official ESCI benchmark. Unjudged products are not proven irrelevant. S/C labels are not exact matches or I negatives."))
    print(f"Prepared {len(queries)} queries, {len(catalog)} products and {len(pairs)} E/I pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    b = sub.add_parser("bitext")
    b.add_argument("--csv", required=True)
    b.add_argument("--out", default="ml_workflow/work/bitext")
    b.add_argument("--limit", type=int, default=460)
    e = sub.add_parser("esci")
    e.add_argument("--examples", required=True)
    e.add_argument("--products", required=True)
    e.add_argument("--out", default="ml_workflow/work/esci")
    e.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    if args.limit < 5:
        parser.error("--limit must be at least 5")
    if args.source == "bitext":
        bitext(args.csv, args.out, args.limit)
    else:
        esci(args.examples, args.products, args.out, args.limit)
