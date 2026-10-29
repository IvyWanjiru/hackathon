"""Lesson 7: compare lexical retrieval with MiniLM; optionally fine-tune embeddings.

Catalog text is available at prediction time. Query relevance judgments are held
out by query, not by individual query/product pair. Scores suggest candidates;
they are not availability, variant validation, or conversion probabilities.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import digest, normalized, read_csv, write_json


def catalog_text(row):
    return row["title"] + ". " + row.get("description", "")


def validate(catalog, queries):
    ids = [row["product_id"] for row in catalog]
    if not catalog or len(set(ids)) != len(ids) or any(not row["title"].strip() for row in catalog):
        raise ValueError("Catalog needs unique product_id values and nonempty titles.")
    known, seen, query_ids = set(ids), {}, set()
    for row in queries:
        if row["split"] not in ("train", "validation", "test") or not row["query"].strip():
            raise ValueError("Queries require text and train/validation/test split.")
        if row["query_id"] in query_ids:
            raise ValueError("One row per query_id; combine relevant IDs with |.")
        query_ids.add(row["query_id"])
        if not set(row["relevant_ids"].split("|")) <= known:
            raise ValueError("A relevance judgment refers to a missing catalog product.")
        key = normalized(row["query"])
        if key in seen and seen[key] != row["split"]:
            raise ValueError("Duplicate queries cross partitions.")
        seen[key] = row["split"]
    if {row["split"] for row in queries} != {"train", "validation", "test"}:
        raise ValueError("Provide independent train, validation and test queries.")


def retrieval_metrics(catalog, queries, score_function):
    recalls = {1: [], 3: []}
    reciprocal, predictions = [], []
    for row in queries:
        scores = np.asarray(score_function(row["query"])).ravel()
        ranked = np.argsort(-scores, kind="stable")
        ids = [catalog[int(i)]["product_id"] for i in ranked]
        relevant = set(row["relevant_ids"].split("|"))
        for k in recalls:
            recalls[k].append(len(set(ids[:k]) & relevant)/len(relevant))
        reciprocal.append(next((1/(i+1) for i, item in enumerate(ids) if item in relevant), 0))
        predictions.append(dict(query_id=row["query_id"], expected=sorted(relevant), top3=ids[:3]))
    if not queries:
        raise ValueError("Evaluation partition is empty.")
    return dict(queries=len(queries), recall_at_1=float(np.mean(recalls[1])),
                recall_at_3=float(np.mean(recalls[3])), mrr=float(np.mean(reciprocal)), predictions=predictions)


def fit_lexical(catalog, analyzer, ngrams):
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True)
    matrix = vectorizer.fit_transform([catalog_text(row) for row in catalog])
    return vectorizer, lambda text: (vectorizer.transform([text]) @ matrix.T).toarray().ravel()


class CatalogMatcher:
    def __init__(self, path):
        index_path = Path(path).resolve()
        self.record = json.loads(index_path.read_text())
        self.catalog = self.record["catalog"]
        if self.record["kind"] == "lexical":
            record = self.record
            self.vectorizer = TfidfVectorizer(analyzer=record["analyzer"], ngram_range=tuple(record["ngram_range"]),
                                            sublinear_tf=True, vocabulary=record["vocabulary"])
            self.vectorizer.idf_ = np.asarray(record["idf"])
            self.matrix = self.vectorizer.transform([catalog_text(row) for row in self.catalog])
            self.score = lambda text: (self.vectorizer.transform([text]) @ self.matrix.T).toarray().ravel()
        else:
            from sentence_transformers import SentenceTransformer
            model_path = self.record["model"]
            if self.record.get("local_model"):
                model_path = str(index_path.parent/model_path)
            self.model = SentenceTransformer(model_path, revision=self.record.get("revision"), trust_remote_code=False)
            self.matrix = self.model.encode([catalog_text(row) for row in self.catalog], normalize_embeddings=True)
            self.score = lambda text: self.model.encode([text], normalize_embeddings=True)[0] @ self.matrix.T

    def search(self, text, k=3):
        scores = np.asarray(self.score(text))
        return [dict(product_id=self.catalog[int(i)]["product_id"], title=self.catalog[int(i)]["title"],
                     similarity=round(float(scores[i]), 5), requires_review=True)
                for i in np.argsort(-scores, kind="stable")[:k]]


def train(catalog_path, queries_path, output, backend="lexical", model_name="sentence-transformers/all-MiniLM-L6-v2",
          finetune=False, revision=None):
    catalog, queries = read_csv(catalog_path), read_csv(queries_path)
    validate(catalog, queries)
    parts = {name: [row for row in queries if row["split"] == name] for name in ("train", "validation", "test")}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    if backend == "lexical":
        best = None
        for analyzer, ngrams in (("word", (1, 2)), ("char_wb", (3, 5))):
            vectorizer, score_function = fit_lexical(catalog, analyzer, ngrams)
            metrics = retrieval_metrics(catalog, parts["validation"], score_function)
            candidates.append(dict(analyzer=analyzer, validation_recall_at_1=metrics["recall_at_1"]))
            if best is None or metrics["recall_at_1"] > best[0]:
                best = metrics["recall_at_1"], analyzer, ngrams, vectorizer, score_function
        _, analyzer, ngrams, vectorizer, score_function = best
        record = dict(kind="lexical", catalog=catalog, analyzer=analyzer, ngram_range=list(ngrams),
                      vocabulary={key: int(value) for key, value in vectorizer.vocabulary_.items()}, idf=vectorizer.idf_.tolist())
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)

        def scorer(encoder):
            matrix = encoder.encode([catalog_text(row) for row in catalog], normalize_embeddings=True)
            return lambda text: encoder.encode([text], normalize_embeddings=True)[0] @ matrix.T

        score_function = scorer(model)
        base_metrics = retrieval_metrics(catalog, parts["validation"], score_function)
        candidates.append(dict(kind="pretrained", validation_recall_at_1=base_metrics["recall_at_1"]))
        selected_model, selected_revision = model_name, revision
        if finetune:
            from datasets import Dataset
            from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
            from transformers import set_seed
            set_seed(42)
            # Use explicitly judged TRAIN query/product pairs. Unjudged items
            # are not negatives; prepare_public.py keeps ESCI E/I judgments.
            pairs_path = Path(queries_path).with_name("pairs.csv")
            if not pairs_path.exists():
                raise ValueError("Fine-tuning requires pairs.csv with explicit TRAIN query/product judgments. See DATASETS.md.")
            raw_pairs = read_csv(pairs_path)
            train_queries = {row["query_id"]: row["query"] for row in parts["train"]}
            products = {row["product_id"]: catalog_text(row) for row in catalog}
            pairs, seen_pairs = [], {}
            for row in raw_pairs:
                if row["query_id"] not in train_queries:
                    continue  # Never train on validation/test judgments.
                if row["product_id"] not in products or row["label"] not in ("0", "1"):
                    raise ValueError("Pairs need a known product_id and exact/not-exact label 1/0.")
                pair_key = (row["query_id"], row["product_id"])
                if pair_key in seen_pairs:
                    if seen_pairs[pair_key] != row["label"]:
                        raise ValueError("Conflicting labels for a query/product pair.")
                    continue
                seen_pairs[pair_key] = row["label"]
                pairs.append(dict(sentence1=train_queries[row["query_id"]], sentence2=products[row["product_id"]], score=float(row["label"])))
            if not pairs or {pair["score"] for pair in pairs} != {0., 1.}:
                raise ValueError("Need explicitly judged positive and negative training pairs.")
            args = SentenceTransformerTrainingArguments(output_dir=str(output/"checkpoints"), num_train_epochs=1,
                        per_device_train_batch_size=8, learning_rate=2e-5, seed=42, report_to="none",
                        save_strategy="no", fp16=False, bf16=False)
            SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_list(pairs),
                                       loss=losses.CosineSimilarityLoss(model)).train()
            tuned_score = scorer(model)
            tuned_metrics = retrieval_metrics(catalog, parts["validation"], tuned_score)
            candidates.append(dict(kind="finetuned", validation_recall_at_1=tuned_metrics["recall_at_1"]))
            if tuned_metrics["recall_at_1"] > base_metrics["recall_at_1"]:
                selected_model = "encoder"
                selected_revision = None
                model.save_pretrained(str(output/selected_model))
                score_function = tuned_score
            else:
                model = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)
                score_function = scorer(model)
        record = dict(kind="minilm", catalog=catalog, model=selected_model, revision=selected_revision,
                      local_model=selected_model == "encoder")
    record.update(catalog_sha256=digest(catalog_path), queries_sha256=digest(queries_path))
    write_json(output/"index.json", record)
    report = dict(candidates=candidates, test=retrieval_metrics(catalog, parts["test"], score_function),
                  catalog_products=len(catalog), warning="Synthetic/local relevance exercise. Review variants. Similarity is not a probability or stock status.")
    write_json(output/"evaluation.json", report)
    print(f"Matching candidate saved to {output}; Recall@1={report['test']['recall_at_1']:.3f}. App unchanged.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="ml_workflow/data/catalog.csv")
    parser.add_argument("--queries", default="ml_workflow/data/queries.csv")
    parser.add_argument("--out", default="ml_workflow/work/matching")
    parser.add_argument("--backend", choices=("lexical", "minilm"), default="lexical")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision")
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()
    if args.finetune and args.backend != "minilm":
        parser.error("--finetune requires --backend minilm")
    train(args.catalog, args.queries, args.out, args.backend, args.model, args.finetune, args.revision)
