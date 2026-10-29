"""PesaShield AI: M-Pesa SMS scam detection with review->train loop."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
STARTER_PATH = ROOT / "training_fraud_sms.json"
VECTORIZER_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=5000)

SCAM_LABELS = ["scam", "suspicious", "legit"]


class TrainingError(ValueError):
    pass


class ScamClassifier:
    def __init__(self, path=None, enabled=True, threshold=0.6):
        self.path = Path(path) if path else ROOT / "models" / "scam_classifier.json"
        self.enabled = enabled
        self.threshold = threshold
        self._vectorizer = None
        self._classifier = None
        self._metadata = None
        self._load()

    def status(self):
        return dict(ready=self._classifier is not None, enabled=self.enabled,
                    threshold=self.threshold, path=str(self.path), metadata=self._metadata or {})

    def _load(self):
        self._vectorizer = self._classifier = self._metadata = None
        if not self.enabled or not self.path.exists():
            return
        try:
            artifact = json.loads(self.path.read_text())
            self._decode(artifact)
            self._metadata = artifact.get("metadata") or {}
        except Exception:
            self._vectorizer = self._classifier = self._metadata = None

    def _decode(self, artifact):
        if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
            raise ValueError("Unsupported fraud model format.")
        classes = list(artifact.get("classes") or [])
        vocabulary = artifact.get("vocabulary") or {}
        idf = artifact.get("idf") or []
        weights = artifact.get("weights") or []
        intercept = artifact.get("intercept") or []
        if len(vocabulary) != len(idf):
            raise ValueError("Corrupted fraud vectorizer.")
        params = artifact.get("vectorizer_params") or {}
        vectorizer = TfidfVectorizer(**{**VECTORIZER_PARAMS, **params})
        vectorizer.vocabulary_ = {k: int(v) for k, v in vocabulary.items()}
        vectorizer.idf_ = np.array(idf, dtype=float)
        vectorizer._tfidf._idf_diag = None
        classifier = LogisticRegression()
        classifier.classes_ = np.array(classes)
        w = np.array(weights, dtype=float)
        b = np.array(intercept)
        n_features = len(vocabulary)
        if len(classes) == 2:
            if w.ndim == 1:
                w = w.reshape(1, -1)
            if b.ndim == 0:
                b = np.array([float(b)])
            elif b.shape == ():
                b = np.array([float(b.item())])
            else:
                b = np.array([float(b[0])] if len(b) == 1 else b)
            # ensure shapes
            if w.shape != (1, n_features):
                raise ValueError("Corrupted fraud model shapes.")
            if b.shape != (1,):
                raise ValueError("Corrupted fraud model shapes.")
            classifier.coef_ = w
            classifier.intercept_ = b
        else:
            if w.shape != (len(classes), n_features) or b.shape != (len(classes),):
                raise ValueError("Corrupted fraud model shapes.")
            classifier.coef_ = w
            classifier.intercept_ = b
        classifier.n_features_in_ = n_features
        self._vectorizer = vectorizer
        self._classifier = classifier

    def _refresh(self):
        self._load()

    def predict(self, text):
        if not self.enabled or self._classifier is None or not isinstance(text, str) or not text.strip():
            return None
        try:
            X = self._vectorizer.transform([text])
            probs = self._classifier.predict_proba(X)[0]
            classes = self._classifier.classes_.tolist()
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            if confidence < self.threshold:
                return None
            return dict(verdict=classes[idx], confidence=round(confidence, 4), model_version=(self._metadata or {}).get("version"))
        except Exception:
            return None

    def train(self, examples):
        if not isinstance(examples, list):
            raise TrainingError("Training examples must be a list.")
        data = []
        for item in examples:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or ""
            label = item.get("label")
            if not isinstance(text, str) or not text.strip() or label not in SCAM_LABELS:
                continue
            data.append(dict(text=text.strip(), label=label))
        if not data:
            raise TrainingError("Add at least one reviewed scam SMS before training.")
        texts = [d["text"] for d in data]
        labels = [d["label"] for d in data]
        vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
        X = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(max_iter=200, class_weight="balanced")
        classifier.fit(X, labels)
        version = f"fraud-{os.urandom(6).hex()}"
        metadata = dict(version=version, trained_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        examples=len(data), classes=list(classifier.classes_), method="LogisticRegression on char n-grams")
        w = classifier.coef_
        b = classifier.intercept_
        if len(classifier.classes_) == 2 and w.shape == (1, X.shape[1]):
            w_out = w[0].tolist()
            b_out = float(b[0])
        else:
            w_out = w.tolist()
            b_out = b.tolist()
        artifact = dict(format_version=1, metadata=metadata, classes=list(classifier.classes_),
                        vectorizer_params={**VECTORIZER_PARAMS, "ngram_range": list(VECTORIZER_PARAMS["ngram_range"])},
                        vocabulary={k: int(v) for k, v in vectorizer.vocabulary_.items()},
                        idf=vectorizer.idf_.tolist(), weights=w_out,
                        intercept=b_out)
        self._decode(artifact)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, suffix=".json", delete=False) as f:
                temporary = f.name
                json.dump(artifact, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self._refresh()
        return metadata
