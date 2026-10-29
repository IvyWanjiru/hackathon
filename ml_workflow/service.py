"""Lesson 8: optional isolated inference service; never imports or edits app.py."""
import argparse
import json
import logging
import os
import secrets
import time
from pathlib import Path

from flask import Flask, jsonify, request

from demand_ml import ReasonClassifier
from .matching import CatalogMatcher


class SetFitReason:
    def __init__(self, config):
        import numpy as np
        from setfit import SetFitModel
        self.np = np
        # Only load your own trusted local artifact: SetFit's sklearn head can use pickle.
        self.model = SetFitModel.from_pretrained(config["path"], local_files_only=True)
        self.labels = config["labels"]
        self.threshold, self.version = config["threshold"], config["version"]
        self.config = config

    def status(self):
        return dict(ready=True, enabled=True, threshold=self.threshold, error=None,
                    algorithm="SetFit + MiniLM", metadata=self.config.get("metadata", {"version": self.version}))

    def predict(self, text):
        scores = self.np.asarray(self.model.predict_proba([text], as_numpy=True))[0]
        index = int(scores.argmax())
        return dict(reason=self.labels[index], confidence=round(float(scores[index]), 4),
                    used=bool(scores[index] >= self.threshold), model_version=self.version)


def load_reason_model(reason_config):
    config = json.loads(Path(reason_config).read_text())
    if not Path(config["path"]).is_absolute():
        config["path"] = str((Path(reason_config).resolve().parent/config["path"]).resolve())
    if config["kind"] == "tfidf":
        model = ReasonClassifier(config["path"], threshold=config["threshold"])
        if not model.status()["ready"]:
            raise ValueError("Reason artifact is missing or invalid.")
        return model
    if config["kind"] == "setfit":
        return SetFitReason(config)
    raise ValueError("Unknown reason model kind.")


def create_app(reason_config=None, matching_index=None):
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    model = load_reason_model(reason_config) if reason_config else None
    matcher = CatalogMatcher(matching_index) if matching_index else None
    token = os.environ.get("GHOST_ML_API_TOKEN")

    @application.before_request
    def authorize():
        if token and not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            return jsonify(error="Unauthorized"), 401

    @application.get("/health")
    def health():
        return jsonify(ready=True, reason=model is not None, matching=matcher is not None)

    @application.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or not 1 <= len(payload["text"].strip()) <= 10000:
            return jsonify(error="text must contain 1–10,000 characters"), 400
        start = time.perf_counter()
        text = payload["text"].strip()
        result = dict(reason=model.predict(text) if model else None,
                      matches=matcher.search(text) if matcher else [],
                      review_required=True,
                      note="Apply only to confirmed unmet requests. Similarity does not establish SKU compatibility or availability.")
        logging.getLogger("ghostdemand.ml").info("prediction latency_ms=%.2f reason_used=%s",
                    1000*(time.perf_counter()-start), bool(result["reason"] and result["reason"]["used"]))
        # Log metadata only, never the customer message.
        return jsonify(result)

    return application


def configured_app():
    """Gunicorn factory: read paths from environment and load once per worker."""
    return create_app(os.environ.get("GHOST_REASON_CONFIG"), os.environ.get("GHOST_MATCH_INDEX"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason-config", default="ml_workflow/work/baseline/serving.json")
    parser.add_argument("--matching-index")
    parser.add_argument("--port", type=int, default=5061)
    args = parser.parse_args()
    create_app(args.reason_config, args.matching_index).run(host="127.0.0.1", port=args.port, debug=False)
