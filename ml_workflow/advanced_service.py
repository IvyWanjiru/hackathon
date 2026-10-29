"""Optional deployment exercise for the later models. No application writes.

GHOST_ENTITY_MODEL -> trusted saved model directory
GHOST_RECOVERY_MODEL -> trusted .cbm file
GHOST_FORECAST_DIR -> directory containing model.txt and serving.json
"""
import json
import math
import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, request

from .advanced import FORECAST_FEATURES, RECOVERY_CATEGORICAL, RECOVERY_FEATURES, RECOVERY_NUMERIC


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64*1024
    token = os.environ.get("GHOST_ML_API_TOKEN")
    entity_model = os.environ.get("GHOST_ENTITY_MODEL")
    recovery_model = os.environ.get("GHOST_RECOVERY_MODEL")
    forecast_dir = os.environ.get("GHOST_FORECAST_DIR")
    entity = recovery = forecast = None
    if entity_model:
        from transformers import pipeline
        entity = pipeline("token-classification", model=entity_model, tokenizer=entity_model, aggregation_strategy="simple")
    if recovery_model:
        from catboost import CatBoostClassifier
        recovery = CatBoostClassifier()
        recovery.load_model(recovery_model)
    if forecast_dir:
        import lightgbm as lgb
        forecast = lgb.Booster(model_file=str(Path(forecast_dir)/"model.txt"))
        categories = json.loads((Path(forecast_dir)/"serving.json").read_text())["categories"]

    @app.before_request
    def authenticate():
        if token and not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            return jsonify(error="Unauthorized"), 401

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.get("/health")
    def health():
        return jsonify(entities=entity is not None, recovery=recovery is not None, forecast=forecast is not None)

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def numeric(data, names):
        for name in names:
            value = data.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, nonnegative number.")

    @app.post("/entities")
    def entities():
        if entity is None:
            return jsonify(error="Entity model not configured"), 503
        text = payload().get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 10000:
            raise ValueError("text must contain 1–10,000 characters")
        # Match the training limit rather than silently omitting trailing fields.
        if len(entity.tokenizer(text)["input_ids"]) > 256:
            raise ValueError("Segment this message into shorter requests before extraction.")
        spans = [dict(label=row["entity_group"], text=text[int(row["start"]):int(row["end"])],
                      start=int(row["start"]), end=int(row["end"]), score=float(row["score"])) for row in entity(text)]
        return jsonify(entities=spans, requires_review=True)

    @app.post("/recovery")
    def recover():
        if recovery is None:
            return jsonify(error="Recovery model not configured"), 503
        data = payload()
        numeric(data, RECOVERY_NUMERIC)
        if any(not isinstance(data.get(name), str) for name in RECOVERY_CATEGORICAL):
            raise ValueError("All categorical features must be strings.")
        import pandas as pd
        frame = pd.DataFrame([{name: data[name] for name in RECOVERY_FEATURES}])
        score = float(recovery.predict_proba(frame)[0, 1])
        return jsonify(score=score, target="observed conversion within 14 days", calibrated=False)

    @app.post("/forecast")
    def predict_sales():
        if forecast is None:
            return jsonify(error="Forecast model not configured"), 503
        data = payload()
        numeric(data, FORECAST_FEATURES[1:])
        if data.get("sku") not in categories or not 1 <= data["week_of_year"] <= 53:
            raise ValueError("Known sku and ISO week_of_year 1–53 required.")
        import pandas as pd
        frame = pd.DataFrame([{name: data[name] for name in FORECAST_FEATURES}])
        frame["sku"] = pd.Categorical(frame.sku, categories=categories)
        return jsonify(predicted_quantity=max(0., float(forecast.predict(frame)[0])), horizon="one week", target="completed sales")

    return app
