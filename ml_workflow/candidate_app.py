"""Optional application entry point. Run it yourself after evaluating a candidate.

Loads the existing application with a chosen reason backend. The source app.py
and production model files stay unchanged; select an explicit staging DB below.
Demo mode continues using the app's separate demo classifier.
"""
import os

from demand_ml import TrainingError
from .service import load_reason_model


class OfflineCandidate:
    def __init__(self, model):
        self.model = model

    def status(self):
        return self.model.status()

    def predict(self, text):
        return self.model.predict(text)

    def train(self, reviewed):
        raise TrainingError("This candidate is trained offline with frozen splits. Run the ML workflow, evaluate, then restart with the new version.")


def create_app():
    config = os.environ.get("GHOST_REASON_CONFIG")
    if not config or not os.environ.get("GHOST_DB"):
        raise ValueError("Set GHOST_REASON_CONFIG and an explicit GHOST_DB path. Use a separate staging DB while learning.")
    import app as ghost
    ghost.ML = OfflineCandidate(load_reason_model(config))
    return ghost.app
