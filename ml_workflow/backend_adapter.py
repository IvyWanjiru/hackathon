"""Opt-in integration example; app.py does not import this module.

Insert the RemoteReason adapter only after comparing the candidate in shadow
mode. On failure or a low score, preserve the app's existing rule fallback.
"""
import json
import os
import urllib.error
import urllib.request


class RemoteReason:
    def __init__(self, url="http://127.0.0.1:5061/predict", timeout=1.5):
        self.url, self.timeout = url, timeout

    def predict(self, text):
        headers = {"Content-Type": "application/json"}
        if os.environ.get("GHOST_ML_API_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["GHOST_ML_API_TOKEN"]
        request = urllib.request.Request(self.url, data=json.dumps({"text": text}).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response).get("reason")
            if not isinstance(result, dict):
                return None
            from demand_ml import REASON_LABELS
            if (result.get("reason") not in REASON_LABELS or type(result.get("used")) is not bool
                    or type(result.get("confidence")) not in (int, float)
                    or not 0 <= result["confidence"] <= 1 or not isinstance(result.get("model_version"), str)):
                return None
            return result
        except (OSError, urllib.error.URLError, ValueError, TypeError):
            return None
