import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app as ghost
from demand_ml import ReasonClassifier, REASON_LABELS, TrainingError, STARTER_PATH

STARTER_COUNT = len(json.loads(STARTER_PATH.read_text()))


class GhostDemandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.trained_path = Path(cls.shared.name) / "trained.json"
        ReasonClassifier(cls.trained_path).train([])

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "model.json"
        shutil.copy(self.trained_path, self.model_path)
        self.model = ReasonClassifier(self.model_path)
        self.patches = [
            patch.object(ghost, "DB", str(self.root / "test.db")),
            patch.object(ghost, "ML", self.model),
            patch.object(ghost, "llm_extract", return_value=None),
        ]
        for item in self.patches:
            item.start()
        self.client = ghost.app.test_client()
        ghost.app.config["TESTING"] = True

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def capture(self, text, source="chatbot"):
        response = self.client.post("/api/signals", json=dict(text=text, source=source))
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()

    def test_capture_uses_model_and_persists_provenance(self):
        signal = self.capture("The delivery would arrive after our deadline")
        self.assertEqual(signal["reason"], "lead_time")
        self.assertEqual(signal["reason_engine"], "local_ml")
        self.assertGreaterEqual(signal["ml_confidence"], 0.55)
        recent = self.client.get("/api/insights").get_json()["recent"][0]
        self.assertEqual(recent["model_version"], signal["model_version"])
        self.assertEqual(recent["ml_reason"], "lead_time")

    def test_reload_predicts_identically(self):
        text = "The cement bags are out of stock today"
        self.assertEqual(self.model.predict(text), ReasonClassifier(self.model_path).predict(text))
        artifact = json.loads(self.model_path.read_text())
        self.assertEqual(set(artifact["classes"]), set(REASON_LABELS))
        self.assertNotIn("raw", artifact)

    def test_low_score_preserves_rules_and_suggestion(self):
        self.model.threshold = 1.0
        signal = self.capture("The delivery would arrive after our deadline")
        self.assertEqual(signal["reason_engine"], "rules")
        self.assertEqual(signal["ml_reason"], "lead_time")
        self.assertEqual(signal["reason"], ghost.rule_extract(signal["raw"], "chatbot")["reason"])

    def test_missing_corrupt_and_disabled_models_fall_back(self):
        self.model_path.unlink()
        self.assertEqual(self.capture("Out of stock: 20 pcs 16mm rebar")["reason_engine"], "rules")
        self.model_path.write_text('{"format_version":1}')
        self.assertFalse(self.client.get("/api/model").get_json()["ready"])
        self.assertEqual(self.capture("Need white cement")["reason_engine"], "rules")
        self.model_path.write_text('[]')
        self.assertFalse(self.client.get("/api/model").get_json()["ready"])
        shutil.copy(self.trained_path, self.model_path)
        self.model.enabled = False
        self.assertEqual(self.capture("The cement bags are out of stock today")["reason_engine"], "rules")

    def test_review_corrects_insights_and_teaches_next_model(self):
        text = "zyxqv qzxvv"
        self.assertIsNone(self.model.predict(text))
        signal = self.capture(text)
        self.assertEqual(signal["reason"], "unknown")
        response = self.client.patch(f"/api/signals/{signal['id']}/review", json=dict(reason="lead_time"))
        self.assertEqual(response.status_code, 200)
        insights = self.client.get("/api/insights").get_json()
        self.assertEqual(insights["demand"][0]["top_reason"], "lead_time")
        self.assertEqual(insights["recent"][0]["reason"], "unknown")
        self.assertEqual(self.client.get("/api/model").get_json()["pending_reviews"], 1)
        response = self.client.post("/api/model/train")
        self.assertEqual(response.status_code, 200, response.get_json())
        status = response.get_json()
        self.assertEqual(status["metadata"]["reviewed_examples"], 1)
        self.assertEqual(status["metadata"]["examples"], STARTER_COUNT + 1)
        self.assertEqual(status["pending_reviews"], 0)
        learned = self.capture(text)
        self.assertEqual(learned["reason"], "lead_time")
        self.assertEqual(learned["reason_engine"], "local_ml")
        self.assertNotEqual(learned["model_version"], signal["model_version"])

    def test_training_ignores_unreviewed_and_deduplicates_reviews(self):
        self.capture("This unverified example must not become ground truth")
        a = self.capture("zyxqv qzxvv")
        b = self.capture("ZYXQV   QZXVV")
        for item, reason in ((a, "abandoned"), (b, "lead_time")):
            self.client.patch(f"/api/signals/{item['id']}/review", json=dict(reason=reason))
        status = self.client.post("/api/model/train").get_json()
        self.assertEqual(status["metadata"]["examples"], STARTER_COUNT + 1)
        self.assertEqual(status["metadata"]["reviewed_examples"], 1)
        self.assertEqual(self.model.predict("zyxqv qzxvv")["reason"], "lead_time")

    def test_starter_evaluation_is_identified(self):
        status = self.client.get("/api/model").get_json()
        self.assertTrue(status["ready"])
        meta = status["metadata"]
        self.assertEqual(meta["starter_examples"], STARTER_COUNT)
        self.assertEqual(meta["reviewed_examples"], 0)
        self.assertEqual(meta["evaluation"]["examples"], STARTER_COUNT // 4)
        self.assertEqual(meta["evaluation"]["starter_examples"], STARTER_COUNT // 4)

    def test_reviews_during_training_remain_pending(self):
        signal = self.capture("zyxqv qzxvv")
        url = f"/api/signals/{signal['id']}/review"
        self.client.patch(url, json=dict(reason="abandoned"))
        with ghost.app.app_context():
            snapshot = ghost.reviewed_examples()
        # Another review arrives after the training data was read.
        self.client.patch(url, json=dict(reason="lead_time"))
        self.model.train(snapshot)
        self.assertEqual(self.client.get("/api/model").get_json()["pending_reviews"], 1)

    def test_concurrent_first_requests_migrate_safely(self):
        def first_request(_):
            with ghost.app.test_client() as client:
                return client.get("/api/insights").status_code
        with ThreadPoolExecutor(max_workers=4) as workers:
            self.assertEqual(list(workers.map(first_request, range(4))), [200] * 4)

    def test_input_validation_and_bulk_are_atomic(self):
        for payload in ([], {}, {"text": 42}, {"text":" "}, {"text":"x"*10001},
                        {"text":"cement", "source":"unknown"}, {"text":"cement", "customer":[]}):
            with self.subTest(payload_type=type(payload).__name__):
                response = self.client.post("/api/signals", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        response = self.client.post("/api/signals/bulk", json={"items":[{"text":"cement"}, {"text":12}]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/insights").get_json()["kpis"]["signals"], 0)
        response = self.client.post("/api/signals/bulk", json={"items":[{"text":"cement"}, {"text":"tiles"}]})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.get_json()), 2)

    def test_invalid_review_and_unknown_signal(self):
        self.assertEqual(self.client.patch("/api/signals/999/review", json={"reason":"lead_time"}).status_code, 404)
        self.assertEqual(self.client.patch("/api/signals/999/review", json={"reason":"nonsense"}).status_code, 400)
        self.assertEqual(self.client.post("/api/signals", data="broken", content_type="application/json").status_code, 400)

    def test_legacy_database_migration_preserves_rows(self):
        with closing(sqlite3.connect(ghost.DB)) as connection:
            connection.execute("""CREATE TABLE signals(id INTEGER PRIMARY KEY, ts TEXT, source TEXT,
                raw TEXT, product TEXT, pkey TEXT, qty INTEGER, budget REAL, reason TEXT,
                urgency INTEGER, sentiment TEXT, location TEXT, customer TEXT)""")
            connection.execute("""INSERT INTO signals VALUES
                (1,'2026-01-01','whatsapp','Need cement','Cement','cement',2,500,'not_stocked',1,'neutral',NULL,NULL)""")
            connection.commit()
        recent = self.client.get("/api/insights").get_json()["recent"][0]
        self.assertEqual(recent["id"], 1)
        self.assertEqual(recent["product"], "Cement")
        self.assertEqual(recent["reason_engine"], "rules")
        self.assertIsNone(recent["reviewed_reason"])

    def test_bad_external_model_fields_are_ignored(self):
        bad = {"product":[], "reason":"other", "qty":-4, "urgency":99,
               "budget":float("nan"), "sentiment":{}, "location":20}
        self.model.enabled = False
        with patch.object(ghost, "llm_extract", return_value=bad):
            signal = self.capture("Need 20 pcs rebar")
        self.assertEqual(signal["qty"], 20)
        self.assertEqual(signal["reason_engine"], "rules")
        self.assertEqual(signal["urgency"], 1)
        self.assertIsNone(signal["budget"])

    def test_rule_fields_and_whatsapp_remain_supported(self):
        signal = self.capture("Hi, do you have 50 bags of white cement? Need it urgent, deliver to Kisumu, budget KES 40,000")
        self.assertEqual(signal["product"], "White Cement")
        self.assertEqual(signal["qty"], 50)
        self.assertEqual(signal["budget"], 40000)
        self.assertEqual(signal["location"], "Kisumu")
        self.assertEqual(signal["urgency"], 3)
        response = self.client.post("/webhook/whatsapp", data={"Body":"Need paint", "From":"test-customer"})
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/api/insights").get_json()["recent"][0]["source"], "whatsapp")

    def test_failed_training_keeps_current_model(self):
        version = self.model.status()["metadata"]["version"]
        with self.assertRaises(TrainingError):
            self.model.train([dict(text="invalid example", reason="unknown")])
        self.assertEqual(ReasonClassifier(self.model_path).status()["metadata"]["version"], version)

    def test_cli_and_homepage(self):
        result = ghost.app.test_cli_runner().invoke(args=["train-model"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["examples"], STARTER_COUNT)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Missed requests.", response.data)
        self.assertIn(b'href="/dashboard"', response.data)
        self.assertIn(b'href="/demo"', response.data)
        self.assertNotIn(b'id="app-config"', response.data)
        dashboard = self.client.get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Follow-up queue", dashboard.data)
        self.assertNotIn(b"Learning lab", dashboard.data)
        self.assertNotIn(b'id="train-model"', dashboard.data)
        self.assertIn(b'"demo": false', dashboard.data)
        demo = self.client.get("/demo")
        self.assertEqual(demo.status_code, 200)
        self.assertIn(b'"demo": true', demo.data)
        self.assertIn(b'href="/dashboard"', demo.data)


if __name__ == "__main__":
    unittest.main()
