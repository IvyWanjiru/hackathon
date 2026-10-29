import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from demand_ml import ReasonClassifier
from ml_workflow.common import ROOT, linked_groups, load_partitions, read_csv, validate_reasons, write_csv
from ml_workflow.eda import profile
from ml_workflow.prepare_data import prepare
from ml_workflow.split_data import split_data
from ml_workflow.train_reason import train
from ml_workflow.matching import train as train_matching, CatalogMatcher, validate
from ml_workflow.service import create_app
from ml_workflow.advanced import entity_rows


class LearningWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.csv = prepare(cls.root/"data")
        cls.parts = split_data(cls.csv, cls.root/"splits")
        cls.report = train(cls.root/"splits", cls.root/"candidate")
        cls.matching_report = train_matching(ROOT/"ml_workflow/data/catalog.csv", ROOT/"ml_workflow/data/queries.csv", cls.root/"matching")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_starter_provenance_and_eda(self):
        report = profile(self.csv, self.root/"eda")
        self.assertEqual(report["rows"], 108)
        self.assertEqual(report["synthetic_rows"], 108)
        self.assertEqual(report["conflicting_text_groups"], 0)
        self.assertEqual(report["missing"]["captured_at"], 108)

    def test_groups_link_transitively(self):
        rows = [dict(text="a", conversation_id="one", group_id="x"),
                dict(text="b", conversation_id="one", group_id="y"),
                dict(text="B!", conversation_id="two", group_id="z")]
        self.assertEqual(len(set(linked_groups(rows))), 1)

    def test_validation_rejects_conflicting_labels(self):
        rows = read_csv(self.csv)
        duplicate = {**rows[0], "id": "conflict", "reason": "abandoned"}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            validate_reasons(rows + [duplicate])

    def test_train_serving_parity_and_no_test_refit(self):
        model = ReasonClassifier(self.root/"candidate/reason_classifier.json")
        meta = model.status()["metadata"]
        self.assertEqual(meta["examples"], len(self.parts["train"]))
        self.assertEqual(meta["evaluation"]["examples"], len(self.parts["test"]))
        prediction = model.predict("Requested headphones are sold out")
        self.assertIn(prediction["reason"], model._model[3])
        self.assertEqual(prediction, ReasonClassifier(model.path).predict("Requested headphones are sold out"))
        self.assertIsNone(model.predict("☃☃☃"))

    def test_unknown_vocabulary_is_counted_as_deferred_and_missing(self):
        import numpy as np
        from ml_workflow.evaluation import evaluate
        from demand_ml import REASON_LABELS
        metrics = evaluate([dict(reason="out_of_stock", source="chatbot", is_synthetic="1")],
                           np.asarray([[1/6]*6]), list(REASON_LABELS), 0., known_mask=[False])
        self.assertEqual(metrics["prediction_missing"], 1)
        self.assertEqual(metrics["coverage"], 0.)
        self.assertEqual(metrics["accuracy"], 0.)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 1)

    def test_cross_partition_text_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, rows in self.parts.items():
                copied = copy.deepcopy(rows)
                if name == "test":
                    copied[0]["text"] = self.parts["train"][0]["text"]
                    copied[0]["reason"] = self.parts["train"][0]["reason"]
                write_csv(Path(temp)/f"{name}.csv", copied)
            with self.assertRaises(ValueError):
                load_partitions(temp)

    def test_frozen_manifest_detects_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            import shutil
            shutil.copytree(self.root/"splits", temp, dirs_exist_ok=True)
            with (Path(temp)/"train.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "frozen split"):
                load_partitions(temp)

    def test_service_validates_inputs_and_does_not_open_app(self):
        client = create_app(self.root/"candidate/serving.json", self.root/"matching/index.json").test_client()
        self.assertEqual(client.post("/predict", json={"text": 1}).status_code, 400)
        self.assertEqual(client.post("/predict", json=[]).status_code, 400)
        self.assertEqual(client.post("/predict", json={"text": "x"*10001}).status_code, 400)
        response = client.post("/predict", json={"text": "20000mAh power bank out of stock"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["review_required"])
        self.assertTrue(response.get_json()["matches"][0]["requires_review"])

    def test_service_optional_token(self):
        with patch.dict(os.environ, {"GHOST_ML_API_TOKEN": "test-token"}):
            client = create_app(self.root/"candidate/serving.json").test_client()
            self.assertEqual(client.get("/health").status_code, 401)
            self.assertEqual(client.get("/health", headers={"Authorization": "Bearer test-token"}).status_code, 200)

    def test_matcher_does_not_merge_variant_skus(self):
        matcher = CatalogMatcher(self.root/"matching/index.json")
        results = matcher.search("USB-C power bank 20000mAh")
        self.assertEqual(results[0]["product_id"], "P04")
        self.assertNotEqual(results[0]["product_id"], "P03")

    def test_matching_query_leakage_rejected(self):
        catalog = read_csv(ROOT/"ml_workflow/data/catalog.csv")
        rows = read_csv(ROOT/"ml_workflow/data/queries.csv")
        rows[-1]["query"] = rows[0]["query"]
        with self.assertRaisesRegex(ValueError, "cross partitions"):
            validate(catalog, rows)

    def test_schema_example_is_not_a_training_dataset(self):
        with self.assertRaisesRegex(ValueError, "not a training dataset"):
            entity_rows(ROOT/"ml_workflow/data/entities_schema_example.jsonl")

    def test_candidate_entry_point_uses_explicit_db_and_blocks_online_training(self):
        import app as ghost
        from ml_workflow.candidate_app import create_app as candidate_app
        from demand_ml import TrainingError
        with patch.dict(os.environ, {"GHOST_REASON_CONFIG": str(self.root/"candidate/serving.json"),
                                     "GHOST_DB": str(self.root/"staging.db")}), patch.object(ghost, "ML"):
            application = candidate_app()
            self.assertIs(application, ghost.app)
            self.assertTrue(ghost.ML.status()["ready"])
            with self.assertRaisesRegex(TrainingError, "offline"):
                ghost.ML.train([])


if __name__ == "__main__":
    unittest.main()
