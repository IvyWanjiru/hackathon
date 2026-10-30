import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as ghost
from demand_ml import ReasonClassifier
from presentation_data import notebook_evaluation


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.trained = Path(cls.shared.name) / 'model.json'
        ReasonClassifier(cls.trained).train([])

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        demo_model = self.root / 'demo-model.json'
        shutil.copy(self.trained, demo_model)
        self.patches = [patch.object(ghost, 'DB', str(self.root / 'live.db')),
                        patch.object(ghost, 'DEMO_DB', str(self.root / 'demo.db')),
                        patch.object(ghost, 'DEMO_ML', ReasonClassifier(demo_model)),
                        patch.object(ghost, 'llm_extract', return_value=None)]
        for item in self.patches:
            item.start()
        self.client = ghost.app.test_client()
        self.demo = {'X-Ghost-Demo': '1'}

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_opening_and_downloads_do_not_seed_or_open_a_database(self):
        page = self.client.get('/present')
        self.assertEqual(page.status_code, 302)
        self.assertEqual(page.headers['Location'], '/demo#activity')
        for path, filename in (('/present/notebook', 'GhostDemand_Training_Improved.ipynb'),
                               ('/present/notes', 'GhostDemand_Panel_Notes.md')):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn('attachment', response.headers['Content-Disposition'])
            self.assertIn(filename, response.headers['Content-Disposition'])
            response.close()
        self.assertFalse((self.root / 'live.db').exists())
        self.assertFalse((self.root / 'demo.db').exists())

    def test_business_frontend_does_not_show_training_internals(self):
        response = self.client.get('/demo')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Channel activity', response.data)
        self.assertNotIn(b'Test macro F1', response.data)
        self.assertNotIn(b'TF-IDF', response.data)
        self.assertNotIn(b'Train model', response.data)
        self.assertNotIn(b'Fraud classifier', response.data)
        self.assertNotIn(b'presentation.js', response.data)

    def test_demo_rehearsal_is_idempotent_and_never_writes_live_requests(self):
        for _ in range(2):
            self.assertEqual(self.client.post('/api/demo/seed', json={}, headers=self.demo).status_code, 200)
        baseline = self.client.get('/api/insights?days=all', headers=self.demo).get_json()['kpis']['signals']
        self.assertEqual(baseline, 24)
        payload = dict(text='The requested 2 wireless earbuds are sold out. Total budget KES 6,000. The customer needs them today.',
                       source='whatsapp', customer='Panel sample', currency='KES', external_id='panel-test-1')
        preview = self.client.post('/api/signals/preview', json=payload, headers=self.demo)
        self.assertEqual(preview.status_code, 200, preview.get_json())
        draft = preview.get_json()
        self.assertEqual(self.client.get('/api/insights?days=all', headers=self.demo).get_json()['kpis']['signals'], baseline)
        saved_payload = {key:value for key,value in draft.items() if key not in ('fields', 'warnings')}
        saved_payload.update(verified=True, fields=dict(product='Wireless Earbuds', qty=2, budget=6000,
                                                      currency='KES', reason='out_of_stock'))
        saved = self.client.post('/api/signals', json=saved_payload, headers=self.demo)
        self.assertEqual(saved.status_code, 201, saved.get_json())
        row = saved.get_json()
        self.assertEqual(row['reviewed_reason'], 'out_of_stock')
        repeat = self.client.post('/api/signals', json=saved_payload, headers=self.demo).get_json()
        self.assertEqual(repeat['id'], row['id'])
        self.assertTrue(repeat['deduplicated'])
        result = self.client.patch(f"/api/signals/{row['id']}", headers=self.demo,
                                  json=dict(owner='Sam', follow_up_on='2026-10-29',
                                            next_action='Confirm restock options.', status='follow_up'))
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertEqual(result.get_json()['owner'], 'Sam')
        self.assertEqual(result.get_json()['status'], 'follow_up')
        insights = self.client.get('/api/insights?days=all', headers=self.demo).get_json()
        self.assertEqual(insights['kpis']['signals'], baseline + 1)
        self.assertEqual(insights['kpis']['recovered'], 1)
        self.assertEqual(insights['kpis']['recovered_value'], {'KES':1900})
        self.assertFalse((self.root / 'live.db').exists())


class EvaluationProvenanceTests(unittest.TestCase):
    def test_snapshot_requires_current_notebook_and_valid_metrics(self):
        self.assertIsNotNone(notebook_evaluation(ghost.ROOT))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ml_workflow').mkdir()
            (root / 'notebooks').mkdir()
            notebook = root / 'notebooks' / 'GhostDemand_Training_Improved.ipynb'
            notebook.write_text('{"cells":[]}')
            snapshot = root / 'ml_workflow' / 'presentation-evaluation.json'
            record = json.loads((ghost.ROOT / 'ml_workflow' / snapshot.name).read_text())
            record['notebook_sha256'] = hashlib.sha256(notebook.read_bytes()).hexdigest()
            snapshot.write_text(json.dumps(record))
            self.assertIsNotNone(notebook_evaluation(root))
            notebook.write_text('{"cells":[1]}')
            self.assertIsNone(notebook_evaluation(root))
            record['notebook_sha256'] = hashlib.sha256(notebook.read_bytes()).hexdigest()
            for invalid in ([], {**record, 'macro_f1':float('nan')},
                            {**record, 'split_sizes':None}, {**record, 'synthetic':False}):
                snapshot.write_text(json.dumps(invalid))
                self.assertIsNone(notebook_evaluation(root))


if __name__ == '__main__':
    unittest.main()
