import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app as ghost
from demand_ml import ReasonClassifier


class AutomaticIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.trained = Path(cls.shared.name) / 'trained.json'
        ReasonClassifier(cls.trained).train([])

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_path = self.root / 'model.json'
        shutil.copy(self.trained, self.model_path)
        self.patches = [patch.object(ghost, 'DB', str(self.root / 'live.db')),
                        patch.object(ghost, 'DEMO_DB', str(self.root / 'demo.db')),
                        patch.object(ghost, 'ML', ReasonClassifier(self.model_path)),
                        patch.object(ghost, 'DEMO_ML', ReasonClassifier(self.model_path)),
                        patch.object(ghost, 'llm_extract', return_value=None),
                        patch.dict(os.environ, {'GHOST_INGEST_TOKEN':'test-relay-secret'})]
        for item in self.patches: item.start()
        ghost.app.config['TESTING'] = True
        self.client = ghost.app.test_client()
        self.auth = {'Authorization':'Bearer test-relay-secret'}
        self.demo = {'X-Ghost-Demo':'1'}

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def event(self, channel='whatsapp', **changes):
        payload = dict(event_id='event-1', text='The delivery would arrive after our deadline',
                       outcome='unmet', currency='KES', customer='Customer A')
        payload.update(changes)
        return self.client.post('/api/ingest/' + channel, json=payload, headers=self.auth)

    def activity(self, headers=None):
        response = self.client.get('/api/intake', headers=headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def test_authentication_is_required_before_any_write(self):
        self.assertEqual(self.client.post('/api/ingest/email', json={}).status_code, 401)
        self.assertEqual(self.client.post('/api/ingest/email', json={}, headers={'Authorization':'Bearer é'}).status_code, 401)
        with patch.dict(os.environ, {'GHOST_INGEST_TOKEN':''}):
            self.assertEqual(self.event().status_code, 503)
        self.assertFalse((self.root / 'live.db').exists())
        self.assertFalse((self.root / 'demo.db').exists())
        response = self.client.post('/api/ingest/email', json={}, headers={**self.auth, **self.demo})
        self.assertEqual(response.status_code, 400)

    def test_capture_runs_the_backend_classifier_without_creating_training_labels(self):
        response = self.event('email')
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertTrue(result['captured'])
        row = self.client.get('/api/signals/' + str(result['event']['signal_id'])).get_json()
        self.assertEqual(row['source'], 'email')
        self.assertEqual(row['reason'], 'lead_time')
        self.assertEqual(row['reason_engine'], 'local_ml')
        ghost.llm_extract.assert_not_called()
        self.assertIsNone(row['reviewed_reason'])
        self.assertTrue(row['needs_review'])
        activity = self.activity()
        self.assertEqual(activity['summary']['captured'], 1)
        self.assertNotIn('ml_confidence', activity['items'][0])
        self.assertNotIn('model_version', activity['items'][0])

    def test_retries_are_idempotent_and_changed_content_is_rejected(self):
        first = self.event().get_json()
        repeat = self.event().get_json()
        self.assertEqual(first['event']['signal_id'], repeat['event']['signal_id'])
        self.assertTrue(repeat['deduplicated'])
        self.assertEqual(self.event(text='Different content').status_code, 409)
        self.assertEqual(self.activity()['summary']['total'], 1)
        self.assertEqual(self.client.get('/api/insights?days=all').get_json()['kpis']['signals'], 1)

    def test_questions_and_negations_need_context_and_fulfilled_events_do_not_count_as_loss(self):
        for index, text in enumerate(['Do you have wireless earbuds?', 'When will delivery arrive?',
                                      'The earbuds are not sold out.']):
            response = self.event(event_id=str(index), text=text, outcome=None).get_json()
            self.assertEqual(response['event']['state'], 'needs_context')
            self.assertFalse(response['captured'])
        result = self.event(event_id='fulfilled',text='They were sold out but the order was fulfilled.',outcome='fulfilled').get_json()
        self.assertEqual(result['event']['state'], 'ignored')
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'], 0)

    def test_outcome_resolution_is_separate_from_blocker_verification(self):
        event = self.event(text='Do you have earbuds?', outcome=None).get_json()['event']
        response = self.client.post(f"/api/intake/{event['id']}/resolve", json={'action':'confirm_unmet'})
        self.assertEqual(response.status_code, 200, response.get_json())
        row = self.client.get('/api/signals/' + str(response.get_json()['event']['signal_id'])).get_json()
        self.assertIsNone(row['reviewed_reason'])
        self.assertTrue(row['needs_review'])
        again = self.client.post(f"/api/intake/{event['id']}/resolve", json={'action':'confirm_unmet'}).get_json()
        self.assertTrue(again['deduplicated'])
        self.assertEqual(self.activity()['summary']['captured'], 1)

    def test_failed_analysis_is_retained_and_can_retry_safely(self):
        with patch.object(ghost, 'draft_signal', side_effect=RuntimeError('Private failure detail')):
            failed = self.event()
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn('Private failure detail', failed.get_data(as_text=True))
        self.assertEqual(self.activity()['summary']['failed'], 1)
        success = self.event().get_json()
        self.assertTrue(success['captured'])
        self.assertEqual(self.activity()['summary']['failed'], 0)
        self.assertEqual(self.activity()['summary']['total'], 1)

    def test_voice_events_need_transcripts_and_cannot_forge_verified_labels(self):
        invalid = self.client.post('/api/ingest/voice_notes',json={'event_id':'v-1','audio_url':'https://example.invalid/audio'},headers=self.auth)
        self.assertEqual(invalid.status_code, 400)
        valid = self.client.post('/api/ingest/voice_notes',json=dict(event_id='v-1',transcript='The requested earbuds are sold out.'),headers=self.auth)
        self.assertTrue(valid.get_json()['captured'])
        for forbidden in ('verified','reviewed_reason','fields','reason'):
            self.assertEqual(self.event(**{forbidden:True}).status_code, 400)

    def test_demo_simulation_covers_five_channels_without_touching_live_data(self):
        self.assertEqual(self.client.post('/api/demo/intake').status_code, 400)
        for _ in range(2):
            response = self.client.post('/api/demo/intake',json={},headers=self.demo)
            self.assertEqual(response.status_code, 200, response.get_json())
        data = self.activity(self.demo)
        self.assertEqual(data['summary']['total'], 5)
        self.assertEqual(data['summary']['captured'], 4)
        self.assertEqual(data['summary']['needs_context'], 1)
        self.assertTrue(all(channel['events'] == 1 for channel in data['channels']))
        self.assertFalse((self.root / 'live.db').exists())
        self.assertEqual(self.client.get('/api/insights?days=all',headers=self.demo).get_json()['kpis']['signals'], 4)

    def test_concurrent_delivery_retries_count_once(self):
        def deliver(_):
            with ghost.app.test_client() as client:
                response=client.post('/api/ingest/chatbot',json=dict(event_id='retry-1',text='Requested earbuds are sold out.'),headers=self.auth)
                return response.status_code, response.get_json()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(deliver,range(8)))
        self.assertTrue(all(status == 200 for status,_ in results), results)
        self.assertEqual(len({result['event']['signal_id'] for _,result in results}), 1)
        self.assertEqual(self.activity()['summary']['total'], 1)
        self.assertEqual(self.client.get('/api/insights?days=all').get_json()['kpis']['signals'], 1)

    def test_startup_initializes_missing_baseline_without_overwriting_saved_weights(self):
        before = self.model_path.read_bytes()
        missing = ReasonClassifier(self.root / 'missing-demo-model.json')
        with patch.object(ghost, 'DEMO_ML', missing), patch.dict(os.environ, {'GHOST_BOOTSTRAP_MODELS':'1'}):
            ghost.bootstrap_models()
        self.assertTrue(missing.status()['ready'])
        self.assertEqual(before, self.model_path.read_bytes())
        disabled = ReasonClassifier(self.root / 'disabled-bootstrap.json')
        with patch.object(ghost, 'DEMO_ML', disabled), patch.dict(os.environ, {'GHOST_BOOTSTRAP_MODELS':'0'}):
            ghost.bootstrap_models()
        self.assertFalse(disabled.path.exists())


if __name__ == '__main__':
    unittest.main()
