import csv
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import app as ghost
from demand_ml import ReasonClassifier


class RetailWorkflowTests(unittest.TestCase):
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
        root = Path(self.temp.name)
        self.model_path = root / 'model.json'
        shutil.copy(self.trained, self.model_path)
        self.demo_model = ReasonClassifier(root / 'demo-model.json')
        self.patches = [patch.object(ghost, 'DB', str(root / 'live.db')),
                        patch.object(ghost, 'DEMO_DB', str(root / 'demo.db')),
                        patch.object(ghost, 'ML', ReasonClassifier(self.model_path)),
                        patch.object(ghost, 'DEMO_ML', self.demo_model),
                        patch.object(ghost, 'llm_extract', return_value=None)]
        for p in self.patches:
            p.start()
        ghost.app.config['TESTING'] = True
        self.client = ghost.app.test_client()
        self.demo = {'X-Ghost-Demo':'1'}

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def capture(self, **changes):
        payload = dict(text='Need wireless earbuds', source='whatsapp')
        payload.update(changes)
        response = self.client.post('/api/signals', json=payload)
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()

    def test_preview_is_editable_and_does_not_write_until_confirmed(self):
        response = self.client.post('/api/signals/preview', json=dict(text='Need 2 earbuds budget USD 40',source='chatbot',external_id='capture-1',currency='KES'))
        self.assertEqual(response.status_code, 200)
        draft = response.get_json()
        self.assertEqual(draft['fields']['currency'], 'USD')
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'], 0)
        payload = {k:v for k,v in draft.items() if k not in ('fields','warnings')}
        payload.update(fields=dict(product='Wireless Earbuds',qty=2,budget=38,currency='USD',reason='out_of_stock'),verified=True)
        saved = self.client.post('/api/signals', json=payload)
        self.assertEqual(saved.status_code, 201, saved.get_json())
        row = saved.get_json()
        self.assertEqual(row['product'], 'Wireless Earbuds')
        self.assertEqual(row['budget'], 38)
        self.assertEqual(row['reviewed_reason'], 'out_of_stock')
        self.assertEqual(row['currency'], 'USD')
        self.assertEqual(self.client.post('/api/signals', json=payload).get_json()['id'], row['id'])
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'], 1)

    def test_changed_or_tampered_preview_is_rejected(self):
        draft = self.client.post('/api/signals/preview', json=dict(text='Need earbuds')).get_json()
        payload = {k:v for k,v in draft.items() if k not in ('fields','warnings')}
        payload['text'] = 'Different message'
        self.assertEqual(self.client.post('/api/signals',json=payload).status_code,400)
        payload['text'] = draft['text'];payload['preview_token'] += 'bad'
        self.assertEqual(self.client.post('/api/signals',json=payload).status_code,400)

    def test_mixed_currencies_are_never_added_together(self):
        self.capture(text='Need earbuds budget KES 6000',fields={'product':'Wireless Earbuds'})
        self.capture(text='Need earbuds budget USD 40',fields={'product':'Wireless Earbuds'})
        data = self.client.get('/api/insights').get_json()
        self.assertEqual(data['kpis']['stated_budgets'],{'KES':6000,'USD':40})
        self.assertIsNone(data['kpis']['stated_value'])
        self.assertIsNone(data['demand'][0]['value'])
        self.assertEqual(data['demand'][0]['customers'],0)
        self.assertEqual(data['demand'][0]['unidentified_requests'],2)

    def test_follow_up_and_recovery_are_separate_from_budget(self):
        row = self.capture(text='Need earbuds budget KES 6000')
        response = self.client.patch(f"/api/signals/{row['id']}",json=dict(reason='out_of_stock',status='recovered',owner='Alex',next_action='Collected after restock',recovered_amount=5500,recovered_currency='KES'))
        self.assertEqual(response.status_code,200,response.get_json())
        data = self.client.get('/api/insights').get_json()
        self.assertEqual(data['kpis']['recovered'],1)
        self.assertEqual(data['kpis']['recovered_value'],{'KES':5500})
        self.assertEqual(data['kpis']['stated_budgets'],{})
        self.assertEqual(data['demand'][0]['state'],'resolved')
        self.assertEqual(data['recent'][0]['budget'],6000)
        self.assertEqual(data['recent'][0]['owner'],'Alex')
        self.assertEqual(self.client.patch(f"/api/signals/{row['id']}",json=dict(status=[],recovered_amount=-10)).status_code,400)

    def test_aliases_unify_wording_and_keep_variants_reviewable(self):
        result = self.client.post('/api/catalog',json=dict(name='Wireless Earbuds',sku='AUD-1',aliases=['bluetooth earbuds','wireless earphones']))
        self.assertEqual(result.status_code,201)
        a = self.capture(text='Need bluetooth earbuds',customer='Shopper A')
        b = self.capture(text='Need wireless earphones',customer='Shopper A')
        self.assertEqual(a['product'],b['product'])
        group = self.client.get('/api/insights').get_json()['demand'][0]
        self.assertEqual(group['count'],2)
        self.assertEqual(group['customers'],1)
        # Availability questions do not establish that the retailer lacks stock.
        self.assertEqual(group['suggested_action'],'Ask what blocked the request')
        self.client.post('/api/catalog',json=dict(name='Running Shoes',aliases=['trainers']))
        preview = self.client.post('/api/signals/preview',json=dict(text='Need trainers size 42')).get_json()
        self.assertTrue(any('variant' in w or 'specification' in w for w in preview['warnings']))
        self.assertIn('42',preview['fields']['product'])
        collision = self.client.post('/api/catalog',json=dict(name='Other earbuds',aliases=['wireless earphones']))
        self.assertEqual(collision.status_code,400)
        self.assertEqual(len(self.client.get('/api/catalog').get_json()['items']),2)

    def test_retry_deduplication_is_scoped_to_channel(self):
        a = self.capture(external_id='event-1')
        b = self.capture(external_id='event-1')
        c = self.capture(external_id='event-1',source='chatbot')
        self.assertEqual(a['id'],b['id'])
        self.assertTrue(b['deduplicated'])
        self.assertNotEqual(a['id'],c['id'])
        for _ in range(2):
            self.client.post('/webhook/whatsapp',data=dict(Body='Need a backpack',From='Shopper B',MessageSid='wa-1'))
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'],3)

    def test_failed_search_and_stockout_events(self):
        result = self.client.post('/api/events/website-search',json=dict(query='earbuds',results_count=5))
        self.assertFalse(result.get_json()['captured'])
        self.assertEqual(self.client.post('/api/events/website-search',json=dict(query='earbuds')).status_code,400)
        result = self.client.post('/api/events/website-search',json=dict(query='earbuds',results_count=0,event_id='s-1'))
        self.assertTrue(result.get_json()['captured'])
        result = self.client.post('/api/events/stockout',json=dict(product='USB-C Fast Charger',qty=3,event_id='st-1'))
        self.assertEqual(result.status_code,201,result.get_json())
        self.assertEqual(result.get_json()['reason'],'out_of_stock')
        self.assertEqual(result.get_json()['product'],'USB-C Fast Charger')
        self.assertEqual(result.get_json()['reason_engine'],'channel_event')

    def test_csv_preview_reports_rows_and_reimports_are_idempotent(self):
        content=b'text,source,customer,currency\n"Need earbuds, please",whatsapp,Shopper A,KES\n"Need a backpack",chatbot,,USD\n,chatbot,,KES\n'
        response = self.client.post('/api/import/preview',data={'file':(io.BytesIO(content),'requests.csv')})
        self.assertEqual(response.status_code,200,response.get_json())
        data=response.get_json()
        self.assertEqual(len(data['drafts']),2)
        self.assertEqual(data['errors'][0]['row'],4)
        self.assertEqual(data['total'],3)
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'],0)
        items=[{k:v for k,v in d.items() if k not in ('fields','warnings')} for d in data['drafts']]
        first=self.client.post('/api/signals/bulk',json={'items':items})
        second=self.client.post('/api/signals/bulk',json={'items':items})
        self.assertEqual(first.status_code,201,first.get_json())
        self.assertTrue(all(r['deduplicated'] for r in second.get_json()))
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'],2)

    def test_bulk_validation_rolls_back_prior_rows(self):
        response=self.client.post('/api/signals/bulk',json={'items':[{'text':'Need earbuds'},{'text':'Need backpack','fields':{'qty':-5}}]})
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'],0)

    def test_evidence_filters_and_pagination(self):
        rows=[self.capture(text=f'Need wireless earbuds request {i}',customer=f'Shopper {i}',fields={'product':'Wireless Earbuds'}) for i in range(17)]
        page=self.client.get('/api/signals?page=2').get_json()
        self.assertEqual(len(page['items']),2)
        self.assertEqual(page['total'],17)
        group=self.client.get('/api/insights').get_json()['demand'][0]
        evidence=self.client.get('/api/opportunities/evidence',query_string={'key':group['key']}).get_json()
        self.assertEqual(len(evidence['items']),17)
        self.assertEqual(evidence['summary']['customers'],17)
        self.assertEqual(self.client.get('/api/signals?source=chatbot').get_json()['total'],0)
        with ghost.app.app_context():
            ghost.db().execute('UPDATE signals SET ts=?',((datetime.now(timezone.utc)-timedelta(days=10)).isoformat(),))
            ghost.db().commit()
        self.assertEqual(self.client.get('/api/insights?days=7').get_json()['kpis']['signals'],0)
        self.assertEqual(self.client.get('/api/insights?days=30').get_json()['kpis']['signals'],17)
        self.assertEqual(self.client.get('/api/insights?days=banana').status_code,400)

    def test_demo_records_and_model_are_isolated_from_live(self):
        self.capture(text='Need a real product')
        model_before=self.model_path.read_bytes()
        self.assertEqual(self.client.post('/api/demo/seed').status_code,400)
        first=self.client.post('/api/demo/seed',headers=self.demo)
        self.assertEqual(first.status_code,200,first.get_json())
        self.assertEqual(first.get_json()['added'],24)
        self.assertEqual(self.client.get('/api/insights',headers=self.demo).get_json()['kpis']['signals'],24)
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'],1)
        self.assertEqual(self.client.post('/api/demo/seed',headers=self.demo).get_json()['added'],0)
        self.assertEqual(self.client.post('/api/model/train',headers=self.demo).status_code,200)
        self.assertEqual(model_before,self.model_path.read_bytes())
        self.assertEqual(self.client.get('/demo').status_code,200)

    def test_csv_export_escapes_formula_cells(self):
        self.capture(text='=HYPERLINK("https://example.invalid")',customer='@formula',fields={'product':'=bad'})
        response=self.client.get('/api/signals/export')
        self.assertEqual(response.status_code,200)
        exported=list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(exported[0]['product'],"'=bad")
        self.assertEqual(exported[0]['customer'],"'@formula")

    def test_follow_up_queue_tracks_dates_owners_and_outcomes(self):
        today=datetime.fromisoformat(self.client.get('/api/operations/follow-ups').get_json()['today']).date()
        yesterday=(today-timedelta(days=1)).isoformat()
        overdue=self.capture(fields={'product':'Earbuds','urgency':3})
        due=self.capture(fields={'product':'Backpack','urgency':2})
        recovered=self.capture(fields={'product':'Charger'})
        closed=self.capture(fields={'product':'Bottle'})
        for row,change in [
            (overdue,dict(follow_up_on=yesterday,next_action='Confirm stock')),
            (due,dict(status='follow_up',owner='Sam',follow_up_on=today.isoformat())),
            (recovered,dict(status='recovered',owner='Alex',follow_up_on=yesterday,recovered_amount=50,recovered_currency='USD')),
            (closed,dict(status='closed',follow_up_on=yesterday))]:
            response=self.client.patch(f"/api/signals/{row['id']}",json=change)
            self.assertEqual(response.status_code,200,response.get_json())
        data=self.client.get('/api/operations/follow-ups').get_json()
        self.assertEqual(data['summary']['active'],2)
        self.assertEqual(data['summary']['overdue'],1)
        self.assertEqual(data['summary']['unassigned'],1)
        self.assertEqual(data['summary']['following_up'],1)
        self.assertEqual(data['summary']['recovered_value'],{'USD':50})
        self.assertEqual(data['summary']['needs_review'],4)
        self.assertEqual(data['items'][0]['id'],overdue['id'])
        self.assertTrue(data['items'][0]['overdue'])
        self.assertTrue(data['items'][1]['due_today'])
        self.assertIsNone(data['items'][0]['reviewed_reason'])
        self.assertEqual(data['items'][0]['model_version'],overdue['model_version'])
        by_owner=self.client.get('/api/operations/follow-ups?owner=sam').get_json()
        self.assertEqual([r['id'] for r in by_owner['items']],[due['id']])
        by_date=self.client.get('/api/operations/follow-ups?due=overdue').get_json()
        self.assertEqual([r['id'] for r in by_date['items']],[overdue['id']])
        won=self.client.get('/api/operations/follow-ups?status=recovered').get_json()
        self.assertEqual([r['id'] for r in won['items']],[recovered['id']])
        self.assertFalse(won['items'][0]['overdue'])

    def test_follow_up_dates_are_validated_and_can_be_cleared(self):
        row=self.capture()
        url=f"/api/signals/{row['id']}"
        for value in ('2026-02-30','tomorrow','20261027',[],42):
            self.assertEqual(self.client.patch(url,json={'follow_up_on':value}).status_code,400)
        response=self.client.patch(url,json=dict(follow_up_on='2026-10-27',owner='Sam'))
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()['follow_up_on'],'2026-10-27')
        self.assertEqual(self.client.patch(url,json={'follow_up_on':None}).get_json()['follow_up_on'],None)
        for query in ('status=invalid','assignment=invalid','urgency=invalid','due=invalid','page=invalid'):
            self.assertEqual(self.client.get('/api/operations/follow-ups?'+query).status_code,400)

    def test_follow_up_queue_filters_and_pagination(self):
        for i in range(14):
            self.capture(text=f'Need earbuds {i}',fields={'product':'Earbuds','urgency':3 if i==0 else 1})
        page=self.client.get('/api/operations/follow-ups?page=2').get_json()
        self.assertEqual(page['total'],14)
        self.assertEqual(len(page['items']),2)
        urgent=self.client.get('/api/operations/follow-ups?urgency=urgent').get_json()
        self.assertEqual(urgent['total'],1)
        self.assertEqual(self.client.get('/api/operations/follow-ups?assignment=unassigned').get_json()['total'],14)
        self.assertEqual(self.client.get('/api/operations/follow-ups?assignment=assigned').get_json()['total'],0)
        self.assertEqual(self.client.get('/api/operations/follow-ups?q=missing').get_json()['total'],0)
        last=self.client.get('/api/operations/follow-ups?page=99').get_json()
        self.assertEqual(last['page'],2)


if __name__=='__main__':
    unittest.main()
