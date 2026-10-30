import hashlib
import hmac
import json
import os
import shutil
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import app as ghost
from demand_ml import ReasonClassifier
from live_channels import email_event, gmail_config
from workspace_auth import safe_next


RAW_MAIL = b'From: Demo Customer <customer@example.test>\r\nSubject: [GhostDemand Demo] Earbuds enquiry\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nNeed 2 wireless earbuds but they are sold out. Budget KES 6,000.'


class FakeMailbox:
    def __init__(self, *, next_uid=25, found=b'25', raw=RAW_MAIL, oversized=False):
        self.next_uid, self.found, self.raw, self.oversized = next_uid, found, raw, oversized
        self.calls = []

    def login(self, account, password):
        self.calls.append(('login', account))
        return 'OK', []

    def select(self, folder, readonly=False):
        self.calls.append(('select', folder, readonly))
        return 'OK', [b'1']

    def response(self, key):
        return key, [b'3' if key == 'UIDVALIDITY' else str(self.next_uid).encode()]

    def uid(self, command, *args):
        self.calls.append((command, *args))
        if command == 'search': return 'OK', [self.found]
        if 'RFC822.SIZE' in args[-1]:
            size = 300000 if self.oversized else len(self.raw)
            header = self.raw.split(b'\r\n\r\n', 1)[0] + b'\r\n\r\n'
            return 'OK', [(f'25 (RFC822.SIZE {size}'.encode(), header)]
        return 'OK', [(b'25 (BODY[]', self.raw)]

    def logout(self): self.calls.append(('logout',))


class LiveChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.trained = Path(cls.shared.name) / 'model.json'
        ReasonClassifier(cls.trained).train([])

    @classmethod
    def tearDownClass(cls): cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        path = self.root / 'model.json'; shutil.copy(self.trained, path)
        self.environment = dict(GHOST_SECRET_KEY='s' * 64, GHOST_ADMIN_PASSWORD='owner-test-password',
            WHATSAPP_APP_SECRET='test-meta-secret', WHATSAPP_PHONE_NUMBER_ID='123456789',
            WHATSAPP_VERIFY_TOKEN='test-verification-token', GHOST_PUBLIC_URL='https://demo.example.test',
            GHOST_GMAIL_ACCOUNT='presentation@example.test', GHOST_GMAIL_APP_PASSWORD='abcdefghijklmnop',
            GHOST_GMAIL_FOLDER='GhostDemandDemo', GHOST_GMAIL_ENABLED='1')
        self.patches = [patch.dict(os.environ, self.environment),
            patch.object(ghost, 'DB', str(self.root / 'live.db')),
            patch.object(ghost, 'DEMO_DB', str(self.root / 'demo.db')),
            patch.object(ghost, 'ML', ReasonClassifier(path)), patch.object(ghost, 'DEMO_ML', ReasonClassifier(path)),
            patch.object(ghost, 'llm_extract', side_effect=AssertionError('Private channel data must stay local'))]
        for item in self.patches: item.start()
        ghost.app.config['TESTING'] = True
        self.client = ghost.app.test_client()

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def login(self):
        self.client.get('/login')
        with self.client.session_transaction() as session: csrf = session['csrf']
        result = self.client.post('/login', data={'csrf':csrf, 'password':'owner-test-password', 'next':'/dashboard#activity'})
        self.assertEqual(result.status_code, 302)
        with self.client.session_transaction() as session: return {'X-Ghost-CSRF':session['csrf']}

    def meta(self, *, message_id='wamid.example-1', phone='123456789', message_type='text'):
        return {'object':'whatsapp_business_account','entry':[{'changes':[{'value':{
            'metadata':{'phone_number_id':phone}, 'messages':[{'id':message_id,'from':'254700123456',
            'type':message_type,'text':{'body':'Need 2 wireless earbuds but they are sold out. Budget KES 6,000.'}}]}}]}]}

    def send_meta(self, payload=None, *, signature=True, headers=None):
        data = json.dumps(payload or self.meta()).encode()
        actual_headers = {'Content-Type':'application/json', **(headers or {})}
        if signature:
            actual_headers['X-Hub-Signature-256'] = 'sha256=' + hmac.new(b'test-meta-secret', data, hashlib.sha256).hexdigest()
        return self.client.post('/webhooks/whatsapp', data=data, headers=actual_headers)

    def sync(self, mailbox):
        with patch('live_channels.imaplib.IMAP4_SSL', return_value=mailbox) as connect:
            with ghost.app.app_context(): result = ghost.app.extensions['ghost_gmail_sync']()
        return result, connect

    def test_live_data_needs_login_while_demo_stays_separate(self):
        self.assertEqual(self.client.get('/dashboard').status_code, 302)
        self.assertEqual(self.client.get('/api/intake').status_code, 401)
        self.assertEqual(self.client.get('/api/signals').status_code, 401)
        self.assertEqual(self.client.get('/api/connections').status_code, 401)
        self.assertEqual(self.client.get('/demo').status_code, 200)
        demo = self.client.get('/api/connections', headers={'X-Ghost-Demo':'1'}).get_json()
        self.assertEqual(demo['providers'], {})
        self.assertFalse((self.root / 'live.db').exists())

    def test_login_checks_csrf_and_prevents_external_redirects(self):
        self.assertEqual(self.client.post('/login',data={'password':'owner-test-password'}).status_code,400)
        headers = self.login()
        self.assertEqual(self.client.get('/api/connections').status_code, 200)
        self.assertEqual(self.client.post('/api/connections/gmail/check',json={}).status_code, 403)
        self.assertEqual(self.client.post('/logout',data={'csrf':headers['X-Ghost-CSRF']}).status_code, 302)
        self.assertEqual(self.client.get('/api/intake').status_code, 401)
        for value in ('https://evil.example', '//evil.example', '/\\evil.example', '/\nmalformed'):
            self.assertEqual(safe_next(value), '/dashboard#activity')

    def test_cookie_policy_supports_loopback_without_downgrading_cloud_sessions(self):
        with patch.dict(ghost.app.config, {'SESSION_COOKIE_SECURE':True}):
            local=ghost.app.test_client().get('/login',base_url='http://127.0.0.1')
            self.assertNotIn('; Secure',local.headers['Set-Cookie'])
            cloud=ghost.app.test_client().get('/login',base_url='https://demo.example.test')
            self.assertIn('; Secure',cloud.headers['Set-Cookie'])

    def test_meta_verification_challenge_and_signatures(self):
        query={'hub.mode':'subscribe','hub.verify_token':'bad','hub.challenge':'12345'}
        self.assertEqual(self.client.get('/webhooks/whatsapp',query_string=query).status_code,403)
        self.assertEqual(self.send_meta(signature=False).status_code,401)
        self.assertFalse((self.root / 'live.db').exists())
        query['hub.verify_token']='test-verification-token'
        verified=self.client.get('/webhooks/whatsapp',query_string=query)
        self.assertEqual(verified.get_data(as_text=True),'12345')
        self.assertEqual(verified.mimetype,'text/plain')
        self.assertEqual(self.send_meta(headers={'X-Ghost-Demo':'1'}).status_code,400)

    def test_signed_meta_message_is_analysed_once_and_contact_is_minimized(self):
        for _ in range(2):
            response=self.send_meta()
            self.assertEqual(response.status_code,200,response.get_json())
            self.assertEqual(response.get_json()['accepted'],1)
        self.login()
        activity=self.client.get('/api/intake').get_json()
        self.assertEqual(activity['summary']['captured'],1)
        self.assertNotIn('254700123456', json.dumps(activity))
        signal=self.client.get('/api/signals/'+str(activity['items'][0]['signal_id'])).get_json()
        self.assertIsNone(signal['reviewed_reason'])
        self.assertEqual(signal['source'],'whatsapp')
        ghost.llm_extract.assert_not_called()

    def test_wrong_number_delivery_status_and_media_do_not_create_requests(self):
        self.assertEqual(self.send_meta(self.meta(phone='another-number')).get_json()['accepted'],0)
        self.assertEqual(self.send_meta(self.meta(message_type='audio')).get_json()['ignored'],1)
        status={'object':'whatsapp_business_account','entry':[{'changes':[{'value':{'metadata':{'phone_number_id':'123456789'},'statuses':[{'status':'delivered'}]}}]}]}
        self.assertEqual(self.send_meta(status).get_json()['accepted'],0)
        self.login()
        self.assertEqual(self.client.get('/api/insights').get_json()['kpis']['signals'],0)

    def test_providers_cannot_receive_until_live_workspace_is_private(self):
        with patch.dict(os.environ, {'GHOST_SECRET_KEY':''}):
            self.assertEqual(self.send_meta().status_code,503)
            result,_=self.sync(FakeMailbox())
            self.assertEqual(result['state'],'setup_needed')

    def test_gmail_baseline_excludes_old_mail_and_uses_tls_readonly(self):
        mailbox=FakeMailbox(next_uid=26)
        result,connect=self.sync(mailbox)
        self.assertTrue(result['baseline'])
        self.assertEqual(result['imported'],0)
        self.assertFalse(any(call[0]=='fetch' for call in mailbox.calls))
        self.assertIn(('select','"GhostDemandDemo"',True),mailbox.calls)
        self.assertEqual(connect.call_args.kwargs['ssl_context'].verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(connect.call_args.args, ('imap.gmail.com',993))

    def test_gmail_new_mail_is_analysed_and_not_marked_read(self):
        self.sync(FakeMailbox(next_uid=25))
        mailbox=FakeMailbox(next_uid=26)
        result,_=self.sync(mailbox)
        self.assertEqual(result['imported'],1)
        self.assertTrue(all('BODY.PEEK' in call[-1] for call in mailbox.calls if call[0]=='fetch'))
        repeat=FakeMailbox(next_uid=26)
        again,_=self.sync(repeat)
        self.assertEqual(again['imported'],0)
        self.login()
        data=self.client.get('/api/intake').get_json()
        self.assertEqual(data['summary']['captured'],1)
        self.assertNotIn('customer@example.test',json.dumps(data))
        ghost.llm_extract.assert_not_called()

    def test_gmail_analysis_failure_keeps_cursor_for_retry(self):
        self.sync(FakeMailbox(next_uid=25))
        with patch.object(ghost,'draft_signal',side_effect=RuntimeError('private failure')):
            result,_=self.sync(FakeMailbox(next_uid=26))
        self.assertEqual(result['state'],'attention')
        self.assertNotIn('private failure',result['message'])
        retried,_=self.sync(FakeMailbox(next_uid=26))
        self.assertEqual(retried['imported'],1)
        self.login()
        self.assertEqual(self.client.get('/api/intake').get_json()['summary']['total'],1)

    def test_gmail_lease_prevents_parallel_listeners(self):
        self.sync(FakeMailbox(next_uid=25))
        with ghost.app.app_context():
            ghost.db().execute("UPDATE live_channel_state SET lease_owner='other',lease_until=? WHERE name='gmail'",(time.time()+100,))
            ghost.db().commit()
        result,connect=self.sync(FakeMailbox(next_uid=26))
        self.assertEqual(result['state'],'busy')
        connect.assert_not_called()

    def test_oversized_email_is_held_without_fetching_body(self):
        self.sync(FakeMailbox(next_uid=25))
        mailbox=FakeMailbox(next_uid=26,oversized=True)
        result,_=self.sync(mailbox)
        self.assertEqual(result['imported'],1)
        self.assertFalse(any(call[0]=='fetch' and call[-1]=='(BODY.PEEK[])' for call in mailbox.calls))
        self.login()
        data=self.client.get('/api/intake').get_json()
        self.assertEqual(data['summary']['needs_context'],1)

    def test_gmail_cannot_select_inbox_and_status_never_exposes_credentials(self):
        with patch.dict(os.environ,{'GHOST_GMAIL_FOLDER':'INBOX'}): self.assertIsNone(gmail_config())
        self.login()
        text=self.client.get('/api/connections').get_data(as_text=True)
        for secret in ('test-meta-secret','test-verification-token','abcdefghijklmnop','presentation@example.test'):
            self.assertNotIn(secret,text)

    def test_html_email_and_bad_charset_are_safe_to_process(self):
        raw=b'From: a@example.test\r\nSubject: Enquiry\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>Need earbuds, sold out.</p><script>do not import this script</script>'
        event=email_event(raw,'html-1')
        self.assertIn('Need earbuds',event['text'])
        self.assertNotIn('import this script',event['text'])
        malformed=b'Content-Type: text/plain; charset=made-up-charset\r\n\r\ntext'
        self.assertEqual(email_event(malformed,'bad-charset')['outcome'],'pending')


if __name__ == '__main__': unittest.main()
