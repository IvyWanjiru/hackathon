import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import app as ghost
from pesashield import analyse_message, expected_amount


RECEIPT = ('UJ01ABC123 Confirmed. You have received Ksh5,000.00 from DEMO CUSTOMER 0712***456 '
           'on 30/10/26 at 9:00 AM. New M-PESA balance is Ksh6,250.00. '
           'Transaction cost, Ksh0.00. Never share your PIN.')
PAYBILL = ('UJ01ABC124 Confirmed. Ksh1,200.00 paid to DEMO PAYBILL 400200 on 30/10/26 '
           'at 9:00 AM. New M-PESA balance is Ksh0.00.')


class MessageAssessmentTests(unittest.TestCase):
    def test_complete_receipt_extracts_fields_without_false_pin_warning(self):
        result = analyse_message(RECEIPT)
        self.assertEqual(result['verdict'], 'legit')
        self.assertEqual(result['reason_codes'], [])
        self.assertEqual(result['receipt']['reference'], 'UJ01ABC123')
        self.assertEqual(result['receipt']['amount'], 5000)
        self.assertEqual(result['receipt']['balance'], 6250)
        self.assertIn('DEMO CUSTOMER', result['receipt']['counterparty'])
        self.assertTrue(result['receipt']['format_complete'])
        self.assertEqual(result['verification']['status'], 'not_verified')
        self.assertIsNone(result['confidence'])
        self.assertIsNone(result['risk_score'])

    def test_paybill_and_zero_balance_are_not_fraud_indicators(self):
        result = analyse_message(PAYBILL)
        self.assertEqual(result['verdict'], 'legit')
        self.assertEqual(result['receipt']['transaction_type'], 'Merchant payment')
        self.assertEqual(result['receipt']['balance'], 0)
        self.assertNotIn('payment_redirection', result['reason_codes'])

    def test_sent_money_receipt(self):
        text = PAYBILL.replace('paid to DEMO PAYBILL 400200', 'sent to DEMO PERSON 0712***456')
        self.assertEqual(analyse_message(text)['verdict'], 'legit')

    def test_pin_and_otp_requests_override_receipt_format(self):
        for request in ('Send your M-PESA PIN to support.', 'Please share your OTP.', 'Tuma PIN yako.',
                        'Send your PI\u200bN to claim a reversal.'):
            with self.subTest(request=request):
                result = analyse_message(RECEIPT + ' ' + request)
                self.assertEqual(result['verdict'], 'scam')
                self.assertIn('pin_request', result['reason_codes'])

    def test_negated_safety_advice_is_not_a_request_for_secrets_or_money(self):
        for text in ("Never share your M-PESA PIN or OTP.", "Don't send your PIN.",
                     'Do not send money back for a reversal.', 'Never pay a fee to claim a prize.',
                     'Someone asking for your M-PESA PIN or OTP code is a scam.'):
            with self.subTest(text=text):
                result = analyse_message(text)
                self.assertEqual(result['verdict'], 'unknown')
                self.assertEqual(result['reason_codes'], [])

    def test_warning_cannot_hide_a_second_affirmative_request(self):
        for text in ('Never share your PIN, but send your OTP to me.',
                     'Do not ignore this message, send your PIN now.',
                     'Never send your PIN, send your OTP instead.',
                     'Do not send your PIN, send money to reverse the mistaken payment.'):
            with self.subTest(text=text):
                self.assertEqual(analyse_message(text)['verdict'], 'scam')

    def test_official_menu_instruction_is_not_an_sms_pin_disclosure(self):
        result = analyse_message('Open your M-PESA menu and enter your PIN on your phone to complete your payment.')
        self.assertNotIn('pin_request', result['reason_codes'])
        self.assertEqual(result['verdict'], 'unknown')

    def test_fake_refund_instruction(self):
        result = analyse_message('I sent you money by mistake. Please send Ksh500 back to this number to reverse it.')
        self.assertEqual(result['verdict'], 'scam')
        self.assertIn('refund_request', result['reason_codes'])

    def test_locked_balance(self):
        result = analyse_message(RECEIPT.replace('Ksh6,250.00', 'LOCKED'))
        self.assertEqual(result['verdict'], 'scam')
        self.assertIn('locked_balance', result['reason_codes'])

    def test_prize_fee_and_loan_upfront_fee(self):
        for text, code in (('Congratulations, you won a prize. Send Ksh200 to claim it.', 'prize_fee'),
                           ('Your loan is approved. Pay Ksh100 processing fee for release.', 'upfront_fee')):
            result = analyse_message(text)
            self.assertEqual(result['verdict'], 'scam')
            self.assertIn(code, result['reason_codes'])

    def test_account_threat_and_changed_till_need_verification(self):
        for text in ('Your M-PESA account will be blocked. Call this number to verify.',
                     'Our till has changed, use this different till 123456 instead.'):
            self.assertEqual(analyse_message(text)['verdict'], 'suspicious')

    def test_links_and_lookalike_domains(self):
        self.assertEqual(analyse_message('Enter your PIN at https://mpesa-claim.example.test')['verdict'], 'scam')
        result = analyse_message('Check https://safaricom.co.ke.example.test/verify')
        self.assertIn('external_link', result['reason_codes'])
        self.assertEqual(analyse_message('Check https://[broken')['verdict'], 'suspicious')
        self.assertNotIn('external_link', analyse_message('Read https://www.safaricom.co.ke/help')['reason_codes'])

    def test_expected_amount_mismatch(self):
        self.assertEqual(analyse_message(RECEIPT, expected=5000)['verdict'], 'legit')
        mismatch = analyse_message(RECEIPT, expected=4500)
        self.assertEqual(mismatch['verdict'], 'suspicious')
        self.assertIn('amount_mismatch', mismatch['reason_codes'])

    def test_incomplete_invalid_or_unrelated_text_does_not_get_green_verdict(self):
        for text in ('Hello, how are you?', 'M-PESA confirmed Ksh5000',
                     RECEIPT.replace('Ksh5,000.00', 'Ksh-5,000.00'),
                     RECEIPT.replace('Ksh5,000.00', 'Ksh5,00.00')):
            with self.subTest(text=text):
                self.assertEqual(analyse_message(text, prediction={'verdict':'legit','confidence':.99})['verdict'], 'unknown')

    def test_invalid_date_and_multiple_receipts(self):
        invalid = analyse_message(RECEIPT.replace('30/10/26', '31/02/26'))
        self.assertEqual(invalid['verdict'], 'suspicious')
        self.assertIn('invalid_date', invalid['reason_codes'])
        self.assertIn('multiple_receipts', analyse_message(RECEIPT+' '+PAYBILL)['reason_codes'])

    def test_unvalidated_model_is_not_authority_for_payment_authenticity(self):
        self.assertEqual(analyse_message('Hello',prediction={'verdict':'scam','confidence':.999})['verdict'], 'unknown')
        self.assertEqual(analyse_message(RECEIPT,prediction={'verdict':'scam','confidence':.999})['verdict'], 'legit')
        result=analyse_message('Your account is blocked. Call to verify.',prediction={'verdict':'scam','confidence':.9})
        self.assertEqual(result['verdict'], 'suspicious')
        self.assertIn('model_caution',result['reason_codes'])

    def test_comparison_amount_validation(self):
        self.assertEqual(expected_amount('5,000.50'),5000.5)
        self.assertIsNone(expected_amount(None))
        for value in (True, -5, 0, 'NaN', 'Infinity', [], {}, '1.001'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): expected_amount(value)


class PesaShieldApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.live = str(Path(self.temp.name)/'live.db')
        self.demo = str(Path(self.temp.name)/'demo.db')
        self.environment=dict(GHOST_SECRET_KEY='s'*64,GHOST_ADMIN_PASSWORD='test-owner-password')
        self.model=Mock(); self.model.predict.return_value=None
        self.patches=[patch.dict(os.environ,self.environment),patch.object(ghost,'DB',self.live),
                      patch.object(ghost,'DEMO_DB',self.demo),patch.object(ghost,'PESASHIELD_ENABLED',True),
                      patch.object(ghost,'active_fraud_model',return_value=self.model),
                      patch.object(ghost,'llm_extract',side_effect=AssertionError('SMS must stay local'))]
        for item in self.patches:item.start()
        ghost.app.config['TESTING']=True
        self.client=ghost.app.test_client()
        with self.client.session_transaction() as s:s.update(workspace_authenticated=True,csrf='test-csrf')
        self.headers={'X-Ghost-CSRF':'test-csrf'}

    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()

    def post(self, payload, **kwargs):
        return self.client.post('/api/pesashield/sms/check',json=payload,headers=self.headers,**kwargs)

    def test_api_returns_explanation_and_receipt_and_saves_provenance(self):
        response=self.post({'text':RECEIPT,'expected_amount':'5000'})
        self.assertEqual(response.status_code,200,response.get_json())
        result=response.get_json()
        self.assertEqual(result['verdict'],'legit')
        self.assertTrue(result['saved'])
        self.assertIsNone(result['risk_score'])
        with closing(sqlite3.connect(self.live)) as conn:
            self.assertEqual(conn.execute('SELECT verdict FROM fraud_messages').fetchone()[0],'legit')
        self.assertEqual(self.client.get('/api/pesashield/insights').get_json()['kpis']['legit'],1)
        report=self.client.get('/api/pesashield/reports').get_json()['items'][0]
        self.assertEqual(report['assessment_label'],'Looks legitimate')
        ghost.llm_extract.assert_not_called()

    def test_retry_without_sender_counts_once(self):
        a=self.post({'text':RECEIPT}).get_json();b=self.post({'text':RECEIPT}).get_json()
        self.assertEqual(a['id'],b['id'])
        self.assertTrue(b['deduplicated'])
        self.assertEqual(a['receipt'],b['receipt'])
        self.assertEqual(self.client.get('/api/pesashield/insights').get_json()['kpis']['total'],1)

    def test_public_demo_analysis_is_not_saved_or_visible_in_live_history(self):
        client=ghost.app.test_client()
        response=client.post('/api/pesashield/sms/check',json={'text':RECEIPT},headers={'X-Ghost-Demo':'1'})
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.get_json()['saved'])
        self.assertIsNone(response.get_json()['id'])
        self.assertFalse(Path(self.demo).exists())
        self.assertEqual(client.get('/api/pesashield/reports',headers={'X-Ghost-Demo':'1'}).get_json()['items'],[])
        self.assertEqual(client.get('/api/pesashield/reports').status_code,401)

    def test_unprotected_workspace_is_stateless(self):
        with patch.dict(os.environ,{'GHOST_ADMIN_PASSWORD':'','GHOST_ADMIN_PASSWORD_HASH':'','WHATSAPP_APP_SECRET':'','WHATSAPP_PHONE_NUMBER_ID':'','GHOST_GMAIL_ACCOUNT':'','GHOST_GMAIL_APP_PASSWORD':''}):
            response=ghost.app.test_client().post('/api/pesashield/sms/check',json={'text':RECEIPT})
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.get_json()['saved'])
        self.assertFalse(Path(self.live).exists())

    def test_input_validation_and_csrf(self):
        for payload in ([], {'text':[]}, {'text':{}}, {'text':' '}, {'text':'x'*10001},
                        {'text':RECEIPT,'sender':{}}, {'text':RECEIPT,'expected_amount':'NaN'},
                        {'text':RECEIPT,'channel':'invalid'}):
            with self.subTest(payload_type=type(payload).__name__):
                self.assertEqual(self.post(payload).status_code,400)
        self.assertEqual(self.client.post('/api/pesashield/sms/check',json={'text':RECEIPT}).status_code,403)

    def test_review_is_explicit_and_private(self):
        rid=self.post({'text':RECEIPT}).get_json()['id']
        response=self.client.patch(f'/api/pesashield/reports/{rid}/review',json={'verdict':'suspicious'},headers=self.headers)
        self.assertEqual(response.status_code,200)
        report=self.client.get('/api/pesashield/reports').get_json()['items'][0]
        self.assertEqual(report['reviewed_verdict'],'suspicious')
        self.assertEqual(report['verdict'],'legit')
        self.assertEqual(ghost.app.test_client().patch(f'/api/pesashield/reports/{rid}/review',json={'verdict':'legit'},headers={'X-Ghost-Demo':'1'}).status_code,403)

    def test_concurrent_retries_without_sender_are_idempotent(self):
        def check(_):
            client=ghost.app.test_client()
            with client.session_transaction() as s:s.update(workspace_authenticated=True,csrf='test-csrf')
            response=client.post('/api/pesashield/sms/check',json={'text':RECEIPT},headers=self.headers)
            self.assertEqual(response.status_code,200)
            return response.get_json()['id']
        with ThreadPoolExecutor(max_workers=4) as pool:ids=list(pool.map(check,range(4)))
        self.assertEqual(len(set(ids)),1)

    def test_existing_history_schema_is_migrated_without_losing_old_checks(self):
        self.post({'text':RECEIPT})
        with closing(sqlite3.connect(self.live)) as conn:
            conn.execute('ALTER TABLE fraud_messages DROP COLUMN analysis_json')
            conn.commit()
        response=self.post({'text':'Please send your M-PESA PIN to me.'})
        self.assertEqual(response.status_code,200)
        reports=self.client.get('/api/pesashield/reports').get_json()['items']
        self.assertEqual(len(reports),2)
        self.assertEqual(reports[-1]['assessment_label'],'Older assessment - recheck')


if __name__=='__main__':unittest.main()
