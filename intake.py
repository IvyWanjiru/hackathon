"""Authenticated automatic intake. Classification and eligibility stay server-side."""
import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone

from flask import g, jsonify, request

from intelligence import CURRENCIES, SOURCE_NAMES

CHANNELS = {
    'whatsapp': 'whatsapp', 'email': 'email', 'chatbot': 'chatbot',
    'sales_notes': 'salesperson', 'voice_notes': 'salesperson_voice',
    'website_search': 'website_search', 'stockout': 'out_of_stock',
    'rejected_quote': 'rejected_quote', 'abandoned_enquiry': 'abandoned_enquiry',
}
CHANNEL_HELP = {
    'whatsapp': 'Receive incoming text enquiries from your WhatsApp Business account.',
    'email': 'Receive selected enquiries from your dedicated business email label or relay.',
    'chatbot': 'Send customer messages and their conversation outcome from your website chatbot.',
    'sales_notes': 'Forward saved notes from your sales system.',
    'voice_notes': 'Forward completed transcripts from your voice-note service.',
}
STATE_NAMES = {'received':'Received', 'captured':'Demand captured', 'needs_context':'Needs context',
               'ignored':'No missed demand', 'failed':'Needs retry'}
FAILURE = re.compile(
    r"\b(?:sold out|out of stock|no stock|not in stock|too expensive|too pricey|over budget|"
    r"price (?:is |was )?too high|declined|rejected|never replied|no response|went quiet|"
    r"abandoned (?:the |their )?(?:cart|enquiry|inquiry)|do not (?:carry|stock|sell)|"
    r"don't (?:carry|stock|sell)|does not (?:carry|stock|sell)|don't offer|do not offer|"
    r"delivery (?:is |was |takes )?too (?:long|slow)|delivery misses the deadline)\b", re.I)
NEGATED_FAILURE = re.compile(r"\b(?:not|no longer|isn't|aren't)\s+(?:sold out|out of stock|too expensive|over budget)\b", re.I)


def normalize_event(channel, payload, default_currency):
    if channel not in CHANNELS:
        raise ValueError('Choose a supported incoming channel.')
    if not isinstance(payload, dict):
        raise ValueError('Send one event as a JSON object.')
    if set(payload) & {'verified', 'fields', 'reason', 'reviewed_reason'}:
        raise ValueError('Incoming events cannot supply verified labels or classifier results.')
    event_id = payload.get('event_id')
    if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200:
        raise ValueError('Provide a stable event_id of up to 200 characters.')
    text = payload.get('transcript') if channel == 'voice_notes' else payload.get('text')
    product = payload.get('product')
    if product is not None and (not isinstance(product, str) or not product.strip() or len(product) > 250):
        raise ValueError('Product must be text of up to 250 characters.')
    qty = payload.get('qty')
    if qty is not None and (type(qty) is not int or not 1 <= qty <= 1_000_000_000):
        raise ValueError('Quantity must be a positive integer.')
    if channel == 'stockout':
        if not product:
            raise ValueError('A stockout event needs a product.')
        text = f'Out of stock: {qty or 1} units {product}'
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError('Provide 1–10,000 characters of transcript text.' if channel == 'voice_notes'
                         else 'Provide 1–10,000 characters of request text.')
    customer = payload.get('customer')
    if customer is not None and (not isinstance(customer, str) or len(customer) > 250):
        raise ValueError('Customer reference must be text of up to 250 characters.')
    outcome = payload.get('outcome')
    if outcome not in (None, 'unmet', 'pending', 'fulfilled', 'not_a_request'):
        raise ValueError('Outcome must be unmet, pending, fulfilled or not_a_request.')
    currency = payload.get('currency', default_currency)
    if currency not in CURRENCIES:
        raise ValueError('Choose a supported currency.')
    results = payload.get('results_count')
    if channel == 'website_search' and (type(results) is not int or results < 0):
        raise ValueError('A website search needs a non-negative integer results_count.')
    conversation = payload.get('conversation_id')
    if conversation is not None and (not isinstance(conversation, str) or len(conversation) > 200):
        raise ValueError('Conversation reference must be text of up to 200 characters.')
    return dict(channel=channel, source=CHANNELS[channel], event_id=event_id.strip(),
                text=text.strip(), customer=customer, currency=currency, outcome=outcome,
                product=product.strip() if product else None, qty=qty,
                conversation_id=conversation, results_count=results)


def eligibility(event):
    if event['outcome'] in ('fulfilled', 'not_a_request'):
        return 'ignored', 'The channel reported a fulfilled request or a non-request.'
    if event['outcome'] == 'pending':
        return 'needs_context', 'The conversation outcome is still pending.'
    if event['channel'] == 'website_search':
        return ('captured', 'A search returned no results; check catalog coverage.') if event['results_count'] == 0 else ('ignored', 'The search returned results.')
    if event['outcome'] == 'unmet' or event['channel'] in ('stockout', 'rejected_quote', 'abandoned_enquiry'):
        return 'captured', 'The channel reported a missed request.'
    if not NEGATED_FAILURE.search(event['text']) and FAILURE.search(event['text']):
        return 'captured', 'The message describes a purchase blocker; details remain available for review.'
    return 'needs_context', 'An enquiry alone does not confirm a missed sale.'


def register_intake(app, *, db, save_signal, is_demo, default_currency):
    def connection():
        conn = db()
        conn.execute('''CREATE TABLE IF NOT EXISTS intake_events(
            id INTEGER PRIMARY KEY, received_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            channel TEXT NOT NULL, event_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL, state TEXT NOT NULL, detail TEXT,
            signal_id INTEGER REFERENCES signals(id), attempts INTEGER NOT NULL DEFAULT 0,
            UNIQUE(channel,event_id))''')
        return conn

    def public_row(row):
        event = json.loads(row['payload_json'])
        return dict(id=row['id'], received_at=row['received_at'], updated_at=row['updated_at'],
                    channel=row['channel'], source=event['source'], source_name=SOURCE_NAMES[event['source']],
                    text=event['text'], customer=event['customer'], state=row['state'],
                    state_name=STATE_NAMES[row['state']], detail=row['detail'], signal_id=row['signal_id'])

    def result(conn, event_id, duplicate=False):
        row = conn.execute('SELECT * FROM intake_events WHERE id=?', (event_id,)).fetchone()
        return dict(event=public_row(row), captured=row['state'] == 'captured', deduplicated=duplicate)

    def analyse(conn, row, *, confirm=False):
        event = json.loads(row['payload_json'])
        state, detail = ('captured', 'A team member confirmed this was a missed request.') if confirm else eligibility(event)
        signal_id = None
        try:
            if state == 'captured':
                fields = {}
                if event['product']: fields['product'] = event['product']
                if event['qty'] is not None: fields['qty'] = event['qty']
                if event['channel'] == 'stockout': fields['reason'] = 'out_of_stock'
                external = 'incoming-' + hashlib.sha256(f"{event['channel']}:{event['event_id']}".encode()).hexdigest()
                previous = g.get('local_extraction_only', False)
                g.local_extraction_only = True
                try:
                    saved = save_signal(event['text'], event['source'], event['customer'],
                                        currency=event['currency'], external_id=external, fields=fields,
                                        verified=False, commit=False, event=event['channel'] == 'stockout')
                finally:
                    g.local_extraction_only = previous
                signal_id = saved['id']
            conn.execute('UPDATE intake_events SET state=?,detail=?,signal_id=?,attempts=attempts+1,updated_at=? WHERE id=?',
                         (state, detail, signal_id, datetime.now(timezone.utc).isoformat(), row['id']))
            conn.commit()
            return result(conn, row['id'])
        except Exception as exc:
            conn.rollback()
            current = conn.execute('SELECT * FROM intake_events WHERE id=?', (row['id'],)).fetchone()
            if current['state'] == 'captured':
                return result(conn, row['id'], True)
            conn.execute("UPDATE intake_events SET state='failed',detail=?,attempts=attempts+1,updated_at=? WHERE id=?",
                         ('Analysis could not complete. Retry this event.', datetime.now(timezone.utc).isoformat(), row['id']))
            conn.commit()
            app.logger.warning('Automatic intake failed (%s); event %s can be retried.', type(exc).__name__, row['id'])
            return dict(error='Analysis could not complete. Retry the same event_id.', **result(conn, row['id'])), 503

    def receive(channel, payload):
        event = normalize_event(channel, payload, default_currency)
        encoded = json.dumps(event, sort_keys=True, separators=(',', ':'))
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        conn = connection()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute('INSERT OR IGNORE INTO intake_events(received_at,updated_at,channel,event_id,payload_hash,payload_json,state) VALUES(?,?,?,?,?,?,?)',
                              (now, now, channel, event['event_id'], fingerprint, encoded, 'received'))
        added = cursor.rowcount == 1
        conn.commit()
        row = conn.execute('SELECT * FROM intake_events WHERE channel=? AND event_id=?', (channel,event['event_id'])).fetchone()
        if row['payload_hash'] != fingerprint:
            return dict(error='This event_id already belongs to different content. Use a new event_id.'), 409
        if not added and row['state'] not in ('received', 'failed'):
            return result(conn, row['id'], True)
        return analyse(conn, row)

    @app.post('/api/ingest/<channel>')
    def ingest_channel(channel):
        token = os.environ.get('GHOST_INGEST_TOKEN', '')
        if not token:
            return jsonify(error='Automatic intake needs server configuration.'), 503
        supplied = request.headers.get('Authorization', '')
        if not hmac.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
            return jsonify(error='A valid channel relay token is required.'), 401
        if is_demo():
            return jsonify(error='Live channel relays cannot write to the demo workspace.'), 400
        return receive(channel, request.get_json())

    @app.get('/api/intake')
    def intake_activity():
        conn = connection()
        counts = {state:0 for state in STATE_NAMES}
        counts.update({row['state']:row['n'] for row in conn.execute('SELECT state,count(*) AS n FROM intake_events GROUP BY state')})
        rows = conn.execute('SELECT * FROM intake_events ORDER BY id DESC LIMIT 20').fetchall()
        channels = []
        for channel, description in CHANNEL_HELP.items():
            status = conn.execute('SELECT count(*) AS n,max(received_at) AS latest FROM intake_events WHERE channel=?', (channel,)).fetchone()
            failed = conn.execute("SELECT count(*) FROM intake_events WHERE channel=? AND state='failed'", (channel,)).fetchone()[0]
            channels.append(dict(id=channel, name=SOURCE_NAMES[CHANNELS[channel]], description=description,
                                 events=status['n'], last_event_at=status['latest'], failures=failed))
        return jsonify(summary=dict(total=sum(counts.values()), **counts), items=[public_row(row) for row in rows],
                       channels=channels, relay_configured=bool(os.environ.get('GHOST_INGEST_TOKEN')), demo=is_demo())

    @app.post('/api/intake/<int:event_id>/resolve')
    def resolve_intake(event_id):
        payload = request.get_json()
        action = payload.get('action') if isinstance(payload, dict) else None
        if action not in ('confirm_unmet', 'dismiss', 'retry'):
            raise ValueError('Choose confirm_unmet, dismiss or retry.')
        conn = connection()
        row = conn.execute('SELECT * FROM intake_events WHERE id=?', (event_id,)).fetchone()
        if row is None: return jsonify(error='Incoming event not found.'), 404
        if row['state'] == 'captured': return result(conn, event_id, True)
        if action == 'dismiss':
            conn.execute("UPDATE intake_events SET state='ignored',detail=?,updated_at=? WHERE id=?",
                         ('A team member confirmed there was no missed request.', datetime.now(timezone.utc).isoformat(), event_id))
            conn.commit()
            return result(conn, event_id)
        if action == 'retry' and row['state'] != 'failed':
            return jsonify(error='Only failed events need a retry.'), 409
        if action == 'confirm_unmet' and row['state'] not in ('needs_context', 'ignored'):
            return jsonify(error='Wait for analysis to finish before confirming the outcome.'), 409
        return analyse(conn, row, confirm=action == 'confirm_unmet')

    @app.post('/api/demo/intake')
    def demo_intake():
        if not is_demo(): return jsonify(error='Sample events are available only in the demo workspace.'), 400
        examples = [
            ('whatsapp', dict(text='Need 2 wireless earbuds but they are sold out. Budget KES 6,000. Needed today.', product='Wireless Earbuds')),
            ('email', dict(text='The quote for 2 USB-C fast chargers was rejected because the price is too high. Budget KES 2,000.', product='USB-C Fast Charger', outcome='unmet')),
            ('chatbot', dict(text='Do you sell the insulated bottle in blue?', outcome='pending')),
            ('sales_notes', dict(text='Customer requested AirLite running shoes size 42. We do not stock that size.', product='AirLite Running Shoes — Size 42')),
            ('voice_notes', dict(transcript='Customer requested an Everyday Backpack today, but delivery misses the deadline.', product='Everyday Backpack')),
        ]
        items = []
        for channel, payload in examples:
            response = receive(channel, dict(event_id='demo-auto-v1-' + channel, customer='Synthetic ' + channel,
                                             currency=default_currency, **payload))
            if isinstance(response, tuple):
                body, status = response
                return jsonify(body), status
            items.append(response)
        return jsonify(items=items, synthetic=True, message='Five synthetic channel events processed. Repeats are counted once.')

    return receive
