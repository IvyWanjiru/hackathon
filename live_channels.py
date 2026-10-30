"""Receive Meta Cloud API messages and watch a dedicated Gmail IMAP label.

No outbound messaging, mailbox-wide import or browser-stored credentials.
"""
import hashlib
import hmac
import imaplib
import json
import os
import re
import ssl
import threading
import time
import uuid
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser
from urllib.parse import urlsplit

from flask import Response, jsonify, request

from workspace_auth import privacy_ready


def now():
    return datetime.now(timezone.utc).isoformat()


def public_base():
    value = os.environ.get('GHOST_PUBLIC_URL', '').rstrip('/')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path):
        return None
    return value


def gmail_config():
    account = os.environ.get('GHOST_GMAIL_ACCOUNT', '').strip()
    password = os.environ.get('GHOST_GMAIL_APP_PASSWORD', '').replace(' ', '')
    folder = os.environ.get('GHOST_GMAIL_FOLDER', 'GhostDemandDemo').strip()
    enabled = os.environ.get('GHOST_GMAIL_ENABLED', '1') != '0'
    if not re.fullmatch(r'[A-Za-z0-9 _.-]{1,80}', folder) or folder.casefold() in ('inbox', 'all mail'):
        return None
    if not enabled or not account or '@' not in account or len(password) != 16:
        return None
    try:
        interval = max(10, min(300, int(os.environ.get('GHOST_GMAIL_POLL_SECONDS', '15'))))
    except ValueError:
        return None
    return dict(account=account, password=password, folder=folder, interval=interval)


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.hidden += 1
        if tag in ('p', 'div', 'br', 'li'): self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)


def email_event(raw, event_id, *, oversized=False):
    message = BytesParser(policy=policy.default).parsebytes(raw)
    subject = str(message.get('Subject', '')).strip()[:250]
    sender = parseaddr(str(message.get('From', '')))[1].casefold()
    customer = 'Email contact ' + hashlib.sha256(sender.encode()).hexdigest()[:12] if sender else None
    text = ''
    if not oversized:
        part = message.get_body(preferencelist=('plain', 'html')) if message.is_multipart() else message
        if part is not None and part.get_content_disposition() != 'attachment':
            try: content = part.get_content()
            except (LookupError, UnicodeError, ValueError, TypeError): content = ''
            if isinstance(content, str):
                if part.get_content_type() == 'text/html':
                    parser = TextOnly(); parser.feed(content); content = ''.join(parser.parts)
                text = content.strip()
    pending = oversized or not text
    if pending:
        text = 'Email content needs manual review. Attachments are not imported.'
    combined = (('Subject: ' + subject + '\n\n') if subject else '') + text
    truncated = len(combined) > 10000
    if truncated:
        combined = combined[:9900] + '\n[Message shortened; confirm the full enquiry in Gmail.]'
    return dict(event_id=event_id, text=combined, customer=customer,
                outcome='pending' if pending or truncated else None)


def register_live_channels(app, *, db, receive, is_demo):
    def store():
        conn = db()
        conn.execute('''CREATE TABLE IF NOT EXISTS live_channel_state(
            name TEXT PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}',
            lease_owner TEXT, lease_until REAL NOT NULL DEFAULT 0)''')
        return conn

    def read_state(name):
        row = store().execute('SELECT data FROM live_channel_state WHERE name=?', (name,)).fetchone()
        return json.loads(row['data']) if row else {}

    def write_state(name, data):
        conn = store()
        conn.execute('INSERT INTO live_channel_state(name,data) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET data=excluded.data',
                     (name, json.dumps(data)))
        conn.commit()

    def lease(owner):
        conn = store()
        conn.execute("INSERT OR IGNORE INTO live_channel_state(name,data) VALUES('gmail','{}')")
        conn.commit()
        claimed = conn.execute("UPDATE live_channel_state SET lease_owner=?,lease_until=? WHERE name='gmail' AND lease_until<?",
                               (owner, time.time() + 120, time.time())).rowcount
        conn.commit()
        return bool(claimed)

    def release(owner):
        conn = store()
        conn.execute("UPDATE live_channel_state SET lease_until=0,lease_owner=NULL WHERE name='gmail' AND lease_owner=?", (owner,))
        conn.commit()

    def ready():
        return privacy_ready()

    @app.route('/webhooks/whatsapp', methods=['GET', 'POST'])
    def meta_whatsapp():
        if not ready(): return jsonify(error='Configure private workspace access before receiving real messages.'), 503
        verify = os.environ.get('WHATSAPP_VERIFY_TOKEN', '')
        secret = os.environ.get('WHATSAPP_APP_SECRET', '')
        phone_id = os.environ.get('WHATSAPP_PHONE_NUMBER_ID', '')
        if not verify or not secret or not phone_id:
            return jsonify(error='WhatsApp needs server configuration.'), 503
        if request.method == 'GET':
            supplied = request.args.get('hub.verify_token', '')
            challenge = request.args.get('hub.challenge', '')
            if (request.args.get('hub.mode') != 'subscribe' or not challenge or len(challenge) > 1000
                    or not hmac.compare_digest(supplied.encode(), verify.encode())):
                return jsonify(error='Webhook verification was not accepted.'), 403
            if is_demo(): return jsonify(error='Provider callbacks use the live workspace.'), 400
            state = read_state('whatsapp')
            state.update(verified_at=now())
            write_state('whatsapp', state)
            return Response(challenge, mimetype='text/plain')
        expected = 'sha256=' + hmac.new(secret.encode(), request.get_data(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(request.headers.get('X-Hub-Signature-256', '').encode(), expected.encode()):
            return jsonify(error='Webhook signature was not accepted.'), 401
        if is_demo(): return jsonify(error='Provider callbacks use the live workspace.'), 400
        payload = request.get_json()
        if not isinstance(payload, dict) or payload.get('object') != 'whatsapp_business_account':
            return jsonify(error='Use a WhatsApp Business webhook payload.'), 400
        results, ignored = [], 0
        entries = payload.get('entry', [])
        if not isinstance(entries, list): raise ValueError('Invalid webhook entry list.')
        for entry in entries:
            if not isinstance(entry, dict): continue
            changes = entry.get('changes', [])
            if not isinstance(changes, list): continue
            for change in changes:
                if not isinstance(change, dict): continue
                value = change.get('value', {})
                if not isinstance(value, dict): continue
                metadata = value.get('metadata', {})
                if not isinstance(metadata, dict) or metadata.get('phone_number_id') != phone_id:
                    ignored += 1; continue
                messages = value.get('messages', [])
                if not isinstance(messages, list): raise ValueError('Invalid webhook message list.')
                for message in messages:
                    if not isinstance(message, dict) or message.get('type') != 'text':
                        ignored += 1; continue
                    text_part = message.get('text', {})
                    text = text_part.get('body') if isinstance(text_part, dict) else None
                    identity, sender = message.get('id'), message.get('from', '')
                    if not isinstance(identity, str) or not identity or not isinstance(sender, str):
                        ignored += 1; continue
                    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
                        ignored += 1; continue
                    event_id = 'meta-' + hashlib.sha256(identity.encode()).hexdigest()
                    contact = 'WhatsApp contact ' + hashlib.sha256(sender.encode()).hexdigest()[:12]
                    result = receive('whatsapp', dict(event_id=event_id, text=text, customer=contact))
                    if isinstance(result, tuple):
                        body, status = result
                        if status >= 500: return jsonify(body), status
                        ignored += 1
                    else:
                        results.append(result)
        state = read_state('whatsapp')
        state.update(last_webhook_at=now())
        if results: state.update(last_message_at=now())
        write_state('whatsapp', state)
        return jsonify(accepted=len(results), ignored=ignored)

    def gmail_sync():
        config = gmail_config()
        if not ready() or config is None:
            return dict(state='setup_needed', message='Configure private workspace access and a Gmail demo label.')
        owner = uuid.uuid4().hex
        if not lease(owner): return dict(state='busy', message='Another listener is checking this label.')
        client = None
        state = read_state('gmail')
        state.update(last_attempt_at=now())
        try:
            client = imaplib.IMAP4_SSL('imap.gmail.com', 993, ssl_context=ssl.create_default_context(), timeout=15)
            client.login(config['account'], config['password'])
            selected, _ = client.select('"' + config['folder'] + '"', readonly=True)
            if selected != 'OK':
                raise ValueError('label_missing')
            _, validity_response = client.response('UIDVALIDITY')
            _, next_response = client.response('UIDNEXT')
            validity = int(validity_response[0])
            uid_next = int(next_response[0])
            fingerprint = hashlib.sha256((config['account'].casefold() + '\n' + config['folder']).encode()).hexdigest()
            if state.get('config_fingerprint') != fingerprint or state.get('uid_validity') != validity:
                # Start with new mail only. Never import an existing personal mailbox.
                state = dict(config_fingerprint=fingerprint, uid_validity=validity, last_uid=uid_next-1,
                             connected_at=now(), last_attempt_at=now(), state='listening',
                             message='Listening for new messages in the demo label.', last_success_at=now())
                write_state('gmail', state)
                return dict(state='listening', baseline=True, imported=0)
            status, found = client.uid('search', None, 'UID', f"{state['last_uid']+1}:*")
            if status != 'OK': raise ValueError('mailbox_search_failed')
            uids = sorted({int(value) for value in (found[0] or b'').split() if int(value) > state['last_uid']})[:10]
            imported = 0
            for uid in uids:
                conn = store()
                renewed = conn.execute("UPDATE live_channel_state SET lease_until=? WHERE name='gmail' AND lease_owner=?",
                                       (time.time()+120, owner)).rowcount
                conn.commit()
                if not renewed: raise ValueError('listener_lease_lost')
                status, metadata = client.uid('fetch', str(uid), '(RFC822.SIZE BODY.PEEK[HEADER.FIELDS (FROM SUBJECT MESSAGE-ID)])')
                if status != 'OK': raise ValueError('mailbox_fetch_failed')
                tuples = [item for item in metadata if isinstance(item, tuple)]
                if not tuples: raise ValueError('mailbox_fetch_failed')
                match = re.search(rb'RFC822.SIZE\s+(\d+)', tuples[0][0])
                if not match: raise ValueError('mailbox_size_missing')
                oversized = int(match.group(1)) > 262144
                raw = tuples[0][1]
                if not oversized:
                    status, content = client.uid('fetch', str(uid), '(BODY.PEEK[])')
                    if status != 'OK': raise ValueError('mailbox_fetch_failed')
                    pieces = [item[1] for item in content if isinstance(item, tuple)]
                    if not pieces: raise ValueError('mailbox_fetch_failed')
                    raw = pieces[0]
                identity = hashlib.sha256(f'{fingerprint}:{validity}:{uid}'.encode()).hexdigest()
                result = receive('email', email_event(raw, 'gmail-' + identity, oversized=oversized))
                if isinstance(result, tuple):
                    raise ValueError('intake_failed')
                state.update(last_uid=uid, last_message_at=now())
                imported += 1
                write_state('gmail', state)
            state.update(state='listening', message='Listening for new messages in the demo label.', last_success_at=now())
            write_state('gmail', state)
            return dict(state='listening', imported=imported)
        except Exception as exc:
            if isinstance(exc, imaplib.IMAP4.error):
                message = 'Gmail sign-in failed. Check the app password and label access.'
            elif isinstance(exc, ValueError) and str(exc) == 'label_missing':
                message = 'Create the demo label and make it available through IMAP.'
            elif isinstance(exc, ValueError) and str(exc) == 'intake_failed':
                message = 'An enquiry could not be analysed. It will be retried.'
            else:
                message = 'Gmail could not be checked. The listener will retry.'
            state.update(state='attention', message=message)
            write_state('gmail', state)
            app.logger.warning('Gmail listener needs attention (%s).', type(exc).__name__)
            return dict(state='attention', message=message)
        finally:
            if client is not None:
                try: client.logout()
                except Exception: pass
            release(owner)

    @app.get('/api/connections')
    def connection_status():
        if is_demo(): return jsonify(demo=True, providers={}, private_workspace=False)
        meta = read_state('whatsapp')
        mail = read_state('gmail')
        meta_configured = ready() and all(os.environ.get(key) for key in
                             ('WHATSAPP_VERIFY_TOKEN', 'WHATSAPP_APP_SECRET', 'WHATSAPP_PHONE_NUMBER_ID'))
        config = gmail_config()
        gmail_configured = ready() and config is not None
        return jsonify(demo=False, private_workspace=ready(), public_url_configured=bool(public_base()), providers=dict(
            whatsapp=dict(configured=bool(meta_configured), state='receiving' if meta_configured and meta.get('last_message_at') else 'verified' if meta_configured and meta.get('verified_at') else 'waiting' if meta_configured else 'setup_needed',
                          callback_url=public_base() + '/webhooks/whatsapp' if public_base() else None,
                          verified_at=meta.get('verified_at'), last_message_at=meta.get('last_message_at')),
            gmail=dict(configured=gmail_configured, state=mail.get('state','waiting') if gmail_configured else 'setup_needed',
                       label=config['folder'] if config else 'GhostDemandDemo', message=mail.get('message') if gmail_configured else 'Configure a Gmail app password and the dedicated demo label.',
                       last_success_at=mail.get('last_success_at'), last_message_at=mail.get('last_message_at'))))

    @app.post('/api/connections/gmail/check')
    def check_gmail():
        if is_demo(): return jsonify(error='Open the private live workspace to check Gmail.'), 400
        if not ready(): return jsonify(error='Private workspace setup is required.'), 503
        result = gmail_sync()
        if result['state'] in ('setup_needed','attention'):
            result['error'] = result['message']
        return jsonify(result), 503 if result['state'] in ('setup_needed','attention') else 200

    app.extensions['ghost_gmail_sync'] = gmail_sync

    def start_listener():
        if getattr(app, '_ghost_gmail_started', False) or gmail_config() is None or not ready(): return
        app._ghost_gmail_started = True
        def run():
            while True:
                needs_backoff = False
                try:
                    with app.app_context():
                        result = gmail_sync()
                        needs_backoff = result.get('state') == 'attention'
                except Exception as exc:
                    needs_backoff = True
                    app.logger.warning('Gmail listener will retry (%s).', type(exc).__name__)
                config = gmail_config()
                delay = config['interval'] if config else 30
                time.sleep(max(60, delay) if needs_backoff else delay)
        threading.Thread(target=run, name='ghost-gmail-listener', daemon=True).start()

    return start_listener
