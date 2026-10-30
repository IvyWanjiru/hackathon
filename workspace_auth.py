"""Private live workspace; public synthetic demo and signed provider callbacks."""
import hmac
import os
import secrets
from datetime import timedelta
from urllib.parse import urlsplit

from flask import jsonify, redirect, render_template, request, session
from flask.sessions import SecureCookieSessionInterface
from werkzeug.security import check_password_hash


def integrations_requested():
    return any(os.environ.get(key) for key in ('WHATSAPP_APP_SECRET', 'WHATSAPP_PHONE_NUMBER_ID',
                                               'GHOST_GMAIL_ACCOUNT', 'GHOST_GMAIL_APP_PASSWORD'))


def password_configured():
    return bool(os.environ.get('GHOST_ADMIN_PASSWORD_HASH') or os.environ.get('GHOST_ADMIN_PASSWORD'))


def privacy_ready():
    strong_password = bool(os.environ.get('GHOST_ADMIN_PASSWORD_HASH')) or len(os.environ.get('GHOST_ADMIN_PASSWORD', '')) >= 12
    return strong_password and len(os.environ.get('GHOST_SECRET_KEY', '')) >= 32


def safe_next(value):
    if (not isinstance(value, str) or len(value) > 2000 or any(ord(c) < 32 for c in value)
            or not value.startswith('/') or value.startswith('//') or '\\' in value):
        return '/dashboard#activity'
    parsed = urlsplit(value)
    return value if not parsed.scheme and not parsed.netloc else '/dashboard#activity'


class WorkspaceSessions(SecureCookieSessionInterface):
    def get_cookie_secure(self, app):
        # Local rehearsal uses loopback HTTP even when the cloud callback is HTTPS.
        host = urlsplit('http://' + request.host).hostname
        if host in ('127.0.0.1', 'localhost', '::1') and request.remote_addr in ('127.0.0.1', '::1'):
            return bool(request.is_secure)
        return super().get_cookie_secure(app)


def register_workspace_auth(app, is_demo):
    app.session_interface = WorkspaceSessions()
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                      SESSION_COOKIE_SECURE=True,
                      PERMANENT_SESSION_LIFETIME=timedelta(hours=8))

    @app.before_request
    def protect_workspace():
        enabled = password_configured() or integrations_requested()
        if not enabled:
            return None
        path = request.path
        public = path in ('/', '/demo', '/present', '/login', '/webhooks/whatsapp') or path.startswith('/static/')
        public = public or path.startswith('/api/demo/') or path.startswith('/api/ingest/')
        demo_api = is_demo() and (path.startswith('/api/signals') or path in
                    ('/api/insights', '/api/intake', '/api/catalog', '/api/model', '/api/connections', '/api/operations/follow-ups')
                    or path.startswith('/api/intake/') or path.startswith('/api/pesashield/'))
        if public or demo_api:
            return None
        if not password_configured():
            if path.startswith('/api/'):
                return jsonify(error='Set a workspace login before enabling live business channels.'), 503
            return redirect('/login')
        if not session.get('workspace_authenticated'):
            if path.startswith('/api/'):
                return jsonify(error='Sign in to the live workspace.', login_required=True), 401
            return redirect('/login?next=' + '/dashboard')
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            supplied = request.headers.get('X-Ghost-CSRF', '') or (request.form.get('csrf', '') if path == '/logout' else '')
            expected = session.get('csrf', '')
            if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
                return jsonify(error='Refresh the workspace and try again.'), 403

    @app.route('/login', methods=['GET', 'POST'])
    def workspace_login():
        session.setdefault('csrf', secrets.token_urlsafe(32))
        error = None
        if request.method == 'POST':
            supplied_csrf = request.form.get('csrf', '')
            if not hmac.compare_digest(supplied_csrf.encode(), session['csrf'].encode()):
                error = 'Refresh the sign-in page and try again.'
            elif not password_configured():
                error = 'Workspace sign-in needs server configuration.'
            else:
                supplied = request.form.get('password', '')
                hashed = os.environ.get('GHOST_ADMIN_PASSWORD_HASH')
                valid = check_password_hash(hashed, supplied) if hashed else hmac.compare_digest(
                    supplied.encode(), os.environ.get('GHOST_ADMIN_PASSWORD', '').encode())
                if valid:
                    session.clear()
                    session.update(workspace_authenticated=True, csrf=secrets.token_urlsafe(32))
                    session.permanent = True
                    return redirect(safe_next(request.form.get('next')))
                error = 'The workspace password was not accepted.'
        return render_template('login.html', error=error, csrf=session['csrf'],
                               next_path=safe_next(request.args.get('next')), configured=password_configured()), 400 if error else 200

    @app.post('/logout')
    def workspace_logout():
        session.clear()
        return redirect('/login')
