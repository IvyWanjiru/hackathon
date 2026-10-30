"""GhostDemand AI turns lost sales into structured demand signals."""
import json
import csv
import hashlib
import io
import math
import os
import re
import secrets
import hmac
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import click
from flask import Flask, Response, g, has_request_context, jsonify, redirect, render_template, request, send_file, session, send_from_directory
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

from demand_ml import REASON_LABELS, ReasonClassifier, TrainingError
from fraud_ml import SCAM_LABELS, ScamClassifier, TrainingError as FraudTrainingError, STARTER_PATH as FRAUD_STARTER_PATH
from intelligence import (
    CURRENCIES, KINDS, REASON_NAMES, SOURCE_NAMES, STATUSES, attach_groups, detect_currency,
    enrich_signal, extract_specifications, follow_up_queue, parse_conversation, product_key,
    request_kind, summarize,
)

ROOT = Path(__file__).resolve().parent
if os.environ.get('GHOST_LOAD_ENV', '1') != '0':
    load_dotenv(ROOT / '.env', override=False)
app = Flask(__name__, template_folder=str(ROOT))
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.secret_key = os.environ.get("GHOST_SECRET_KEY") or secrets.token_hex(32)
preview_signer = URLSafeTimedSerializer(app.secret_key, salt="ghost-preview-v1")
DB = os.environ.get("GHOST_DB", str(ROOT / "ghostdemand.db"))
DEMO_DB = os.environ.get("GHOST_DEMO_DB", str(ROOT / "ghostdemand-demo.db"))
from workspace_auth import integrations_requested
if integrations_requested() and Path(DB).resolve() == Path(DEMO_DB).resolve():
    raise ValueError('Live integrations require separate live and demo databases.')
DEFAULT_CURRENCY = os.environ.get("GHOST_CURRENCY", "KES")
BUSINESS_TIMEZONE = ZoneInfo(os.environ.get("GHOST_TIMEZONE", "Africa/Nairobi"))
if DEFAULT_CURRENCY not in CURRENCIES:
    raise ValueError("GHOST_CURRENCY must be one of " + ", ".join(CURRENCIES))
ML = ReasonClassifier(
    os.environ.get("GHOST_ML_PATH", str(ROOT / "models" / "reason_classifier.json")),
    enabled=os.environ.get("GHOST_ML_ENABLED", "1").lower() not in ("0", "false", "no"),
    threshold=float(os.environ.get("GHOST_ML_THRESHOLD", "0.55")),
)
DEMO_ML = ReasonClassifier(ROOT / "models" / "demo_reason_classifier.json")
FRAUD_ML = ScamClassifier(
    os.environ.get("GHOST_FRAUD_ML_PATH", str(ROOT / "models" / "scam_classifier.json")),
    enabled=os.environ.get("GHOST_FRAUD_ML_ENABLED", "1").lower() not in ("0", "false", "no"),
    threshold=float(os.environ.get("GHOST_FRAUD_ML_THRESHOLD", "0.6")),
)
DEMO_FRAUD_ML = ScamClassifier(ROOT / "models" / "demo_scam_classifier.json")
PESASHIELD_ENABLED = os.environ.get("PESASHIELD_ENABLED", "1").lower() not in ("0", "false", "no")
SOURCES = list(SOURCE_NAMES)


def is_demo():
    return has_request_context() and request.headers.get("X-Ghost-Demo") == "1"


def active_model():
    return DEMO_ML if is_demo() else ML


def active_fraud_model():
    return DEMO_FRAUD_ML if is_demo() else FRAUD_ML


# ---------- storage (additive migration preserves existing signals) ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DEMO_DB if is_demo() else DB, timeout=15)
        g.db.row_factory = sqlite3.Row
        g.db.execute("""CREATE TABLE IF NOT EXISTS signals(
            id INTEGER PRIMARY KEY, ts TEXT, source TEXT, raw TEXT, product TEXT, pkey TEXT,
            qty INTEGER, budget REAL, reason TEXT, urgency INTEGER, sentiment TEXT,
            location TEXT, customer TEXT)""")
        additions = {
            "reason_engine": "TEXT NOT NULL DEFAULT 'rules'", "ml_reason": "TEXT",
            "ml_confidence": "REAL", "model_version": "TEXT", "reviewed_reason": "TEXT",
            "reviewed_at": "TEXT",
            "currency": "TEXT NOT NULL DEFAULT 'UNKNOWN'", "external_id": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'open'", "owner": "TEXT", "next_action": "TEXT",
            "recovered_amount": "REAL", "recovered_currency": "TEXT", "updated_at": "TEXT",
            "follow_up_on": "TEXT",
            "specifications": "TEXT", "kind": "TEXT NOT NULL DEFAULT 'product'",
        }
        columns = {row[1] for row in g.db.execute("PRAGMA table_info(signals)")}
        if set(additions) - columns:
            # Recheck under a write lock when multiple requests start together.
            g.db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in g.db.execute("PRAGMA table_info(signals)")}
            for name, sql_type in additions.items():
                if name not in columns:
                    g.db.execute(f"ALTER TABLE signals ADD COLUMN {name} {sql_type}")
            g.db.commit()
        g.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS signal_event ON signals(source,external_id) WHERE external_id IS NOT NULL")
        g.db.execute("""CREATE TABLE IF NOT EXISTS catalog(
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, sku TEXT, created_at TEXT NOT NULL)""")
        g.db.execute("""CREATE TABLE IF NOT EXISTS product_aliases(
            alias_key TEXT PRIMARY KEY, catalog_id INTEGER NOT NULL REFERENCES catalog(id))""")
        g.db.execute("""CREATE TABLE IF NOT EXISTS fraud_messages(
            id INTEGER PRIMARY KEY, ts TEXT, channel TEXT, sender_hash TEXT, sender_raw_hash_algo TEXT,
            body_hash TEXT, body TEXT, urls_json TEXT, amounts_json TEXT, mpesa_refs_json TEXT,
            extracted_till TEXT, extracted_account TEXT, risk_score REAL, verdict TEXT,
            reason_codes_json TEXT, model_version TEXT, ml_confidence REAL, reason_engine TEXT,
            reviewed_verdict TEXT, reviewed_at TEXT, reviewed_by TEXT, status TEXT DEFAULT 'open',
            location_hint TEXT, analysis_json TEXT)""")
        fraud_columns = {row[1] for row in g.db.execute("PRAGMA table_info(fraud_messages)")}
        if "analysis_json" not in fraud_columns:
            g.db.execute("BEGIN IMMEDIATE")
            fraud_columns = {row[1] for row in g.db.execute("PRAGMA table_info(fraud_messages)")}
            if "analysis_json" not in fraud_columns:
                g.db.execute("ALTER TABLE fraud_messages ADD COLUMN analysis_json TEXT")
            g.db.commit()
        g.db.execute("""CREATE TABLE IF NOT EXISTS fraud_transactions(
            id INTEGER PRIMARY KEY, ts TEXT, amount REAL, currency TEXT DEFAULT 'KES',
            sender_hash TEXT, recipient_hash TEXT, channel TEXT, device_hint TEXT, hour_of_day INTEGER,
            is_new_recipient INTEGER, velocity_24h INTEGER, risk_score REAL, verdict TEXT, flags_json TEXT,
            reviewed_verdict TEXT, reviewed_at TEXT)""")
        g.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS fraud_msg_dedup ON fraud_messages(channel,body_hash,sender_hash)")
    return g.db


@app.teardown_appcontext
def close(_):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


# ---------- free text -> structured demand signal ----------
REASONS = {
    "out_of_stock": r"out of stock|sold out|no stock|finished|restock|not in stock|stock was depleted|zero units|inventory is exhausted",
    "price_too_high": r"too (expensive|high|much)|cheaper|discount|pricey|over budget|lower price|best price",
    "unsupported_service": r"don'?t (offer|provide|do) (that |this )?(service|install|repair)|we (do not|don't) (install|repair|assemble|deliver)|"
                           r"service not (available|offered|supported)|not a service we|cannot (install|repair|assemble)|we only sell",
    "not_stocked": r"don'?t (have|sell|stock)|do not (carry|sell|stock)|not available|never stocked|outside our (range|assortment)|"
                   r"not (in|part of) (our )?(range|catalogue|catalog)|we do not carry",
    "quote_rejected": r"declined|rejected|went (with|elsewhere)|competitor|chose another|lost the (deal|quote)",
    "lead_time": r"too long|delivery time|lead time|when (will|can)|wait",
    "abandoned": r"abandon|never replied|went quiet|left the (cart|chat)|no response|ghosted",
}
URGENT = r"urgent|asap|today|immediately|tomorrow|right now|emergency"
NEG = r"disappoint|annoy|angry|frustrat|useless|never again|terrible|waste"
POS = r"thanks|great|please|love|interested"


def parse_money(value):
    value = value.replace(",", "").lower()
    return float(value[:-1]) * 1000 if value.endswith("k") else float(value)


def normalize(product):
    return product_key(product)


def rule_extract(text, source):
    text = text.strip()
    low = text.lower()
    reason = next((r for r, pattern in REASONS.items() if r not in ("not_stocked", "unsupported_service") and re.search(pattern, low)), None)
    if not reason and re.search(REASONS["unsupported_service"], low):
        reason = "unsupported_service"
    if not reason and re.search(REASONS["not_stocked"], low):
        reason = "not_stocked"
    reason = reason or {"out_of_stock": "out_of_stock", "rejected_quote": "quote_rejected",
                        "abandoned_enquiry": "abandoned", "unavailable_product": "not_stocked"}.get(source, "unknown")
    qty = re.search(r"(\d{1,9})\s*(?:pcs|pieces|units|bags|boxes|cartons|kg|litres|liters|tons?|x)\b", low) \
        or re.search(r"\b(\d{1,5})\s+(?=[a-z])", low)
    budget = re.search(r"(?:kshs?|kes|usd|eur|gbp|ugx|tzs|\$|€|£|budget(?: of)?|up to)\s*([\d,]+(?:\.\d+)?k?)", low)
    loc = re.search(r"(?:deliver(?:ed)? to|in|to|at)\s+([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)", text)
    prod = re.search(r"(?:looking for|do you (?:have|sell|stock)|need|want|searching for|searched for|"
                     r"requested|asked for|enquir\w+ (?:about|for)|out of stock[: ]+|quote for)\s+(.+?)"
                     r"(?:\?|\.|,|;| for | in | to | by | at | but | because | urgent| asap|$)", text, re.I)
    product = prod.group(1) if prod else (text if len(text.split()) <= 6 else " ".join(text.split()[:6]))
    product = re.sub(r"^\d+(?:\s*(?:pcs|pieces|units|bags|boxes|cartons|kg|litres|liters|tons?|x)\b\s*|\s+)(?:of\s+)?",
                     "", product.strip(), flags=re.I).strip(" -:") or text[:80]
    specifications = extract_specifications(text) or extract_specifications(product) or None
    sentiment = "negative" if re.search(NEG, low) else "positive" if re.search(POS, low) else "neutral"
    urgency = 3 if re.search(URGENT, low) else 2 if reason in ("out_of_stock", "quote_rejected") else 1
    money = parse_money(budget.group(1)) if budget else None
    return dict(product=product.title(), qty=max(1, int(qty.group(1))) if qty else None,
                budget=money if money is None or math.isfinite(money) else None, reason=reason,
                urgency=urgency, sentiment=sentiment, location=loc.group(1) if loc else None,
                specifications=specifications, kind=request_kind(text))


def llm_extract(text, source):
    """Optional richer field extraction when ANTHROPIC_API_KEY is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        message = anthropic.Anthropic(timeout=20, max_retries=0).messages.create(
            model=os.environ.get("GHOST_MODEL", "claude-sonnet-5-5"), max_tokens=300,
            system="Extract a lost-sale demand signal. Reply with JSON only: product (str), specifications (str|null), "
                   "kind (product|service), qty (int|null), budget (number|null), reason (out_of_stock|price_too_high|"
                   "not_stocked|unsupported_service|quote_rejected|lead_time|abandoned|unknown), urgency (1-3), "
                   "sentiment (negative|neutral|positive), location (str|null). Use null for quantity or budget when "
                   "the customer did not state them. Use unknown when the unsuccessful outcome is not explicit.",
            messages=[{"role": "user", "content": f"Source: {source}\nText: {text}"}])
        payload = json.loads(re.sub(r"```json|```", "", message.content[0].text).strip())
        return payload if isinstance(payload, dict) else None
    except Exception:
        app.logger.warning("Claude extraction unavailable; using local extraction.")
        return None


def validated_fields(candidate):
    """External model output cannot replace trusted fields with invalid values."""
    clean = {}
    for key in ("product", "location"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            clean[key] = value.strip()[:250]
    for key, allowed in (("reason", REASON_LABELS), ("sentiment", ("negative", "neutral", "positive")),
                         ("kind", KINDS)):
        if candidate.get(key) in allowed:
            clean[key] = candidate[key]
    specs = candidate.get("specifications")
    if isinstance(specs, str) and specs.strip():
        clean["specifications"] = specs.strip()[:250]
    if type(candidate.get("qty")) is int and 1 <= candidate["qty"] <= 1_000_000_000:
        clean["qty"] = candidate["qty"]
    if type(candidate.get("urgency")) is int and candidate["urgency"] in (1, 2, 3):
        clean["urgency"] = candidate["urgency"]
    budget = candidate.get("budget")
    if type(budget) in (float, int) and 0 <= budget <= 1e15 and math.isfinite(budget):
        clean["budget"] = budget
    return clean


def extract(text, source):
    base = rule_extract(text, source)
    richer = validated_fields((llm_extract(text, source) if not is_demo() and not g.get('local_extraction_only') else None) or {})
    base.update(richer)
    base["reason_engine"] = "claude" if "reason" in richer else "rules"
    prediction = active_model().predict(text)
    if prediction and prediction["used"]:
        base["reason"] = prediction["reason"]
        base["reason_engine"] = "local_ml"
    base["ml_reason"] = prediction["reason"] if prediction else None
    base["ml_confidence"] = prediction["confidence"] if prediction else None
    base["model_version"] = prediction["model_version"] if prediction else None
    return base


def prepare_signal(item):
    if not isinstance(item, dict):
        raise ValueError("Each signal must be a JSON object.")
    text = item.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("'text' must contain between 1 and 10,000 characters.")
    source = item.get("source", "chatbot")
    if source not in SOURCES:
        raise ValueError("'source' must be a supported channel.")
    customer = item.get("customer")
    if customer is not None and (not isinstance(customer, str) or len(customer) > 250):
        raise ValueError("'customer' must be a string of up to 250 characters.")
    return text.strip(), source, customer


def input_options(payload):
    currency = payload.get("currency", DEFAULT_CURRENCY)
    if currency not in CURRENCIES:
        raise ValueError("Choose a supported currency.")
    external_id = payload.get("external_id")
    if external_id is not None and (not isinstance(external_id, str) or len(external_id) > 200):
        raise ValueError("'external_id' must be a string of up to 200 characters.")
    external_id = external_id or None
    return currency, external_id


def catalog_match(text, product):
    aliases = db().execute("SELECT a.alias_key,c.id,c.name FROM product_aliases a JOIN catalog c ON c.id=a.catalog_id").fetchall()
    text_key = " " + product_key(text) + " "
    product_normalized = product_key(product)
    matches = {r["id"]: r["name"] for r in aliases
               if r["alias_key"] == product_normalized or " " + r["alias_key"] + " " in text_key}
    warnings = []
    if len(matches) > 1:
        warnings.append("This message mentions multiple catalog products. Capture each product separately.")
        return product, warnings
    if len(matches) == 1:
        canonical = next(iter(matches.values()))
        variants = re.findall(r"(?:size\s*\w+|\b\d+\s*(?:gb|tb|mm|inch)\b)", text, re.I)
        if variants and any(product_key(v) not in product_key(canonical) for v in variants):
            warnings.append("A size or specification needs review. Keep different variants separate.")
            return product, warnings
        return canonical, warnings
    return product, warnings


def draft_signal(text, source, currency):
    signal = extract(text, source)
    signal["product"], warnings = catalog_match(text, signal["product"])
    signal["currency"] = detect_currency(text, currency)
    if source == "website_search":
        warnings.append("A failed search can reflect naming or filters. Check actual product availability before treating it as a range gap.")
    if signal["currency"] == "UNKNOWN":
        warnings.append("The budget currency is unclear. Confirm the currency before saving.")
    if signal["reason_engine"] == "rules":
        warnings.append("Check the product and blocker against the original customer request.")
    elif signal.get("ml_confidence") is not None and signal["reason_engine"] != "local_ml":
        warnings.append("The blocker needs confirmation. Please verify it before saving.")
    return signal, warnings


def apply_fields(signal, fields):
    if not isinstance(fields, dict):
        raise ValueError("'fields' must be a JSON object.")
    allowed = {"product", "qty", "budget", "reason", "urgency", "sentiment", "location", "currency",
               "specifications", "kind"}
    if set(fields) - allowed:
        raise ValueError("Unsupported editable signal field.")
    cleaned = validated_fields(fields)
    if "currency" in fields and fields["currency"] in CURRENCIES:
        cleaned["currency"] = fields["currency"]
    for key, value in fields.items():
        if value is None and key in ("budget", "location", "qty", "specifications"):
            cleaned[key] = None
        elif key not in cleaned:
            raise ValueError(f"Invalid value for '{key}'.")
    if "reason" in cleaned and cleaned["reason"] != signal["reason"]:
        signal["reason_engine"] = "human"
    signal.update(cleaned)
    return signal


def build_preview(payload):
    text, source, customer = prepare_signal(payload)
    currency, external_id = input_options(payload)
    signal, warnings = draft_signal(text, source, currency)
    identity = dict(text=text, source=source, customer=customer, currency=currency, external_id=external_id)
    token = preview_signer.dumps(dict(identity=identity, signal=signal))
    return dict(**identity, fields=signal, warnings=warnings, preview_token=token)


def save_signal(text, source="chatbot", customer=None, *, commit=True, currency=None,
                external_id=None, fields=None, verified=False, preview_token=None, event=False):
    connection = db()
    if external_id:
        previous = connection.execute("SELECT * FROM signals WHERE source=? AND external_id=?", (source, external_id)).fetchone()
        if previous:
            return dict(previous, deduplicated=True)
    currency = currency or DEFAULT_CURRENCY
    if preview_token:
        if not isinstance(preview_token, str):
            raise ValueError("Preview the request again before saving.")
        try:
            preview = preview_signer.loads(preview_token, max_age=3600)
        except (BadSignature, SignatureExpired) as exc:
            raise ValueError("The preview expired. Preview this request again before saving.") from exc
        identity = dict(text=text, source=source, customer=customer, currency=currency, external_id=external_id)
        if preview["identity"] != identity:
            raise ValueError("The request changed. Preview it again before saving.")
        signal = dict(preview["signal"])
    else:
        signal, _ = draft_signal(text, source, currency)
    if fields is not None:
        apply_fields(signal, fields)
    signal.setdefault("kind", "product")
    signal.setdefault("specifications", None)
    if event:
        signal["reason_engine"] = "channel_event"
    if type(verified) is not bool:
        raise ValueError("'verified' must be true or false.")
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    signal["reviewed_reason"] = signal["reason"] if verified else None
    signal["reviewed_at"] = now if verified else None
    columns = ("product", "qty", "budget", "reason", "urgency", "sentiment", "location",
               "reason_engine", "ml_reason", "ml_confidence", "model_version", "currency",
               "reviewed_reason", "reviewed_at", "specifications", "kind")
    cursor = connection.execute(
        "INSERT OR IGNORE INTO signals(ts,source,raw,pkey,customer,external_id," + ",".join(columns) + ")"
        " VALUES(" + ",".join("?" for _ in range(6 + len(columns))) + ")",
        (now, source, text, normalize(signal["product"]), customer, external_id, *(signal[k] for k in columns)))
    if commit:
        connection.commit()
    if cursor.rowcount == 0:
        row = connection.execute("SELECT * FROM signals WHERE source=? AND external_id=?", (source, external_id)).fetchone()
        return dict(row, deduplicated=True)
    return {"id": cursor.lastrowid, "source": source, "raw": text, "deduplicated": False, **signal}


def save_payload(payload, commit=True):
    text, source, customer = prepare_signal(payload)
    currency, external_id = input_options(payload)
    return save_signal(text, source, customer, commit=commit, currency=currency, external_id=external_id,
                       fields=payload.get("fields"), verified=payload.get("verified", False),
                       preview_token=payload.get("preview_token"))


def reviewed_examples():
    return [dict(text=row["raw"], reason=row["reviewed_reason"], reviewed_at=row["reviewed_at"])
            for row in db().execute("SELECT raw, reviewed_reason, reviewed_at FROM signals "
                                    "WHERE reviewed_reason IS NOT NULL ORDER BY reviewed_at, id")]


def model_status():
    status = active_model().status()
    status["unreviewed_signals"] = db().execute(
        "SELECT count(*) FROM signals WHERE reviewed_reason IS NULL").fetchone()[0]
    status["reviewed_signals"] = db().execute(
        "SELECT count(*) FROM signals WHERE reviewed_reason IS NOT NULL").fetchone()[0]
    status["pending_reviews"] = db().execute(
        "SELECT count(*) FROM signals WHERE reviewed_at > ?",
        ((status["metadata"] or {}).get("reviewed_through", ""),)).fetchone()[0]
    return status


# ---------- routes ----------
@app.errorhandler(ValueError)
def invalid_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(HTTPException)
def http_error(error):
    if request.path.startswith(("/api/", "/webhook/")):
        return jsonify(error=error.description), error.code
    return error


@app.get("/")
def landing():
    return render_template("landing.html")


@app.get("/present")
def presentation():
    return redirect('/demo#activity')


@app.get("/present/notebook")
def presentation_notebook():
    path = ROOT/'notebooks'/'GhostDemand_Training_Improved.ipynb'
    if not path.is_file():
        return jsonify(error="The training notebook is unavailable in this checkout."), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.get("/present/notes")
def presentation_notes():
    path = ROOT/'PANEL_PRESENTATION.md'
    if not path.is_file():
        return jsonify(error="The presentation notes are unavailable in this checkout."), 404
    return send_file(path, as_attachment=True, download_name='GhostDemand_Panel_Notes.md')


@app.get("/dashboard")
@app.get("/demo")
def index():
    return render_template("index.html", demo=request.path == "/demo", config=dict(
        sources=SOURCE_NAMES, reasons=REASON_NAMES, statuses=STATUSES, currencies=CURRENCIES,
        kinds=KINDS, defaultCurrency=DEFAULT_CURRENCY, demo=request.path == "/demo",
        csrf=session.get('csrf', '')))


@app.post("/api/signals")
def add_signal():
    return jsonify(save_payload(request.get_json())), 201


@app.post("/api/signals/bulk")
def add_bulk():
    payload = request.get_json()
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not 1 <= len(items) <= 100:
        raise ValueError("'items' must be a list containing 1 to 100 signals.")
    for item in items:
        prepare_signal(item)
        input_options(item)
    connection = db()
    with connection:
        signals = [save_payload(item, commit=False) for item in items]
    return jsonify(signals), 201


@app.route("/webhook/whatsapp", methods=["GET", "POST"])
def whatsapp():
    if request.method == "GET":
        return "", 200
    # Support form-encoded (Twilio) and JSON
    body = ""
    from_num = ""
    msg_sid = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        body = (payload.get("Body") or payload.get("body") or payload.get("text") or "").strip()
        from_num = payload.get("From") or payload.get("from") or payload.get("WaId") or ""
        msg_sid = payload.get("MessageSid") or payload.get("SmsMessageSid") or payload.get("messageId")
    else:
        body = (request.values.get("Body") or request.values.get("body") or "").strip()
        from_num = request.values.get("From") or request.values.get("from") or ""
        msg_sid = request.values.get("MessageSid") or request.values.get("SmsMessageSid")
    if body:
        try:
            save_signal(*prepare_signal(dict(text=body, source="whatsapp", customer=from_num or None)),
                        external_id=msg_sid)
        except Exception:
            app.logger.exception("whatsapp webhook save failed")
        # Also auto-scan for fraud
        try:
            rule_codes, rule_details, rule_score = _rule_scan_sms(body)
            prediction = active_fraud_model().predict(body)
            urls = re.findall(r"http[s]?://\S+", body)
            amounts = re.findall(r"kes\s*[\d,]+|\b[\d,]+\s*kes\b|\$\s*[\d,]+", body, re.I)
            mpesa_refs = re.findall(r"\b[A-Z0-9]{10,12}\b", body)
            verdict = "legit"
            confidence = rule_score
            reason_engine = "rules"
            model_version = None
            if prediction:
                verdict = prediction["verdict"]
                confidence = max(confidence, prediction["confidence"])
                reason_engine = "local_ml"
                model_version = prediction.get("model_version")
            if rule_codes and verdict == "legit":
                verdict = "suspicious"
            if "pin_request" in rule_codes or "fake_reversal" in rule_codes:
                verdict = "scam"
                confidence = max(confidence, 0.85)
            body_hash = _hash_text(body)
            sender_hash = _hash_phone(from_num) if from_num else None
            connection = db()
            try:
                with connection:
                    connection.execute("""INSERT INTO fraud_messages(ts,channel,sender_hash,body_hash,body,urls_json,amounts_json,mpesa_refs_json,
                        risk_score,verdict,reason_codes_json,model_version,ml_confidence,reason_engine)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now(timezone.utc).isoformat(timespec="microseconds"), "whatsapp", sender_hash, body_hash,
                         body[:5000], json.dumps(urls), json.dumps(amounts), json.dumps(mpesa_refs),
                         confidence, verdict, json.dumps(rule_codes), model_version, prediction["confidence"] if prediction else None, reason_engine))
            except Exception:
                pass
        except Exception:
            app.logger.exception("whatsapp fraud scan failed")
    return "", 204


@app.get("/api/model")
def get_model():
    return jsonify(model_status())


@app.post("/api/model/train")
def train_model():
    examples = reviewed_examples()
    try:
        active_model().train(examples)
    except TrainingError as error:
        return jsonify(error=str(error)), 409
    except OSError:
        app.logger.exception("Unable to read training data or save the trained model")
        return jsonify(error="Unable to read training data or save the model. Check server file permissions."), 503
    return jsonify(model_status())


@app.patch("/api/signals/<int:signal_id>/review")
def review_signal(signal_id):
    payload = request.get_json()
    reason = payload.get("reason") if isinstance(payload, dict) else None
    if reason not in REASON_LABELS:
        raise ValueError("Choose a supported reason label.")
    cursor = db().execute(
        "UPDATE signals SET reviewed_reason=?, reviewed_at=? WHERE id=?",
        (reason, datetime.now(timezone.utc).isoformat(timespec="microseconds"), signal_id))
    if cursor.rowcount == 0:
        return jsonify(error="Signal not found."), 404
    db().commit()
    return jsonify(id=signal_id, reviewed_reason=reason)


def selected_rows():
    clauses, params = [], []
    for name, column, allowed in (("source", "source", SOURCES), ("reason", "COALESCE(reviewed_reason,reason)", REASON_LABELS),
                                  ("status", "status", STATUSES)):
        value = request.args.get(name)
        if value:
            if value not in allowed:
                raise ValueError(f"Unsupported {name} filter.")
            clauses.append(f"{column}=?")
            params.append(value)
    days = request.args.get("days", "all")
    if days != "all":
        if days not in ("7", "30", "90"):
            raise ValueError("Choose a 7, 30, 90 day or all-time view.")
        clauses.append("ts>=?")
        params.append((datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat())
    query = request.args.get("q", "").strip()[:150]
    if query:
        clauses.append("(product LIKE ? OR raw LIKE ? OR customer LIKE ?)")
        params.extend([f"%{query}%"] * 3)
    if request.args.get("review") == "needed":
        clauses.append("reviewed_reason IS NULL")
    sql = "SELECT * FROM signals" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY ts DESC, id DESC"
    return [enrich_signal(row) for row in db().execute(sql, params)]


@app.get("/api/insights")
def insights():
    return jsonify(summarize(selected_rows()))


@app.get("/api/signals")
def list_signals():
    rows = selected_rows()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError as exc:
        raise ValueError("Invalid page number.") from exc
    page_size = 15
    return jsonify(items=rows[(page-1)*page_size:page*page_size], total=len(rows), page=page,
                   pages=max(1, math.ceil(len(rows)/page_size)))


@app.get("/api/operations/follow-ups")
def get_follow_ups():
    rows = [enrich_signal(row) for row in db().execute("SELECT * FROM signals")]
    return jsonify(follow_up_queue(
        rows, today=datetime.now(BUSINESS_TIMEZONE).date(),
        status=request.args.get("status", "active"), assignment=request.args.get("assignment", "any"),
        owner=request.args.get("owner", "")[:100], urgency=request.args.get("urgency", "all"),
        due=request.args.get("due", "all"), query=request.args.get("q", "").strip()[:150],
        page=request.args.get("page", "1")))


@app.get('/api/signals/<int:signal_id>')
def get_signal(signal_id):
    row = db().execute('SELECT * FROM signals WHERE id=?', (signal_id,)).fetchone()
    if row is None:
        return jsonify(error='Request not found.'), 404
    return jsonify(enrich_signal(row))


@app.get("/api/opportunities/evidence")
def opportunity_evidence():
    rows = [row for row in attach_groups(selected_rows()) if row.get("gkey") == request.args.get("key")]
    if not rows:
        return jsonify(error="This opportunity has no evidence in the current view."), 404
    summary = summarize(rows)["demand"][0]
    return jsonify(product=summary["product"], items=rows, summary=summary)


@app.post("/api/signals/preview")
def preview_signal():
    return jsonify(build_preview(request.get_json()))


@app.patch("/api/signals/<int:signal_id>")
def update_signal(signal_id):
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise ValueError("Use a JSON object to update a signal.")
    current = db().execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()
    if current is None:
        return jsonify(error="Signal not found."), 404
    values = dict(current)
    editable = {k: v for k, v in payload.items() if k in ("product", "qty", "budget", "urgency", "location",
                                                          "specifications", "kind")}
    apply_fields(values, editable)
    if "reason" in payload:
        if payload["reason"] not in REASON_LABELS:
            raise ValueError("Choose a supported reason label.")
        values["reviewed_reason"] = payload["reason"]
        values["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    for name, limit in (("owner", 100), ("next_action", 600)):
        if name in payload:
            if not isinstance(payload[name], str) or len(payload[name]) > limit:
                raise ValueError(f"'{name}' must be text of up to {limit} characters.")
            values[name] = payload[name].strip() or None
    if "status" in payload:
        if not isinstance(payload["status"], str) or payload["status"] not in STATUSES:
            raise ValueError("Choose a supported follow-up status.")
        values["status"] = payload["status"]
    if "follow_up_on" in payload:
        follow_up_on = payload["follow_up_on"]
        if follow_up_on not in (None, ""):
            try:
                if not isinstance(follow_up_on, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", follow_up_on):
                    raise ValueError()
                date.fromisoformat(follow_up_on)
            except ValueError as exc:
                raise ValueError("Choose a valid follow-up date (YYYY-MM-DD).") from exc
        values["follow_up_on"] = follow_up_on or None
    for name in ("currency", "recovered_currency"):
        if name in payload:
            if payload[name] not in CURRENCIES:
                raise ValueError("Choose a supported currency.")
            values[name] = payload[name]
    if "recovered_amount" in payload:
        amount = payload["recovered_amount"]
        if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or not 0 <= amount <= 1e15):
            raise ValueError("Recovered value must be a non-negative amount.")
        values["recovered_amount"] = amount
    if values["status"] == "recovered" and values.get("recovered_amount") is not None and not values.get("recovered_currency"):
        raise ValueError("Choose the currency for the recovered value.")
    values["pkey"] = normalize(values["product"])
    values["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    keys = ("product", "pkey", "qty", "budget", "urgency", "location", "currency", "reviewed_reason", "reviewed_at",
            "specifications", "kind", "status", "owner", "next_action", "follow_up_on", "recovered_amount",
            "recovered_currency", "updated_at")
    db().execute("UPDATE signals SET " + ",".join(f"{key}=?" for key in keys) + " WHERE id=?", (*(values[k] for k in keys), signal_id))
    db().commit()
    return jsonify(enrich_signal(values))


def import_items_from_upload(filename, content):
    name = (filename or "").casefold()
    fingerprint = hashlib.sha256(content.encode()).hexdigest()[:24]
    if name.endswith(".txt") or name.endswith(".text"):
        items = []
        for index, enquiry in enumerate(parse_conversation(content), 1):
            items.append(dict(text=enquiry["text"], source="salesperson", customer=enquiry.get("customer"),
                              external_id=f"chat-{fingerprint}-{index}"))
        return items, []
    try:
        reader = csv.DictReader(io.StringIO(content), strict=True)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("The CSV needs a 'text' column. Optional: source, customer, currency, external_id.")
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError("Use a valid UTF-8 CSV or a .txt conversation export.") from exc
    items, errors = [], []
    for index, row in enumerate(rows, 2):
        item = {key: value for key, value in row.items() if key in ("text", "source", "customer", "currency", "external_id") and value}
        item.setdefault("external_id", f"csv-{fingerprint}-{index}")
        items.append((index, item))
    previews, failed = [], errors
    for index, item in items:
        try:
            previews.append(item)
        except ValueError as exc:
            failed.append(dict(row=index, error=str(exc)))
    return [item for _, item in items], []


@app.post("/api/import/preview")
def preview_import():
    upload = request.files.get("file")
    if upload is None:
        raise ValueError("Choose a CSV or a .txt conversation record.")
    try:
        content = upload.read().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Use a valid UTF-8 CSV or conversation text file.") from exc
    try:
        items, parse_errors = import_items_from_upload(upload.filename, content)
    except ValueError:
        raise
    drafts, errors = [], list(parse_errors)
    if not items:
        raise ValueError("No customer enquiries were found in that file.")
    if len(items) > 100:
        raise ValueError("Import 1 to 100 requests at a time.")
    # CSV row one is the header; report the row a person sees in their file.
    first_row = 2 if (upload.filename or '').casefold().endswith('.csv') else 1
    for index, item in enumerate(items, first_row):
        try:
            drafts.append(build_preview(item))
        except ValueError as exc:
            errors.append(dict(row=index, error=str(exc)))
    return jsonify(drafts=drafts, errors=errors, total=len(drafts) + len(errors))


@app.get("/api/signals/export")
def export_signals():
    fields = ("id", "ts", "source", "product", "specifications", "kind", "qty", "budget", "currency", "effective_reason",
              "customer", "status", "owner", "recovered_amount", "recovered_currency", "raw")
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(fields)
    for row in selected_rows():
        writer.writerow([("'" + row.get(k)) if isinstance(row.get(k), str) and row[k].startswith(("=", "+", "-", "@")) else row.get(k) for k in fields])
    return Response(stream.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=ghostdemand-signals.csv"})



# ---------- PesaShield AI (M-Pesa fraud detection) ----------
FRAUD_SECRET = os.environ.get("PESASHIELD_SECRET", "dev-pesashield-secret").encode()

from pesashield import ANALYSIS_VERSION, analyse_message, expected_amount, normalized_text
from workspace_auth import privacy_ready


def private_fraud_history():
    return not is_demo() and privacy_ready() and bool(session.get('workspace_authenticated'))


SCAM_RULES = [
    (r"m-?pesa.*pin|pin.*m-?pesa", "pin_request", "Asking for M-Pesa PIN/OTP"),
    (r"send.*pin|confirm.*pin|verify.*pin", "pin_request", "Asking for PIN/verification"),
    (r"reversal.*send|send.*money.*reverse|reverse.*pin", "fake_reversal", "Fake reversal request"),
    (r"won|congratulations.*kes|prize.*claim", "prize_scam", "Prize/lottery scam"),
    (r"fuliza.*suspend|account.*block|blocked.*call", "impersonation", "Account suspension impersonation"),
    (r"till.*\d{5,}|paybill.*\d{5,}.*send", "payment_redirection", "Payment redirection request"),
    (r"http[s]?://\S+", "suspicious_link", "Contains external link"),
]


def _rule_scan_sms(text):
    low = (text or "").lower()
    codes = []
    details = []
    for pattern, code, desc in SCAM_RULES:
        if re.search(pattern, low):
            codes.append(code)
            details.append(desc)
    score = min(0.95, 0.3 + 0.15 * len(set(codes)))
    return list(set(codes)), details, score


def _hash_phone(value):
    if not value:
        return None
    return hmac.new(FRAUD_SECRET, str(value).strip().lower().encode(), hashlib.sha256).hexdigest()[:32]


def _hash_text(value):
    if not value:
        return None
    return hashlib.sha256(str(value).strip().lower().encode()).hexdigest()


@app.get("/api/pesashield/status")
def pesashield_status():
    if not PESASHIELD_ENABLED:
        return jsonify(enabled=False)
    return jsonify(enabled=True, model=active_fraud_model().status(), labels=SCAM_LABELS)


@app.post("/api/pesashield/sms/check")
def pesashield_sms_check():
    if not PESASHIELD_ENABLED:
        raise ValueError("PesaShield is disabled.")
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise ValueError("Send one message as a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("Paste a message between 1 and 10,000 characters.")
    text = normalized_text(text)
    if not text:
        raise ValueError("Paste a readable M-PESA message.")
    sender = payload.get("sender")
    if sender is not None and (not isinstance(sender, str) or len(sender) > 250):
        raise ValueError("Sender must be text of up to 250 characters.")
    channel = payload.get("channel", "sms")
    if channel not in ("sms", "whatsapp", "email", "manual"):
        raise ValueError("Choose sms, whatsapp, email or manual.")
    expected = expected_amount(payload.get("expected_amount"))
    prediction = active_fraud_model().predict(text)
    analysis = analyse_message(text, prediction=prediction, expected=expected)
    # Public demonstrations can check text without retaining financial messages.
    if not private_fraud_history():
        return jsonify(id=None, saved=False, deduplicated=False, **analysis)
    # Different comparison amounts are separate checks; identical retries count once.
    fingerprint = json.dumps([ANALYSIS_VERSION, " ".join(text.casefold().split()), expected])
    body_hash = _hash_text(fingerprint)
    sender_hash = _hash_phone(sender)
    connection = db()
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT id,analysis_json FROM fraud_messages WHERE body_hash=? AND channel=? AND ifnull(sender_hash,'')=ifnull(?, '') ORDER BY id DESC LIMIT 1",
                                 (body_hash, channel, sender_hash)).fetchone()
        if row and row['analysis_json']:
            return jsonify(id=row['id'], saved=True, deduplicated=True, **json.loads(row['analysis_json']))
        cursor = connection.execute("""INSERT INTO fraud_messages(ts,channel,sender_hash,body_hash,body,urls_json,amounts_json,mpesa_refs_json,
            risk_score,verdict,reason_codes_json,model_version,ml_confidence,reason_engine,analysis_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="microseconds"), channel, sender_hash, body_hash,
             text, json.dumps(analysis['urls']), json.dumps(analysis['amounts']), json.dumps(analysis['mpesa_refs']),
             None, analysis['verdict'], json.dumps(analysis['reason_codes']),
             prediction.get('model_version') if prediction else None, prediction.get('confidence') if prediction else None,
             analysis['reason_engine'], json.dumps(analysis)))
        mid = cursor.lastrowid
    return jsonify(id=mid, saved=True, deduplicated=False, **analysis)


@app.get("/api/pesashield/reports")
def pesashield_reports():
    if not PESASHIELD_ENABLED:
        raise ValueError("PesaShield is disabled.")
    if not private_fraud_history():
        return jsonify(items=[], history_available=False)
    rows = [dict(r) for r in db().execute("SELECT id,ts,channel,sender_hash,body,urls_json,amounts_json,mpesa_refs_json,risk_score,verdict,reason_codes_json,model_version,ml_confidence,reason_engine,reviewed_verdict,reviewed_at,status,analysis_json FROM fraud_messages ORDER BY ts DESC, id DESC LIMIT 200")]
    for r in rows:
        saved = json.loads(r.pop("analysis_json") or "{}")
        r["assessment_label"] = saved.get("verdict_label", "Older assessment - recheck")
        r["risk_level"] = saved.get("risk_level", "unknown")
        r["reason_titles"] = saved.get("rule_details", [])
        r["urls"] = json.loads(r["urls_json"] or "[]")
        r["amounts"] = json.loads(r["amounts_json"] or "[]")
        r["mpesa_refs"] = json.loads(r["mpesa_refs_json"] or "[]")
        r["reason_codes"] = json.loads(r["reason_codes_json"] or "[]")
        r["body_preview"] = (r["body"] or "")[:200]
    return jsonify(items=rows, history_available=True)


@app.patch("/api/pesashield/reports/<int:report_id>/review")
def pesashield_review(report_id):
    if not PESASHIELD_ENABLED:
        raise ValueError("PesaShield is disabled.")
    if not private_fraud_history():
        return jsonify(error='Sign in to a private live workspace to review saved checks.'), 403
    payload = request.get_json() or {}
    verdict = payload.get("verdict")
    if verdict not in SCAM_LABELS:
        raise ValueError("Choose scam, suspicious, or legit.")
    cursor = db().execute("UPDATE fraud_messages SET reviewed_verdict=?, reviewed_at=? WHERE id=?", (verdict, datetime.now(timezone.utc).isoformat(timespec="microseconds"), report_id))
    if cursor.rowcount == 0:
        return jsonify(error="Report not found."), 404
    db().commit()
    return jsonify(id=report_id, reviewed_verdict=verdict)


@app.get("/api/pesashield/model")
def pesashield_model():
    if not PESASHIELD_ENABLED:
        raise ValueError("PesaShield is disabled.")
    return jsonify(active_fraud_model().status())


@app.post("/api/pesashield/model/train")
def pesashield_train():
    if not PESASHIELD_ENABLED:
        raise ValueError("PesaShield is disabled.")
    data = []
    try:
        starter = json.loads(FRAUD_STARTER_PATH.read_text()) if FRAUD_STARTER_PATH.exists() else []
    except Exception:
        starter = []
    for s in starter:
        if isinstance(s, dict) and s.get("text") and s.get("label") in SCAM_LABELS:
            data.append(dict(text=s["text"], label=s["label"]))
    for row in db().execute("SELECT body, reviewed_verdict FROM fraud_messages WHERE reviewed_verdict IS NOT NULL"):
        if row["body"] and row["reviewed_verdict"] in SCAM_LABELS:
            data.append(dict(text=row["body"], label=row["reviewed_verdict"]))
    try:
        meta = active_fraud_model().train(data)
    except FraudTrainingError as e:
        return jsonify(error=str(e)), 409
    except Exception as e:
        return jsonify(error=str(e)), 503
    return jsonify(model_status=active_fraud_model().status(), metadata=meta)


@app.get("/api/pesashield/insights")
def pesashield_insights():
    if not PESASHIELD_ENABLED:
        raise ValueError("PesaShield is disabled.")
    if not private_fraud_history():
        return jsonify(kpis=dict(total=0, scam=0, suspicious=0, legit=0, unknown=0, unverified=0, reviewed=0),
                       recent=[], labels=SCAM_LABELS, history_available=False)
    db_conn = db()
    total = db_conn.execute("SELECT COUNT(*) as c FROM fraud_messages").fetchone()["c"]
    scam = db_conn.execute("SELECT COUNT(*) as c FROM fraud_messages WHERE verdict='scam'").fetchone()["c"]
    suspicious = db_conn.execute("SELECT COUNT(*) as c FROM fraud_messages WHERE verdict='suspicious'").fetchone()["c"]
    reviewed = db_conn.execute("SELECT COUNT(*) as c FROM fraud_messages WHERE reviewed_verdict IS NOT NULL").fetchone()["c"]
    recent = [dict(r) for r in db_conn.execute("SELECT id,ts,verdict,risk_score,reason_codes_json FROM fraud_messages ORDER BY ts DESC LIMIT 10")]
    for r in recent:
        r["reason_codes"] = json.loads(r["reason_codes_json"] or "[]")
    legit = db_conn.execute("SELECT COUNT(*) FROM fraud_messages WHERE verdict='legit'").fetchone()[0]
    unknown = db_conn.execute("SELECT COUNT(*) FROM fraud_messages WHERE verdict='unknown'").fetchone()[0]
    return jsonify(kpis=dict(total=total, scam=scam, suspicious=suspicious, legit=legit, unknown=unknown, unverified=suspicious+unknown, reviewed=reviewed),
                   recent=recent, labels=SCAM_LABELS)



@app.get("/api/catalog")
def get_catalog():
    rows = [dict(row) for row in db().execute("SELECT * FROM catalog ORDER BY name")]
    for row in rows:
        row["aliases"] = [r[0] for r in db().execute("SELECT alias_key FROM product_aliases WHERE catalog_id=? ORDER BY alias_key", (row["id"],))]
    return jsonify(items=rows)


@app.post("/api/catalog")
def add_catalog():
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise ValueError("Use a JSON object for a product.")
    name, aliases, sku = payload.get("name"), payload.get("aliases", []), payload.get("sku", "")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200 or not product_key(name):
        raise ValueError("Enter a product name of 1 to 200 characters.")
    if not isinstance(aliases, list) or len(aliases) > 20 or any(not isinstance(a, str) or not a.strip() or len(a) > 200 for a in aliases):
        raise ValueError("Use up to 20 non-empty product aliases.")
    if not isinstance(sku, str) or len(sku) > 80:
        raise ValueError("SKU must contain up to 80 characters.")
    keys = {product_key(a) for a in [name, *aliases]}
    if "" in keys:
        raise ValueError("Aliases must include letters or numbers.")
    connection = db()
    try:
        with connection:
            cursor = connection.execute("INSERT INTO catalog(name,sku,created_at) VALUES(?,?,?)", (name.strip(), sku.strip(), datetime.now(timezone.utc).isoformat()))
            connection.executemany("INSERT INTO product_aliases(alias_key,catalog_id) VALUES(?,?)", [(key, cursor.lastrowid) for key in keys])
            for row in connection.execute("SELECT id,product FROM signals").fetchall():
                if product_key(row["product"]) in keys:
                    connection.execute("UPDATE signals SET product=?,pkey=? WHERE id=?", (name.strip(), normalize(name), row["id"]))
    except sqlite3.IntegrityError as exc:
        raise ValueError("A name or alias already belongs to another catalog product. Keep each alias unique.") from exc
    return jsonify(id=cursor.lastrowid, name=name.strip(), aliases=sorted(keys)), 201


@app.post("/api/events/website-search")
def website_event():
    payload = request.get_json()
    if not isinstance(payload, dict) or type(payload.get("results_count")) is not int or payload["results_count"] < 0:
        raise ValueError("Provide a non-negative integer 'results_count'.")
    if payload["results_count"] != 0:
        return jsonify(captured=False, message="Search returned results; no failed-search signal captured.")
    item = dict(text=payload.get("query"), source="website_search", customer=payload.get("customer"),
                external_id=payload.get("event_id"), currency=payload.get("currency", DEFAULT_CURRENCY))
    result = save_payload(item)
    return jsonify(captured=True, signal=result), 201


@app.post("/api/events/stockout")
def stockout_event():
    payload = request.get_json()
    if not isinstance(payload, dict) or not isinstance(payload.get("product"), str):
        raise ValueError("Provide a product name.")
    fields = dict(product=payload["product"], qty=payload.get("qty", 1), reason="out_of_stock")
    apply_fields(dict(reason="out_of_stock"), fields)
    item = dict(text=f"Out of stock: {fields['qty']} units {fields['product']}", source="out_of_stock",
                customer=payload.get("customer"), external_id=payload.get("event_id"), currency=payload.get("currency", DEFAULT_CURRENCY))
    text, source, customer = prepare_signal(item)
    currency, external_id = input_options(item)
    return jsonify(save_signal(text, source, customer, currency=currency, external_id=external_id, fields=fields, event=True)), 201


@app.post("/api/demo/seed")
def seed_demo():
    if not is_demo():
        raise ValueError("Demo data can only be loaded in the demo workspace.")
    from demo_data import seed
    if not active_model().status()["ready"]:
        active_model().train([])
    return jsonify(seed(db(), save_signal))


@app.cli.command("train-model")
def train_model_command():
    """Train on synthetic starter examples and explicitly reviewed signals."""
    try:
        metadata = active_model().train(reviewed_examples())
    except (TrainingError, OSError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(metadata, indent=2))


def bootstrap_models():
    """Initialize missing baseline artifacts only; never overwrite a saved model."""
    if os.environ.get('GHOST_BOOTSTRAP_MODELS', '1') == '0':
        return
    for classifier in (ML, DEMO_ML):
        if classifier.enabled and not classifier.path.exists():
            classifier.train([])


from flask import send_from_directory, request, jsonify


@app.route("/terms.html")
def terms():   return send_from_directory("static", "terms.html")


@app.route("/privacy.html")
def privacy(): return send_from_directory("static", "privacy.html")


@app.route("/consent.html")
def consent(): return send_from_directory("static", "consent.html")


@app.route("/api/consent", methods=["POST"])  # optional: store consent records for proof
def record_consent():
    data = request.get_json(silent=True) or {}
    # TODO: save data + request.remote_addr + user agent to your DB
    return jsonify(ok=True)

from intake import register_intake
receive_intake = register_intake(app, db=db, save_signal=save_signal, is_demo=is_demo, default_currency=DEFAULT_CURRENCY)
from workspace_auth import register_workspace_auth
register_workspace_auth(app, is_demo)
from live_channels import register_live_channels
start_channel_listeners = register_live_channels(app, db=db, receive=receive_intake, is_demo=is_demo)


if __name__ == "__main__":
    bootstrap_models()
    start_channel_listeners()
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", "5000")))
