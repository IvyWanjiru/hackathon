"""Explainable checks for pasted M-PESA text, not payment authentication."""
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

ANALYSIS_VERSION = 'mpesa-text-v2'
VERDICTS = {
    'scam': ('High fraud risk', 'high'),
    'suspicious': ('Suspicious - verify first', 'medium'),
    'legit': ('Looks legitimate', 'low'),
    'unknown': ('Unable to determine', 'unknown'),
}
VERIFICATION_NOTE = ('This is a text assessment, not proof of payment. Confirm the transaction in your '
                     'own M-PESA app, statement or business records before releasing goods or returning money.')
GUIDANCE_URL = 'https://www.safaricom.co.ke/images/Downloads/Buy-goods-guides.pdf'
MONEY_VALUE = r'(?P<value>\d+(?:,\d{3})*(?:\.\d{1,2})?)(?![\d,]|\.\d)'
MONEY = re.compile(r'\b(?:kshs?|kes)\.?\s*' + MONEY_VALUE, re.I)
REFERENCE = re.compile(r'\b([A-Z0-9]{10,12})\s+confirmed\b', re.I)
NEGATION = re.compile(r"\b(?:do\s+not|don['’]t|never|not\s+to|avoid|should\s+not|must\s+not)\s+(?:ever\s+)?$", re.I)


def normalized_text(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if unicodedata.category(c) != 'Cf').strip()


def expected_amount(value):
    if value in (None, ''):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('Expected amount must be a positive KES amount with up to two decimal places.')
    try:
        amount = Decimal(str(value).replace(',', ''))
        if not amount.is_finite() or not Decimal('0') < amount <= Decimal('1000000000000') or amount.as_tuple().exponent < -2:
            raise ValueError
    except (InvalidOperation, ValueError):
        raise ValueError('Expected amount must be a positive KES amount with up to two decimal places.') from None
    return float(amount)


def _money(match):
    if not match:
        return None
    value = Decimal(match.group('value').replace(',', ''))
    return float(value) if value.is_finite() and value <= Decimal('1000000000000') else None


def parse_receipt(text):
    reference_match = REFERENCE.search(text)
    reference = reference_match.group(1).upper() if reference_match else None
    if reference and not (re.search(r'[A-Z]', reference) and re.search(r'\d', reference)):
        reference = None
    types = ((r'\byou\s+have\s+received\b|\breceived\s+(?:kshs?|kes)', 'Money received'),
             (r'\bsent\s+to\b', 'Money sent'), (r'\bpaid\s+to\b', 'Merchant payment'),
             (r'\bwithdrawn?\b|\bwithdraw\b', 'Cash withdrawal'),
             (r'\b(?:bought|purchased)\b.{0,35}\bairtime\b', 'Airtime purchase'),
             (r'\b(?:deposited|deposit)\b', 'Cash deposit'))
    transaction_type = next((label for pattern, label in types if re.search(pattern, text, re.I)), None)
    transaction_text = re.split(r'\b(?:new\s+)?m[\s-]?pesa\s+balance\b', text, maxsplit=1, flags=re.I)[0]
    amount = _money(MONEY.search(transaction_text))
    balance_match = re.search(r'\b(?:new\s+)?m[\s-]?pesa\s+balance\s*(?:is|:)\s*(?:kshs?|kes)\.?\s*' + MONEY_VALUE, text, re.I)
    balance = _money(balance_match)
    date_match = re.search(r'\bon\s+(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))\s+at\s+(\d{1,2}:\d{2})\s*(AM|PM)?\b', text, re.I)
    timestamp, date_invalid = None, False
    if date_match:
        date, time, suffix = date_match.groups()
        fmt = '%d/%m/%Y' if len(date.rsplit('/', 1)[1]) == 4 else '%d/%m/%y'
        try:
            stamp = datetime.strptime(f'{date} {time}' + (f' {suffix.upper()}' if suffix else ''),
                                      fmt + (' %I:%M %p' if suffix else ' %H:%M'))
            timestamp = stamp.strftime('%d %b %Y, %H:%M') + ' (as written)'
        except ValueError:
            date_invalid = True
    party = re.search(r'\b(?:from|sent\s+to|paid\s+to)\s+(.{1,110}?)\s+on\s+\d{1,2}/', text, re.I)
    party_text = ' '.join(party.group(1).split()) if party else None
    checks = []
    for passed, label in ((reference, 'Transaction reference and confirmation wording found.'),
                          (amount is not None and amount > 0, 'Transaction amount found.'),
                          (transaction_type, 'A recognised transaction description was found.'),
                          (timestamp, 'Transaction date and time can be read.'),
                          (balance is not None, 'A numeric M-PESA balance was found.')):
        if passed: checks.append(label)
    complete = len(checks) == 5
    return dict(reference=reference, amount=amount, currency='KES' if amount is not None else None,
                transaction_type=transaction_type, counterparty=party_text, timestamp=timestamp,
                balance=balance, format_complete=complete, checks=checks, invalid_date=date_invalid)


def _affirmative_matches(text, pattern):
    for clause in re.split(r'(?<=[!?;])\s*|\n+|(?<!\d)\.(?!\d)\s*|\bbut\b', text, flags=re.I):
        for match in re.finditer('(?=(' + pattern + '))', clause, re.I):
            if not NEGATION.search(clause[max(0, match.start()-40):match.start()]):
                yield match.group(1)


def analyse_message(text, *, prediction=None, expected=None):
    clean = normalized_text(text)
    receipt = parse_receipt(clean)
    flags = []

    def flag(code, severity, title, detail):
        if not any(item['code'] == code for item in flags):
            flags.append(dict(code=code, severity=severity, title=title, detail=detail))

    credentials = r'\b(?:send|share|tell|give|reply|forward|disclose|provide|tuma|nitumie)\b.{0,65}?\b(?:pin|otp|password|one[ -]time\s+(?:code|password)|verification\s+code|security\s+code)\b'
    if any(_affirmative_matches(clean, credentials)):
        flag('pin_request', 'high', 'Request to disclose a PIN or code',
             'The message asks you to send or share a secret. A safety reminder saying never to share it is treated differently.')
    urls = re.findall(r'(?i)\b(?:https?://|www\.)[^\s<>"\']+', clean)
    external_urls = []
    for url in urls:
        try:
            host = (urlsplit(url if '://' in url else 'https://' + url).hostname or '').lower()
        except ValueError:
            host = ''
        if not any(host == domain or host.endswith('.' + domain) for domain in ('safaricom.co.ke', 'safaricom.com')):
            external_urls.append(url)
    if external_urls:
        flag('external_link', 'medium', 'External link needs checking',
             'A link outside the recognised Safaricom domains is present. Do not use a message link to verify your account.')
        if any(_affirmative_matches(clean, r'\b(?:enter|confirm|verify)\b.{0,45}\b(?:pin|otp|password|verification\s+code)\b')):
            flag('credential_link', 'high', 'Credentials requested through a link',
                 'The message combines an external link with a request for a PIN, password or code.')
    money_action = r'\b(?:send|refund|return|transfer|pay)\b.{0,65}?(?:\b(?:money|cash|funds|shillings|pesa)\b|\b(?:kshs?|kes)(?=\s|\d|[.,]))'
    actions = list(_affirmative_matches(clean, money_action))
    if actions and re.search(r'\b(?:reversal|reverse|by\s+mistake|wrong\s+(?:number|account)|accidentally|mistakenly)\b', clean, re.I):
        flag('refund_request', 'high', 'Direct refund or reversal request',
             'Someone asks you to send money in response to an alleged mistake or reversal. Verify independently and use the provider\'s reversal process.')
    if REFERENCE.search(clean) and re.search(r'\bbalance\s*(?:is|:)?\s*(?:locked|blocked)\b', clean, re.I):
        flag('locked_balance', 'high', 'Balance shown as locked or blocked',
             'The message substitutes a locked or blocked balance for a numeric balance. Do not rely on it as proof of payment.')
    caution_context = ' '.join(clause for clause in re.split(r'[!?;\n]|(?<!\d)\.(?!\d)|\bbut\b', clean, flags=re.I)
                               if not re.search(r"\b(?:never|do\s+not|don['’]t|avoid)\s+(?:share|send|pay|give|disclose|click|open)\b", clause, re.I))
    prize = re.search(r'\b(?:won|winner|prize|lottery|congratulations)\b', caution_context, re.I)
    fee_actions = list(_affirmative_matches(clean, r'\b(?:pay|send)\b.{0,40}\b(?:fee|charge|deposit)\b'))
    if prize and (actions or fee_actions):
        flag('prize_fee', 'high', 'Payment requested to claim a prize', 'The message combines a prize claim with a request to send money or pay a fee.')
    elif prize and re.search(r'\bclaim\b', clean, re.I):
        flag('prize_claim', 'medium', 'Unverified prize claim', 'Confirm the promotion independently before responding or providing details.')
    if re.search(r'\b(?:loan|fuliza)\b', clean, re.I) and (actions or fee_actions) and re.search(r'\b(?:approval|processing|registration|unlock|activate|release|fee)\b', clean, re.I):
        flag('upfront_fee', 'high', 'Upfront payment to release a loan', 'The message asks for a payment linked to loan approval, activation or release.')
    if re.search(r'\b(?:account|fuliza|m[ -]?pesa)\b.{0,60}\b(?:blocked|suspended|deactivated)\b', caution_context, re.I) and re.search(r'\b(?:call|reply|verify|send|click)\b', caution_context, re.I):
        flag('account_threat', 'medium', 'Account threat with an instruction', 'An account suspension threat is being used to prompt contact or action.')
    if re.search(r'\b(?:new|changed|different|instead|ignore)\b.{0,45}\b(?:till|paybill|payment\s+number)\b|\b(?:till|paybill)\b.{0,45}\b(?:changed|instead)\b', clean, re.I):
        flag('payment_redirection', 'medium', 'Changed payment destination', 'The message asks you to use a changed or different payment destination. Check it against your trusted business details.')
    if receipt['invalid_date']:
        flag('invalid_date', 'medium', 'Receipt date or time is invalid', 'The date or clock time in the message is not a valid calendar timestamp.')
    references = {match.group(1).upper() for match in REFERENCE.finditer(clean)}
    if len(references) > 1:
        flag('multiple_receipts', 'medium', 'Several receipt references found', 'Paste one complete message at a time to avoid mixing transactions.')
    if expected is not None and receipt['amount'] is not None and Decimal(str(expected)) != Decimal(str(receipt['amount'])):
        flag('amount_mismatch', 'medium', 'Amount differs from your expected amount',
             f'The message states KES {receipt["amount"]:,.2f}; you expected KES {expected:,.2f}.')

    if any(item['severity'] == 'high' for item in flags):
        verdict, summary = 'scam', 'Strong fraud indicators were found in this message. Do not act on its instructions.'
    elif flags:
        verdict, summary = 'suspicious', 'This message needs an independent check before you act on it.'
    elif receipt['format_complete']:
        verdict, summary = 'legit', 'The message follows a recognised receipt structure and no listed scam indicators were found. It may still be copied or forged.'
    else:
        verdict, summary = 'unknown', 'There is not enough receipt information or clear fraud evidence to classify this message reliably.'

    # The tiny optional starter model is supporting evidence, never payment proof.
    if (verdict == 'suspicious' and prediction and prediction.get('verdict') in ('scam', 'suspicious')
            and prediction.get('confidence', 0) >= 0.85):
        flag('model_caution', 'supporting', 'The text classifier also suggests caution',
             'This supporting suggestion is not a verified fraud probability.')
    advice = ['Open your usual M-PESA app or statement yourself and compare the reference, amount and recipient.']
    if verdict == 'scam':
        advice.insert(0, 'Do not send money, disclose a PIN or OTP, or open links in this message.')
    elif verdict in ('suspicious', 'unknown'):
        advice.insert(0, 'Pause the transaction and check the original message through a trusted channel.')
    else:
        advice.insert(0, 'Check your own transaction records before treating this as a received payment.')
    if any(item['code'] == 'refund_request' for item in flags):
        advice.append('Use Safaricom\'s official support or reversal process rather than sending a separate refund.')
    label, level = VERDICTS[verdict]
    return dict(verdict=verdict, verdict_label=label, risk_level=level, summary=summary,
                reasons=flags, reason_codes=[item['code'] for item in flags],
                rule_details=[item['title'] for item in flags], receipt=receipt, urls=urls,
                amounts=[match.group(0) for match in MONEY.finditer(clean)],
                mpesa_refs=sorted(references), expected_amount=expected, advice=advice,
                verification=dict(status='not_verified', note=VERIFICATION_NOTE),
                guidance_url=GUIDANCE_URL, analysis_version=ANALYSIS_VERSION,
                reason_engine='rules+local_ml' if any(item['code'] == 'model_caution' for item in flags) else 'rules',
                risk_score=None, confidence=None)
