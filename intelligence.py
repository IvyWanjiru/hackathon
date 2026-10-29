"""Evidence-based opportunity summaries. Suggestions are explicit business rules."""
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

SOURCE_NAMES = {
    "whatsapp": "WhatsApp", "website_search": "Website search", "salesperson": "Sales notes",
    "salesperson_voice": "Voice note", "chatbot": "Chatbot", "rejected_quote": "Unaccepted quote",
    "unavailable_product": "Unavailable product", "out_of_stock": "Stockout", "abandoned_enquiry": "Abandoned enquiry",
}
REASON_NAMES = {
    "out_of_stock": "Out of stock", "price_too_high": "Price concern", "not_stocked": "Product not offered",
    "unsupported_service": "Unsupported service", "quote_rejected": "Quote not accepted",
    "lead_time": "Delivery too slow", "abandoned": "Enquiry abandoned", "unknown": "Unknown outcome",
}
STATUSES = {"open": "Open", "follow_up": "Following up", "recovered": "Recovered", "closed": "Closed"}
CURRENCIES = ("KES", "USD", "EUR", "GBP", "UGX", "TZS", "UNKNOWN")
ACTIONS = {
    "out_of_stock": ("Restock or offer a substitute", "Check supplier availability and confirm a restock date with interested customers."),
    "not_stocked": ("Validate a new product opportunity", "Confirm specifications with interested customers before adding a product to the range."),
    "price_too_high": ("Review the price or bundle", "Compare stated budgets with your offer; test an alternative or bundle without assuming a discount is profitable."),
    "quote_rejected": ("Review and follow up on the quote", "Ask what prevented acceptance and check whether a revised offer would help."),
    "lead_time": ("Offer a faster delivery option", "Check a closer supplier or collection option, then confirm the customer's deadline."),
    "abandoned": ("Follow up on unfinished enquiries", "Prepare a relevant follow-up for customers who agreed to be contacted."),
    "unsupported_service": ("Review whether the service can be offered",
                            "Confirm the requested service and decide whether to add it, refer the customer, or record it as out of scope."),
    "unknown": ("Ask what blocked the request",
                "The outcome was not stated. Confirm with the salesperson or original conversation before treating this as a product gap."),
}

KINDS = {"product": "Product", "service": "Service"}
SPEC_PATTERN = re.compile(
    r"size\s*\d+[a-z]?|\b\d+(?:\.\d+)?\s*(?:gb|tb|mb|mm|cm|inch(?:es)?|in|ml|litres?|liters?|kg|g|w|mah|hz|v)\b|\b(?:xxl|xl|xs)\b",
    re.I)
SERVICE_PATTERN = re.compile(
    r"\b(install(?:ation)?|repair(?:s|ing)?|servicing|assembly|fitting|home visit|consultation|maintenance|delivery service)\b",
    re.I)
PHRASE_SYNONYMS = (
    (re.compile(r"\bear\s*pods?\b", re.I), "earbuds"),
    (re.compile(r"\bearphones?\b", re.I), "earbuds"),
    (re.compile(r"\bearbuds?\b", re.I), "earbuds"),
    (re.compile(r"\btype\s*-?c\b", re.I), "usbc"),
    (re.compile(r"\busb\s*-?c\b", re.I), "usbc"),
    (re.compile(r"\badaptors?\b", re.I), "charger"),
    (re.compile(r"\badapters?\b", re.I), "charger"),
    (re.compile(r"\bchargers?\b", re.I), "charger"),
    (re.compile(r"\brunning\s+shoes?\b", re.I), "runningshoes"),
    (re.compile(r"\btrainers?\b", re.I), "runningshoes"),
    (re.compile(r"\bsneakers?\b", re.I), "runningshoes"),
    (re.compile(r"\brucksacks?\b", re.I), "backpack"),
    (re.compile(r"\bbackpacks?\b", re.I), "backpack"),
    (re.compile(r"\blaptop\s+bags?\b", re.I), "backpack"),
)
GENERIC_TOKENS = {
    "wireless", "bluetooth", "portable", "everyday", "work", "laptop", "fast", "insulated",
    "thermal", "original", "genuine", "branded", "new", "the", "a", "an", "for", "and", "of",
}
STAFF_SPEAKERS = {"staff", "agent", "seller", "shop", "store", "bot", "assistant", "you", "admin", "system"}
CHAT_LINE = re.compile(
    r"^\s*(?:\[\d[^\]]{5,40}\]|\d{1,4}[./-]\d{1,2}[./-]\d{1,4},?\s+\d{1,2}:\d{2}(?:\s*:\d{2})?(?:\s*[ap]m)?)\s*"
    r"(?:[-–]\s*)?(?P<name>[^:]{1,80}):\s*(?P<text>.+)$",
    re.I)
NAMED_LINE = re.compile(
    r"^\s*(?P<name>customer|client|buyer|shopper|user|visitor|staff|agent|seller|shop|store|bot|assistant|you)\s*[:\-]\s*(?P<text>.+)$",
    re.I)


def product_key(value):
    # Preserve dimensions, brands, sizes and model numbers; no automatic stemming.
    return " ".join(re.findall(r"[\w]+", (value or "").casefold(), re.UNICODE))


def extract_specifications(text):
    found, seen = [], set()
    for match in SPEC_PATTERN.finditer(text or ""):
        value = " ".join(match.group(0).casefold().split())
        if value not in seen:
            seen.add(value)
            found.append(value)
    return ", ".join(found)


def request_kind(text):
    return "service" if SERVICE_PATTERN.search(text or "") else "product"


def synonym_text(value):
    text = value or ""
    for pattern, replacement in PHRASE_SYNONYMS:
        text = pattern.sub(replacement, text)
    return text


def demand_signature(product, specifications=""):
    spec_tokens = set(product_key(specifications).split())
    spec_tokens.update(product_key(extract_specifications(product)).split())
    tokens = [token for token in product_key(synonym_text(product)).split()
              if token not in GENERIC_TOKENS and token not in spec_tokens]
    return tokens, " ".join(sorted(spec_tokens))


def similar_demand(left, right):
    left_tokens, left_specs = left
    right_tokens, right_specs = right
    if left_specs and right_specs and left_specs != right_specs:
        return False
    if not left_tokens or not right_tokens:
        return False
    left_set, right_set = set(left_tokens), set(right_tokens)
    if left_set == right_set:
        return True
    union = left_set | right_set
    if len(left_set & right_set) / len(union) >= 0.5:
        return True
    smaller, larger = (left_set, right_set) if len(left_set) <= len(right_set) else (right_set, left_set)
    return smaller < larger and left_tokens[-1] == right_tokens[-1]


def attach_groups(rows):
    """Cluster similar products/services; keep each row's original product wording."""
    rows = [dict(row) for row in rows]
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    signatures = [demand_signature(row.get("product"), row.get("specifications")) for row in rows]
    for i, left in enumerate(signatures):
        for j in range(i + 1, len(signatures)):
            if similar_demand(left, signatures[j]):
                parent[find(i)] = find(j)
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(i)
    for indexes in members.values():
        token_keys = [" ".join(signatures[i][0]) or product_key(rows[i].get("product")) for i in indexes]
        key = Counter(token_keys).most_common(1)[0][0]
        specs = {signatures[i][1] for i in indexes if signatures[i][1]}
        if len(specs) == 1:
            key = f"{key} | {next(iter(specs))}"
        names = Counter(rows[i]["product"] for i in indexes)
        title = names.most_common(1)[0][0]
        for i in indexes:
            rows[i]["gkey"] = key
            rows[i]["group_title"] = title
    return rows


def parse_conversation(text):
    """Turn a pasted/uploaded transcript into customer enquiry texts."""
    if not isinstance(text, str) or not text.strip():
        return []
    entries, current = [], None
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = CHAT_LINE.match(line) or NAMED_LINE.match(line)
        if match:
            if current:
                entries.append(current)
            current = dict(name=match.group("name").strip(), text=match.group("text").strip())
        elif current:
            current["text"] = f"{current['text']} {line}".strip()
    if current:
        entries.append(current)
    skip = re.compile(r"media omitted|this message was deleted|messages and calls are end-to-end", re.I)
    enquiries = []
    for entry in entries:
        speaker = entry["name"].casefold()
        body = " ".join(entry["text"].split())
        if speaker in STAFF_SPEAKERS or skip.search(body) or len(body) < 12:
            continue
        enquiries.append(dict(text=body[:10000], customer=None if speaker in ("customer", "client", "buyer", "shopper", "user", "visitor") else entry["name"][:250]))
    if enquiries:
        return enquiries[:100]
    blob = " ".join(text.split())
    return [dict(text=blob[:10000], customer=None)] if blob else []


def detect_currency(text, default="KES"):
    patterns = {"KES": r"\b(?:kes|ksh|kshs)\b", "USD": r"\busd\b|\$", "EUR": r"\beur\b|€",
                "GBP": r"\bgbp\b|£", "UGX": r"\bugx\b", "TZS": r"\btzs\b"}
    found = [code for code, pattern in patterns.items() if re.search(pattern, text, re.I)]
    return found[0] if len(found) == 1 else "UNKNOWN" if len(found) > 1 else default


def money_totals(rows, amount="budget", currency="currency"):
    totals = defaultdict(float)
    for row in rows:
        value = row.get(amount)
        if value is not None:
            totals[row.get(currency) or "UNKNOWN"] += value
    return {code: round(value, 2) for code, value in sorted(totals.items())}


def parse_timestamp(value):
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def enrich_signal(row):
    row = dict(row)
    # Older databases used a stemmed/sorted key; group by the current product
    # name without rewriting original records just to read the dashboard.
    row["pkey"] = product_key(row["product"])
    row["effective_reason"] = row.get("reviewed_reason") or row["reason"]
    row["needs_review"] = not bool(row.get("reviewed_reason"))
    row["status"] = row.get("status") or "open"
    row["specifications"] = row.get("specifications") or None
    row["kind"] = row.get("kind") if row.get("kind") in KINDS else "product"
    return row


def follow_up_queue(rows, *, today, status="active", assignment="any", owner="",
                    urgency="all", due="all", query="", page=1):
    """A business work queue, independent of dashboard filters or ML status."""
    if status not in ("active", "all", *STATUSES):
        raise ValueError("Choose a supported follow-up status.")
    if assignment not in ("any", "assigned", "unassigned"):
        raise ValueError("Choose a supported owner filter.")
    if urgency not in ("all", "urgent"):
        raise ValueError("Choose all requests or urgent requests.")
    if due not in ("all", "overdue", "today", "unscheduled"):
        raise ValueError("Choose a supported follow-up date filter.")
    try:
        page = max(1, int(page))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid page number.") from exc
    day = today.isoformat()
    active = [row for row in rows if row["status"] in ("open", "follow_up")]
    recovered = [row for row in rows if row["status"] == "recovered"]
    is_overdue = lambda row: row["status"] in ("open", "follow_up") and bool(row.get("follow_up_on")) and row["follow_up_on"] < day
    summary = dict(active=len(active), unassigned=sum(not row.get("owner") for row in active),
                   following_up=sum(row["status"] == "follow_up" for row in active),
                   overdue=sum(is_overdue(row) for row in active), recovered=len(recovered),
                   recovered_value=money_totals(recovered, "recovered_amount", "recovered_currency"),
                   needs_review=sum(row["needs_review"] for row in rows))
    owners = {}
    for row in rows:
        if row.get("owner"):
            owners.setdefault(row["owner"].casefold(), row["owner"])
    found = []
    for original in rows:
        row = dict(original)
        if status == "active" and row["status"] not in ("open", "follow_up"):
            continue
        if status not in ("active", "all") and row["status"] != status:
            continue
        if assignment == "unassigned" and row.get("owner") or assignment == "assigned" and not row.get("owner"):
            continue
        if owner and (row.get("owner") or "").casefold() != owner.casefold():
            continue
        if urgency == "urgent" and row["urgency"] != 3:
            continue
        if due == "overdue" and not is_overdue(row):
            continue
        if due == "today" and (row.get("follow_up_on") != day or row["status"] not in ("open", "follow_up")):
            continue
        if due == "unscheduled" and row.get("follow_up_on"):
            continue
        haystack = " ".join(str(row.get(key) or "") for key in ("product", "raw", "customer", "owner", "next_action"))
        if query and query.casefold() not in haystack.casefold():
            continue
        row["overdue"] = is_overdue(row)
        row["due_today"] = row.get("follow_up_on") == day and row["status"] in ("open", "follow_up")
        row["suggested_action"] = ACTIONS.get(row["effective_reason"], ("Review the request", ""))[0]
        found.append(row)
    found.sort(key=lambda row: (not row["overdue"], -row["urgency"], row.get("follow_up_on") or "9999-12-31", row["ts"], row["id"]))
    pages = max(1, (len(found) + 11) // 12)
    page = min(page, pages)
    return dict(items=found[(page-1)*12:page*12], total=len(found), page=page, pages=pages,
                summary=summary, owners=sorted(owners.values(), key=str.casefold), today=day)


def summarize(rows, now=None):
    now = now or datetime.now(timezone.utc)
    rows = attach_groups(rows)
    groups = {}
    for row in rows:
        key = row.get("gkey") or row["pkey"] or product_key(row["product"])
        group = groups.setdefault(key, {"key": key, "product": row.get("group_title") or row["product"], "rows": []})
        group["rows"].append(row)
    demand = []
    for group in groups.values():
        evidence = group.pop("rows")
        active = [r for r in evidence if r["status"] in ("open", "follow_up")]
        reason_evidence = active or evidence
        reasons = Counter(r["effective_reason"] for r in reason_evidence)
        sources = Counter(r["source"] for r in evidence)
        known = {r["customer"].strip().casefold() for r in evidence if r.get("customer")}
        urgent = sum(r["urgency"] == 3 for r in active)
        # Comparable score: no currency arithmetic, inferred buyers or forecast.
        score = len(active) * 2 + urgent * 3 + len(known) + len(sources)
        top_reason = reasons.most_common(1)[0][0]
        action, explanation = ACTIONS[top_reason]
        if top_reason in ("not_stocked", "unknown") and "website_search" in sources:
            action = "Check search coverage and product range"
            explanation = ("Replay the failed query and check filters and product aliases. "
                           "Confirm whether the product is actually missing before adding it to the range.")
        recent = sum(parse_timestamp(r["ts"]) >= now - timedelta(days=7) for r in evidence)
        previous = sum(now - timedelta(days=14) <= parse_timestamp(r["ts"]) < now - timedelta(days=7) for r in evidence)
        group.update(count=len(evidence), open_requests=len(active), customers=len(known),
                     unidentified_requests=sum(not bool(r.get("customer")) for r in evidence),
                     qty=sum(r.get("qty") or 0 for r in evidence), budgets=money_totals(active),
                     original_wording=list(dict.fromkeys(r["product"] for r in evidence)),
                     value=sum(r.get("budget") or 0 for r in evidence) if len(money_totals(evidence)) <= 1 else None,
                     sources=dict(sources), reasons=dict(reasons), top_reason=top_reason, urgent=urgent,
                     score=score, priority="High" if score >= 16 or urgent >= 2 else "Medium" if score >= 8 else "Early signal",
                     suggested_action=action, action_detail=explanation,
                     action_basis=f"{reasons[top_reason]} of {len(reason_evidence)} {'open ' if active else ''}requests: {REASON_NAMES[top_reason].lower()}.",
                     recent_count=recent, previous_count=previous,
                     trend=f"{recent} in last 7 days · {previous} in prior 7 days",
                     review_count=sum(r["needs_review"] for r in evidence),
                     last=max(r["ts"] for r in evidence), signal_ids=[r["id"] for r in evidence],
                     state="active" if active else "resolved")
        demand.append(group)
    demand.sort(key=lambda x: (-x["score"], -x["count"], x["product"]))
    active_rows = [r for r in rows if r["status"] in ("open", "follow_up")]
    recovered = [r for r in rows if r["status"] == "recovered"]
    return dict(
        kpis=dict(signals=len(rows), products=len(demand), active_opportunities=sum(g["state"] == "active" for g in demand),
                  needs_review=sum(r["needs_review"] for r in rows), urgent=sum(r["urgency"] == 3 for r in active_rows),
                  recovered=len(recovered), recovered_value=money_totals(recovered, "recovered_amount", "recovered_currency"),
                  stated_budgets=money_totals(active_rows),
                  stated_value=sum(r.get("budget") or 0 for r in rows) if len(money_totals(rows)) <= 1 else None),
        demand=demand, recent=rows[:12],
        reasons=dict(Counter(r["effective_reason"] for r in active_rows)),
        sources=dict(Counter(r["source"] for r in rows)),
        note="Stated budgets are expressions of interest, not lost revenue. Recovery values are recorded by your team.",
    )
