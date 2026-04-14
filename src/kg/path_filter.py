import re

BAD_RELATIONS = {"Relate_To", "Has", "Operate_In", "Is_Member_Of"}
BAD_OBJECTS = {"impact", "process", "measure", "concerns"}

STRONG_RELATIONS = {
    "Impact",
    "Positive_Impact_On",
    "Negative_Impact_On",
    "Increase",
    "Decrease",
    "Raise",
    "Control",
    "Announce",
    "Introduce",
    "Produce",
}

GOOD_KEYWORDS = {
    "interest", "rate", "borrowing", "stock", "market", "economy",
    "inflation", "bond", "treasury", "dollar", "growth", "investment",
    "unemployment", "consumer", "price", "policy", "yield", "gold",
    "spending", "federal reserve", "mortgage", "business"
}


def is_vague(text):
    return str(text).strip().lower() in BAD_OBJECTS


def has_good_keyword(text):
    text = str(text).lower()
    return any(word in text for word in GOOD_KEYWORDS)


def normalise_text(text):
    text = str(text).lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)

    stop_words = {"the", "a", "an", "of", "and"}
    words = [w for w in text.split() if w not in stop_words]

    cleaned = []
    for w in words:
        if w.endswith("s") and len(w) > 4:
            w = w[:-1]
        cleaned.append(w)

    text = " ".join(cleaned)

    replacements = {
        "unemployment rate": "unemployment",
        "interest rate": "interest rates",
        "consumer price": "consumer prices",
        "asset price": "asset prices",
        "stock market": "stocks",
        "u s federal reserve": "federal reserve",
        "the u s economy": "economy",
        "u s economy": "economy",
    }

    return replacements.get(text, text)


def semantically_same_object(obj1, obj2):
    n1 = normalise_text(obj1)
    n2 = normalise_text(obj2)

    if n1 == n2:
        return True

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    if not tokens1 or not tokens2:
        return False

    overlap = len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))
    return overlap >= 0.8


def light_filter(rows):
    cleaned = []

    for row in rows:
        s = row.get("event")
        m = row.get("middle")
        o = row.get("related")
        r1 = row.get("relation1")
        r2 = row.get("relation2")

        if not s or not m or not o or not r1 or not r2:
            continue

        if s == m or m == o or s == o:
            continue

        if is_vague(o):
            continue

        if r1 in BAD_RELATIONS and r2 in BAD_RELATIONS:
            continue

        cleaned.append(row)

    return cleaned


def score_path(row):
    score = 0
    r1 = row.get("relation1", "")
    r2 = row.get("relation2", "")
    middle = row.get("middle", "")
    obj = row.get("related", "")

    if r1 in STRONG_RELATIONS:
        score += 2
    if r2 in STRONG_RELATIONS:
        score += 3

    if r1 in BAD_RELATIONS:
        score -= 1
    if r2 in BAD_RELATIONS:
        score -= 1

    if has_good_keyword(middle):
        score += 1
    if has_good_keyword(obj):
        score += 3

    if is_vague(obj):
        score -= 4

    if len(str(obj)) > 8:
        score += 1

    return score


def remove_semantic_repetition(rows):
    kept = []

    for row in rows:
        repeated = False

        for prev in kept:
            same_subject = normalise_text(row["event"]) == normalise_text(prev["event"])
            same_middle = normalise_text(row["middle"]) == normalise_text(prev["middle"])

            if same_subject and same_middle and semantically_same_object(row["related"], prev["related"]):
                repeated = True
                break

        if not repeated:
            kept.append(row)

    return kept


def select_diverse_middle_paths(rows, k=10, max_per_middle=2):
    selected = []
    middle_counts = {}

    for row in rows:
        middle = row["middle"]
        middle_counts.setdefault(middle, 0)

        if middle_counts[middle] >= max_per_middle:
            continue

        selected.append(row)
        middle_counts[middle] += 1

        if len(selected) >= k:
            break

    return selected


def prepare_paths_for_generation(rows, target_k=10):
    rows = light_filter(rows)
    rows = sorted(rows, key=score_path, reverse=True)
    rows = remove_semantic_repetition(rows)
    rows = select_diverse_middle_paths(rows, k=target_k, max_per_middle=2)
    return rows