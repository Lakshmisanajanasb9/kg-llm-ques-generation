from neo4j import GraphDatabase
import re

print("starting script")

URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "sai.sanju9"

ENTITY = "Europe"   # change this to China / United States / etc.
LIMIT_ALL = 600
TARGET_CLEAN = 10

BAD_RELATIONS = {"Relate_To", "Has", "Operate_In", "Is_Member_Of"}
BAD_OBJECTS = {"impact", "process", "measure", "concerns"}
BAD_MIDDLES = {"Meta Platforms Inc.", "Labour Party"}

STRONG_RELATIONS = {
    "Positive_Impact_On",
    "Negative_Impact_On",
    "Increase",
    "Decrease",
    "Raise",
    "Announce",
    "Introduce",
    "Participates_In",
    "Invests_In",
}

GOOD_KEYWORDS = {
    "interest", "rate", "borrowing", "stock", "market", "economy",
    "inflation", "bond", "treasury", "dollar", "growth", "investment",
    "unemployment", "consumer", "price", "policy", "yield", "gold",
    "spending", "federal reserve", "mortgage", "business", "brexit",
    "central bank", "european union", "oil", "production"
}

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def print_all_paths(session, entity):
    result = session.run(
        """
        MATCH (a:Entity {name: $entity})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)
        WHERE a.name IS NOT NULL
          AND b.name IS NOT NULL
          AND c.name IS NOT NULL
          AND r1.rname IS NOT NULL
          AND r2.rname IS NOT NULL
          AND r1.split = 'train'
          AND r2.split = 'train'
        RETURN
            a.name AS subject,
            r1.rname AS r1,
            b.name AS middle,
            r2.rname AS r2,
            c.name AS object,
            r1.time AS t1,
            r2.time AS t2,
            r1.rid AS rid1,
            r2.rid AS rid2
        ORDER BY t1 DESC, t2 DESC
        LIMIT $limit
        """,
        entity=entity,
        limit=LIMIT_ALL,
    )

    rows = list(result)
    print(f"\n=== ALL RAW TEMPORAL 2-HOP PATHS FOR '{entity}' ({len(rows)}) ===\n")

    for i, record in enumerate(rows, start=1):
        print(
            f"{i}. "
            f"{record['subject']} --[{record['r1']} @ t={record['t1']}, rid={record['rid1']}]--> "
            f"{record['middle']} --[{record['r2']} @ t={record['t2']}, rid={record['rid2']}]--> "
            f"{record['object']}"
        )

    return rows


def dedupe_keep_latest(rows):
    best = {}

    for row in rows:
        key = (
            row["subject"],
            row["r1"],
            row["middle"],
            row["r2"],
            row["object"],
        )
        score = (row["t1"], row["t2"])

        if key not in best or score > (best[key]["t1"], best[key]["t2"]):
            best[key] = row

    return list(best.values())


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
        "united states": "us",
        "united kingdom": "uk",
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


def remove_semantic_repetition(rows):
    kept = []

    for row in rows:
        is_repeat = False

        for prev in kept:
            same_subject = normalise_text(row["subject"]) == normalise_text(prev["subject"])
            same_middle = normalise_text(row["middle"]) == normalise_text(prev["middle"])

            if same_subject and same_middle and semantically_same_object(row["object"], prev["object"]):
                is_repeat = True
                break

        if not is_repeat:
            kept.append(row)

    return kept


def light_filter(rows):
    cleaned = []

    for row in rows:
        s = row["subject"]
        m = row["middle"]
        o = row["object"]
        r1 = row["r1"]
        r2 = row["r2"]

        if s == m or m == o or s == o:
            continue

        if is_vague(o):
            continue

        # stricter: reject if either relation is bad
        if r1 in BAD_RELATIONS or r2 in BAD_RELATIONS:
            continue

        # reject weak double-vague chains
        if r1 == "Impact" and r2 == "Impact":
            continue

        # reject Produce-based paths for this use case
        if r1 == "Produce" or r2 == "Produce":
            continue

        # reject noisy middle nodes
        if m in BAD_MIDDLES:
            continue

        # very specific known bad pattern
        if m == "Donald Trump" and o == "European Central Bank":
            continue

        cleaned.append(row)

    return cleaned


def score_path(row):
    score = 0
    r1 = row["r1"]
    r2 = row["r2"]
    middle = row["middle"]
    obj = row["object"]

    if r1 in STRONG_RELATIONS:
        score += 2
    if r2 in STRONG_RELATIONS:
        score += 3

    # mild penalty for vague-but-common relations
    if r1 == "Impact":
        score -= 1
    if r2 == "Impact":
        score -= 1
    if r1 == "Control":
        score -= 2
    if r2 == "Control":
        score -= 2

    if has_good_keyword(middle):
        score += 1
    if has_good_keyword(obj):
        score += 3

    if is_vague(obj):
        score -= 4

    if len(str(obj)) > 8:
        score += 1

    return score


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


def print_clean_paths(rows, entity):
    cleaned = dedupe_keep_latest(rows)
    cleaned = light_filter(cleaned)
    cleaned = sorted(cleaned, key=score_path, reverse=True)
    cleaned = remove_semantic_repetition(cleaned)
    cleaned = select_diverse_middle_paths(
        cleaned,
        k=TARGET_CLEAN,
        max_per_middle=2,
    )

    print(f"\n=== CLEANED PATHS FOR '{entity}' ({len(cleaned)}) ===\n")

    if not cleaned:
        print("No cleaned paths found after filtering.")
        return

    for i, row in enumerate(cleaned, start=1):
        print(
            f"{i}. "
            f"{row['subject']} --[{row['r1']} @ t={row['t1']}]--> "
            f"{row['middle']} --[{row['r2']} @ t={row['t2']}]--> "
            f"{row['object']}"
        )


try:
    with driver.session() as session:
        print("connected, testing query")
        test = session.run("RETURN 1 AS x")
        print("test result:", test.single()["x"])

        all_rows = print_all_paths(session, ENTITY)
        print_clean_paths(all_rows, ENTITY)

finally:
    driver.close()
    print("\ndone")