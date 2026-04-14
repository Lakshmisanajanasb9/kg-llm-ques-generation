import requests
from collections import Counter
from typing import List, Dict, Optional, Any

HEADERS = {
    "User-Agent": "kg-llm-project/1.0",
    "Accept": "application/sparql-results+json",
}

SEARCH_CACHE: Dict[str, Optional[str]] = {}
SPARQL_CACHE: Dict[str, List[Dict[str, Any]]] = {}

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MAX_PER_PATTERN = 3
MAX_PER_MIDDLE = 4
MAX_PER_RELATED = 3
MIN_PATH_SCORE = -2


# --------------------------------------------------
# FILTERS
# Hard block only the worst ontology / wiki junk
# --------------------------------------------------

HARD_BAD_RELATIONS = {
    "instance of",
    "subclass of",
    "said to be the same as",
    "different from",
    "topic's main category",
    "properties for this type",
    "topic's main wikimedia portal",
    "described by source",
    "category for people born here",
    "category for people who died here",
    "main subject",
    "category combines topics",
    "on focus list of wikimedia project",
}

SOFT_BAD_RELATIONS = {
    "headquarters location",
    "official language",
    "applies to jurisdiction",
    "main regulatory text",
    "country",
    "country of citizenship",
    "located in the administrative territorial entity",
    "twinned administrative body",
    "emergency phone number",
    "electrical plug type",
    "time zone",
    "named after",
    "archives at",
    "office held by head of the organization",
    "capital",
    "continent",
}

LIGHT_BAD_ENTITIES = {
    "business",
    "enterprise",
    "commercial organization",
    "company",
    "business activity",
    "business (subject area)",
}

BAD_ENTITY_WORDS = {
    "wikimedia",
    "wikipedia",
    "template",
    "category",
    "list of",
    "disambiguation",
    "surname",
    "family name",
    "wikiproject",
    "portal",
    "outline of",
    "history of",
}

GOOD_RELATION_HINTS = {
    "has effect",
    "part of",
    "has part(s)",
    "participant in",
    "follows",
    "followed by",
    "replaced by",
    "replaces",
    "owned by",
    "owner of",
    "use",
    "applies to part",
    "conflict",
    "significant event",
    "executive body",
    "central bank",
    "legislative body",
    "head of government",
    "head of state",
}

GOOD_KEYWORDS = {
    "economy", "economic", "bank", "central bank", "finance", "financial",
    "interest", "interest rate", "inflation", "exports", "stocks", "stock",
    "bonds", "bond", "growth", "consumer", "market", "investment", "investor",
    "trade", "risk", "policy", "recession", "gdp", "eurozone", "shares",
    "spending", "prices", "asset", "assets", "monetary", "fiscal",
    "immigration", "sanction", "tariff", "protest", "rights", "regulation",
    "executive order", "asylum", "children", "families", "law", "government",
    "federal", "reserve", "administration", "congress", "crisis",
}

WEAK_KEYWORDS = {
    "language", "headquarters", "phone", "plug", "time zone",
    "administrative", "twinning", "wikimedia", "wikipedia",
    "category", "template", "list of", "vital articles",
    "named after", "origin", "symbolism", "outline of",
}

BAD_RELATED_WORDS = {
    "battle",
    "bombing",
    "war",
    "military",
}

TOPIC_QID_OVERRIDES = {
    "european central bank": "Q8901",
}


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

def score_search_result(result: Dict[str, Any], query: str) -> int:
    score = 0

    label = result.get("label", "").lower()
    desc = result.get("description", "").lower()
    aliases = " ".join(result.get("aliases", [])).lower()
    query_l = query.lower().strip()

    if label == query_l:
        score += 12
    if label.startswith(query_l):
        score += 6
    if query_l in label:
        score += 4
    if query_l in aliases:
        score += 5

    good_words = {
        "president", "prime minister", "politician", "government",
        "economist", "central bank", "economic", "finance",
        "commodity", "currency", "country", "organisation",
        "bank", "financial", "company", "holding company",
        "policy", "immigration", "protest", "rights",
    }
    for word in good_words:
        if word in desc:
            score += 2

    bad_words = {
        "disambiguation", "wikimedia", "category", "list",
        "template", "surname", "family name"
    }
    for word in bad_words:
        if word in label or word in desc:
            score -= 6

    return score


def search_entity(query: str) -> Optional[str]:
    query_l = query.strip().lower()

    if query_l in TOPIC_QID_OVERRIDES:
        return TOPIC_QID_OVERRIDES[query_l]

    if query in SEARCH_CACHE:
        return SEARCH_CACHE[query]

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": query,
        "language": "en",
        "limit": 10,
    }

    try:
        res = requests.get(WIKIDATA_API_URL, params=params, headers=HEADERS, timeout=20)
        res.raise_for_status()
        data = res.json()

        results = data.get("search", [])
        if not results:
            SEARCH_CACHE[query] = None
            return None

        ranked = sorted(results, key=lambda r: score_search_result(r, query), reverse=True)
        best = ranked[0]
        SEARCH_CACHE[query] = best["id"]
        print("Chosen:", best.get("label"), "-", best.get("description"))
        return best["id"]

    except Exception as e:
        print(f"search_entity failed for '{query}': {e}")
        SEARCH_CACHE[query] = None
        return None


# --------------------------------------------------
# SCORING / DEDUPE
# --------------------------------------------------

def contains_bad_entity_word(text: str) -> bool:
    text_l = text.strip().lower()
    return any(word in text_l for word in BAD_ENTITY_WORDS)


def text_keyword_score(text: str, good_keywords: set[str], bad_keywords: set[str]) -> float:
    text_l = text.strip().lower()
    score = 0.0

    for kw in good_keywords:
        if kw in text_l:
            score += 2.0

    for kw in bad_keywords:
        if kw in text_l:
            score -= 0.5

    return score


def score_path(event: str, relation1: str, middle: str, relation2: str, related: str) -> float:
    score = 0.0

    rel1_l = relation1.strip().lower()
    rel2_l = relation2.strip().lower()
    middle_l = middle.strip().lower()
    related_l = related.strip().lower()
    event_l = event.strip().lower()

    if rel1_l in GOOD_RELATION_HINTS:
        score += 3.0
    if rel2_l in GOOD_RELATION_HINTS:
        score += 3.0

    for text in [event_l, rel1_l, middle_l, rel2_l, related_l]:
        score += text_keyword_score(text, GOOD_KEYWORDS, WEAK_KEYWORDS)

    if contains_bad_entity_word(middle_l):
        score -= 4.0
    if contains_bad_entity_word(related_l):
        score -= 4.0

    if rel1_l in SOFT_BAD_RELATIONS:
        score -= 1.0
    if rel2_l in SOFT_BAD_RELATIONS:
        score -= 1.0

    if any(word in related_l for word in BAD_RELATED_WORDS):
        score -= 5.0

    return score


def dedupe_paths(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    clean = []

    for p in paths:
        key = (
            p["event"].strip().lower(),
            p["relation1"].strip().lower(),
            p["middle"].strip().lower(),
            p["relation2"].strip().lower(),
            p["related"].strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            clean.append(p)

    return clean


def limit_path_families(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pattern_counts = Counter()
    middle_counts = Counter()
    related_counts = Counter()

    clean = []

    for p in paths:
        pattern = (
            p["relation1"].strip().lower(),
            p["relation2"].strip().lower(),
        )
        middle_l = p["middle"].strip().lower()
        related_l = p["related"].strip().lower()

        if pattern_counts[pattern] >= MAX_PER_PATTERN:
            continue
        if middle_counts[middle_l] >= MAX_PER_MIDDLE:
            continue
        if related_counts[related_l] >= MAX_PER_RELATED:
            continue

        clean.append(p)
        pattern_counts[pattern] += 1
        middle_counts[middle_l] += 1
        related_counts[related_l] += 1

    return clean


# --------------------------------------------------
# SPARQL
# --------------------------------------------------

def build_two_hop_sparql(qid: str, limit: int = 300) -> str:
    return f"""
SELECT DISTINCT
  ?eventLabel
  ?prop1Label
  ?midLabel
  ?prop2Label
  ?endLabel
WHERE {{
  VALUES ?event {{ wd:{qid} }}

  ?event ?p1 ?mid .
  ?mid ?p2 ?end .

  FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
  FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))

  ?prop1 wikibase:directClaim ?p1 .
  ?prop2 wikibase:directClaim ?p2 .

  FILTER(ISIRI(?mid))
  FILTER(ISIRI(?end))
  FILTER(STRSTARTS(STR(?mid), "http://www.wikidata.org/entity/"))
  FILTER(STRSTARTS(STR(?end), "http://www.wikidata.org/entity/"))

  FILTER(?mid != ?event)
  FILTER(?end != ?event)
  FILTER(?end != ?mid)

  # Only hard-block the worst ontology / wiki / identifier junk here
  FILTER(?p1 NOT IN (
    wdt:P31,    # instance of
    wdt:P279,   # subclass of
    wdt:P910,   # topic's main category
    wdt:P921,   # main subject
    wdt:P971,   # category combines topics
    wdt:P1343,  # described by source
    wdt:P5008   # on focus list of Wikimedia project
  ))

  FILTER(?p2 NOT IN (
    wdt:P31,
    wdt:P279,
    wdt:P910,
    wdt:P921,
    wdt:P971,
    wdt:P1343,
    wdt:P5008,
    wdt:P213,   # ISNI
    wdt:P214,   # VIAF
    wdt:P227,   # GND ID
    wdt:P244,   # LOC authority ID
    wdt:P268,   # BnF ID
    wdt:P269,   # IdRef ID
    wdt:P646,   # Freebase ID
    wdt:P349,   # NDL ID
    wdt:P691,   # NKC ID
    wdt:P7859,  # WorldCat Identities ID
    wdt:P9037,  # BHCL UUID
    wdt:P8189   # J9U ID
  ))

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
""".strip()


def run_sparql(query: str) -> List[Dict[str, Any]]:
    if query in SPARQL_CACHE:
        return SPARQL_CACHE[query]

    try:
        res = requests.get(
            WIKIDATA_SPARQL_URL,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=45,
        )
        res.raise_for_status()
        data = res.json()
        rows = data.get("results", {}).get("bindings", [])
        SPARQL_CACHE[query] = rows
        return rows
    except Exception as e:
        print(f"run_sparql failed: {e}")
        return []


# --------------------------------------------------
# MAIN RETRIEVAL
# --------------------------------------------------

def get_wikidata_context_sparql(
    topic: str,
    max_paths: int = 30,
    sparql_limit: int = 300,
) -> List[Dict[str, Any]]:
    qid = search_entity(topic)
    print("[SPARQL] topic:", topic)
    print("[SPARQL] chosen qid:", qid)

    if not qid:
        print("[SPARQL] no qid found")
        return []

    query = build_two_hop_sparql(qid=qid, limit=sparql_limit)
    rows = run_sparql(query)
    print("[SPARQL] raw row count:", len(rows))

    paths: List[Dict[str, Any]] = []

    reject_counts = {
        "hard_bad_rel1": 0,
        "hard_bad_rel2": 0,
        "bad_entity": 0,
        "bad_related_word": 0,
        "low_score": 0,
    }

    rel1_counter = Counter()

    for row in rows:
        event = row.get("eventLabel", {}).get("value", "").strip()
        relation1 = row.get("prop1Label", {}).get("value", "").strip()
        middle = row.get("midLabel", {}).get("value", "").strip()
        relation2 = row.get("prop2Label", {}).get("value", "").strip()
        related = row.get("endLabel", {}).get("value", "").strip()

        if not all([event, relation1, middle, relation2, related]):
            continue

        rel1_l = relation1.lower()
        rel2_l = relation2.lower()
        middle_l = middle.lower()
        related_l = related.lower()

        rel1_counter[rel1_l] += 1

        if rel1_l in HARD_BAD_RELATIONS:
            reject_counts["hard_bad_rel1"] += 1
            continue

        if rel2_l in HARD_BAD_RELATIONS:
            reject_counts["hard_bad_rel2"] += 1
            continue

        if middle_l in LIGHT_BAD_ENTITIES or related_l in LIGHT_BAD_ENTITIES:
            reject_counts["bad_entity"] += 1
            continue

        if contains_bad_entity_word(middle_l) or contains_bad_entity_word(related_l):
            reject_counts["bad_entity"] += 1
            continue

        if any(word in related_l for word in BAD_RELATED_WORDS):
            reject_counts["bad_related_word"] += 1
            continue

        path_score = score_path(event, relation1, middle, relation2, related)

        if path_score < MIN_PATH_SCORE:
            reject_counts["low_score"] += 1
            continue

        paths.append({
            "event": event,
            "relation1": relation1,
            "middle": middle,
            "relation2": relation2,
            "related": related,
            "time": None,
            "score": path_score,
        })

    print("[SPARQL] top rel1 labels:", rel1_counter.most_common(15))
    print("[SPARQL] reject counts:", reject_counts)

    paths = dedupe_paths(paths)
    paths = sorted(paths, key=lambda x: x["score"], reverse=True)
    paths = limit_path_families(paths)

    final_paths = paths[:max_paths]
    print("[SPARQL] final path count:", len(final_paths))
    return final_paths


def format_wikidata_context(paths: List[Dict[str, Any]]) -> str:
    if not paths:
        return ""

    return "\n".join(
        f"{p['event']} --[{p['relation1']}]--> {p['middle']} --[{p['relation2']}]--> {p['related']}"
        for p in paths
    )


if __name__ == "__main__":
    topic = "Trump administration"

    rows = get_wikidata_context_sparql(
        topic,
        max_paths=30,
        sparql_limit=300,
    )

    print("\n--- RESULTS ---")
    for r in rows:
        print(r)

    print("\n--- FORMATTED ---")
    print(format_wikidata_context(rows))