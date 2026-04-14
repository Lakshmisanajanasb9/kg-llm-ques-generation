import requests
from typing import List, Dict, Optional, Any

HEADERS = {
    "User-Agent": "kg-llm-project/1.0",
    "Accept": "application/sparql-results+json",
}

SEARCH_CACHE: Dict[str, Optional[str]] = {}
SPARQL_CACHE: Dict[str, List[Dict[str, Any]]] = {}

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"


LIGHT_BAD_RELATIONS = {
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
}

BAD_RELATIONS = {
    "headquarters location",
    "official language",
    "applies to jurisdiction",
    "main regulatory text",
    "country",
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
    "facet of",
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

GOOD_KEYWORDS = {
    "economy", "economic", "bank", "central bank", "finance", "financial",
    "interest", "interest rate", "inflation", "exports", "stocks", "stock",
    "bonds", "bond", "growth", "consumer", "market", "investment", "investor",
    "trade", "risk", "policy", "recession", "gdp", "eurozone", "shares",
    "spending", "prices", "asset", "assets", "monetary", "fiscal",
    "immigration", "sanction", "tariff", "protest", "rights", "regulation",
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


def contains_bad_entity_word(text: str) -> bool:
    text_l = text.strip().lower()
    return any(word in text_l for word in BAD_ENTITY_WORDS)


def text_keyword_score(text: str, good_keywords: set[str], bad_keywords: set[str]) -> int:
    text_l = text.strip().lower()
    score = 0

    for kw in good_keywords:
        if kw in text_l:
            score += 2

    for kw in bad_keywords:
        if kw in text_l:
            score -= 3

    return score


def score_path(event: str, relation1: str, middle: str, relation2: str, related: str) -> int:
    score = 0

    rel1_l = relation1.strip().lower()
    rel2_l = relation2.strip().lower()
    middle_l = middle.strip().lower()
    related_l = related.strip().lower()
    event_l = event.strip().lower()

    for text in [event_l, rel1_l, middle_l, rel2_l, related_l]:
        score += text_keyword_score(text, GOOD_KEYWORDS, WEAK_KEYWORDS)

    if contains_bad_entity_word(middle_l):
        score -= 6
    if contains_bad_entity_word(related_l):
        score -= 6

    if rel1_l in LIGHT_BAD_RELATIONS or rel1_l in BAD_RELATIONS:
        score -= 8
    if rel2_l in LIGHT_BAD_RELATIONS or rel2_l in BAD_RELATIONS:
        score -= 8

    if any(word in related_l for word in BAD_RELATED_WORDS):
        score -= 8

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


def build_two_hop_sparql(qid: str, limit: int = 100) -> str:
    return f"""
SELECT DISTINCT ?eventLabel ?p1Label ?midLabel ?p2Label ?endLabel WHERE {{
  VALUES ?event {{ wd:{qid} }}

  ?event ?p1 ?mid .
  ?mid ?p2 ?end .

  FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
  FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))

  FILTER(?mid != ?event)
  FILTER(?end != ?event)
  FILTER(?end != ?mid)

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


def get_wikidata_context_sparql(topic: str, max_paths: int = 20, sparql_limit: int = 120) -> List[Dict[str, Any]]:
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

    for row in rows:
        event = row.get("eventLabel", {}).get("value", "").strip()
        relation1 = row.get("p1Label", {}).get("value", "").strip()
        middle = row.get("midLabel", {}).get("value", "").strip()
        relation2 = row.get("p2Label", {}).get("value", "").strip()
        related = row.get("endLabel", {}).get("value", "").strip()

        if not all([event, relation1, middle, relation2, related]):
            continue

        rel1_l = relation1.lower()
        rel2_l = relation2.lower()
        middle_l = middle.lower()
        related_l = related.lower()

        if rel1_l in LIGHT_BAD_RELATIONS or rel1_l in BAD_RELATIONS:
            continue
        if rel2_l in LIGHT_BAD_RELATIONS or rel2_l in BAD_RELATIONS:
            continue

        if middle_l in LIGHT_BAD_ENTITIES or related_l in LIGHT_BAD_ENTITIES:
            continue

        if contains_bad_entity_word(middle_l) or contains_bad_entity_word(related_l):
            continue

        if any(word in related_l for word in BAD_RELATED_WORDS):
            continue

        path_score = score_path(event, relation1, middle, relation2, related)
        if path_score < 1:
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

    paths = dedupe_paths(paths)
    paths = sorted(paths, key=lambda x: x["score"], reverse=True)

    print("[SPARQL] final path count:", len(paths[:max_paths]))
    return paths[:max_paths]


def format_wikidata_context(paths: List[Dict[str, Any]]) -> str:
    if not paths:
        return ""

    return "\n".join(
        f"{p['event']} --[{p['relation1']}]--> {p['middle']} --[{p['relation2']}]--> {p['related']}"
        for p in paths
    )