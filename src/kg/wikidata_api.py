import requests
from collections import defaultdict

HEADERS = {"User-Agent": "kg-llm-project/1.0"}

SEARCH_CACHE = {}
LABEL_CACHE = {}
ENTITY_CACHE = {}

LIGHT_BAD_RELATIONS = {
    '''"instance of",
    "subclass of",
    "said to be the same as",
    "different from",
    "topic's main category",
    "properties for this type",'''
    "instance of",
    "subclass of",
    "facet of",
}

LIGHT_BAD_ENTITIES = {
    '''"business",
    "enterprise",
    "commercial organization",
    "company",
    "business activity",
    "business (subject area)",
    "wikimedia",
    "wikipedia",
    "template",
    "category",
    "list of",
    "disambiguation",
    "wikiproject",
    "portal",
    "outline of",
    "history of",'''
}

middle_counts = defaultdict(int)

# increase this if you want even more paths per middle node
MAX_PER_MIDDLE = 8

# retrieval knobs
FIRST_HOP_LIMIT = 60   #TRY A LARGER NUMBER LIKE 100 OR 200
SECOND_HOP_LIMIT = 20
MAX_CLAIMS_PER_PROPERTY = 8


def score_search_result(result, query):
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
        '''"president", "prime minister", "politician", "government",
        "economist", "central bank", "economic", "finance",
        "commodity", "currency", "country", "organisation",
        "bank", "financial", "company", "holding company"'''
    }
    for word in good_words:
        if word in desc:
            score += 2

    bad_words = {
        '''"disambiguation", "wikimedia", "category", "list",
        "template", "surname", "family name"'''
    }
    for word in bad_words:
        if word in label or word in desc:
            score -= 6

    return score


def search_entity(query):
    if query in SEARCH_CACHE:
        return SEARCH_CACHE[query]

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": query,
        "language": "en",
        "limit": 10,
    }

    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=20)
        res.raise_for_status()

        if "json" not in res.headers.get("Content-Type", "").lower():
            print("Unexpected response:", res.text[:300])
            SEARCH_CACHE[query] = None
            return None

        data = res.json()
        results = data.get("search", [])
        if not results:
            SEARCH_CACHE[query] = None
            return None

        ranked = sorted(results, key=lambda r: score_search_result(r, query), reverse=True)
        best = ranked[0]
        print("Chosen:", best.get("label"), "-", best.get("description"))
        SEARCH_CACHE[query] = best["id"]
        return best["id"]

    except Exception as e:
        print(f"search_entity failed for '{query}': {e}")
        SEARCH_CACHE[query] = None
        return None


def get_label(entity_id):
    if entity_id in LABEL_CACHE:
        return LABEL_CACHE[entity_id]

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "languages": "en",
    }

    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=20)
        res.raise_for_status()

        if "json" not in res.headers.get("Content-Type", "").lower():
            LABEL_CACHE[entity_id] = entity_id
            return entity_id

        data = res.json()
        entity = data.get("entities", {}).get(entity_id, {})
        label = entity.get("labels", {}).get("en", {}).get("value", entity_id)
        LABEL_CACHE[entity_id] = label
        return label
    except Exception:
        LABEL_CACHE[entity_id] = entity_id
        return entity_id


def get_neighbors(qid, limit=FIRST_HOP_LIMIT):
    """
    Return raw entity-to-entity edges from Wikidata.
    High-recall version: keeps more claims per property.
    """
    if qid in ENTITY_CACHE:
        claims = ENTITY_CACHE[qid]
    else:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
            res.raise_for_status()

            if "json" not in res.headers.get("Content-Type", "").lower():
                print(f"get_neighbors non-JSON for {qid}: {res.text[:300]}")
                return []

            data = res.json()
            claims = data.get("entities", {}).get(qid, {}).get("claims", {})
            ENTITY_CACHE[qid] = claims
        except Exception as e:
            print(f"get_neighbors failed for {qid}: {e}")
            return []

    results = []

    for prop, values in claims.items():
        for v in values[:MAX_CLAIMS_PER_PROPERTY]:
            try:
                mainsnak = v.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {}).get("value")

                if isinstance(datavalue, dict) and "id" in datavalue:
                    target = datavalue["id"]
                    results.append((prop, target))

                    if len(results) >= limit:
                        return results
            except Exception:
                continue

    return results


def dedupe_paths(paths):
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


def get_wikidata_context(topic, max_paths=120):
    global middle_counts
    middle_counts = defaultdict(int)

    qid = search_entity(topic)
    print("[WIKI] topic:", topic)
    print("[WIKI] chosen qid:", qid)

    if not qid:
        print("[WIKI] no qid found")
        return []

    event_label = get_label(qid)
    print("[WIKI] chosen label:", event_label)

    paths = []

    first_hop = get_neighbors(qid, limit=FIRST_HOP_LIMIT)
    print("[WIKI] raw first_hop count:", len(first_hop))

    for prop1, mid in first_hop:
        relation1 = get_label(prop1)
        middle = get_label(mid)

        if not middle or mid == qid:
            continue

        rel1_l = relation1.strip().lower()
        middle_l = middle.strip().lower()

        if rel1_l in LIGHT_BAD_RELATIONS:
            continue
        if middle_l in LIGHT_BAD_ENTITIES:
            continue

        # stop one middle node from flooding everything
        if middle_counts[middle_l] >= MAX_PER_MIDDLE:
            continue

        print("[WIKI] first hop:", relation1, "->", middle)

        # high-recall: expand many second-hop edges, not just 1
        second_hop = get_neighbors(mid, limit=SECOND_HOP_LIMIT)
        print("[WIKI] raw second_hop count from", middle, "=", len(second_hop))

        for prop2, end in second_hop:
            relation2 = get_label(prop2)
            related = get_label(end)

            if not related:
                continue

            # avoid trivial loops only
            if end == qid or end == mid:
                continue
            if related.strip().lower() == event_label.strip().lower():
                continue
            if related.strip().lower() == middle.strip().lower():
                continue

            rel2_l = relation2.strip().lower()
            related_l = related.strip().lower()

            bad_relation_pairs = {
                ("instance of", "subclass of"),
                ("country", "participant in"),
            }

            bad_related_words = {
                "battle",
                "bombing",
                "war",
                "military",
            }

            if (rel1_l, rel2_l) in bad_relation_pairs:
                continue

            if rel1_l == "country" and middle_l == "united states":
                continue

            if any(word in related_l for word in bad_related_words):
                continue

            if rel2_l in LIGHT_BAD_RELATIONS:
                continue
            if related_l in LIGHT_BAD_ENTITIES:
                continue

            print("[WIKI] candidate path:", event_label, relation1, middle, relation2, related)

            paths.append({
                "event": event_label,
                "relation1": relation1,
                "middle": middle,
                "relation2": relation2,
                "related": related,
                "time": None,
            })

            middle_counts[middle_l] += 1

            if len(paths) >= max_paths * 2:
                break

        if len(paths) >= max_paths * 2:
            break

    paths = dedupe_paths(paths)
    print("[WIKI] final path count before trim:", len(paths))
    return paths[:max_paths]


def format_wikidata_context(paths):
    if not paths:
        return ""

    return "\n".join(
        f"{p['event']} --[{p['relation1']}]--> {p['middle']} --[{p['relation2']}]--> {p['related']}"
        for p in paths
    )