import requests
import re
import time
import json
from pathlib import Path
from src.rag.generate_questions import generate_questions

HEADERS = {"User-Agent": "kg-llm-project/1.0"}

TOPIC = "President Obama"

GOOD_PROPERTIES = {
    "position held",
    "member of political party",
    "occupation",
    "country of citizenship",
    "applies to jurisdiction",
    "country",
    "legislated by",
    "signatory",
}

BAD_PROPERTIES = {
    "instance of",
    "subclass of",
    "sex or gender",
    "given name",
    "family name",
    "father",
    "mother",
    "child",
    "spouse",
    "has pet",
    "medical condition",
    "educated at",
    "place of birth",
    "handedness",
    "topic's main category",
    "topic has template",
    "language of work or name",
    "published in",
    "described by source",
    "properties for this type",
    "topic's main wikimedia portal",
    "part of",
}

SEARCH_CACHE = {}
ENTITY_CACHE = {}

HTTP_CACHE_FILE = Path("wikidata_http_cache.json")
LABEL_CACHE_FILE = Path("wikidata_label_cache.json")


def load_json_cache(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_json_cache(path: Path, data):
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Failed to save cache {path}: {e}")


HTTP_CACHE = load_json_cache(HTTP_CACHE_FILE)
LABEL_CACHE = load_json_cache(LABEL_CACHE_FILE)


def make_cache_key(url, params=None):
    if not params:
        return url

    parts = [f"{k}={params[k]}" for k in sorted(params)]
    return f"{url}?{'&'.join(parts)}"


def safe_get_json(url, params=None, timeout=20, retries=4, use_cache=True):
    cache_key = make_cache_key(url, params)

    if use_cache and cache_key in HTTP_CACHE:
        return HTTP_CACHE[cache_key]

    wait = 1.0

    for attempt in range(retries):
        try:
            time.sleep(0.6)
            res = requests.get(url, params=params, headers=HEADERS, timeout=timeout)

            if res.status_code == 429:
                print(f"Rate limited. Sleeping {wait:.1f}s...")
                time.sleep(wait)
                wait *= 2
                continue

            res.raise_for_status()

            if "json" not in res.headers.get("Content-Type", "").lower():
                print(f"Unexpected non-JSON response for {cache_key}")
                return None

            data = res.json()

            if use_cache:
                HTTP_CACHE[cache_key] = data
                save_json_cache(HTTP_CACHE_FILE, HTTP_CACHE)

            return data

        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"Request failed after retries: {e}")
                return None

            print(f"Request failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            wait *= 2

        except ValueError as e:
            print(f"JSON decode failed for {cache_key}: {e}")
            return None

    return None


def is_good_property(prop: str) -> bool:
    prop = prop.lower().strip()

    if prop in BAD_PROPERTIES:
        return False

    if prop in GOOD_PROPERTIES:
        return True

    return False


def search_entity_candidates(query, limit=10):
    cache_key = (query.strip().lower(), limit)
    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": query.strip(),
        "language": "en",
        "limit": limit,
    }

    try:
        data = safe_get_json(url, params=params, timeout=20)
        if not data:
            return []

        results = data.get("search", [])

        candidates = []
        for r in results:
            candidates.append({
                "id": r.get("id"),
                "label": r.get("label", ""),
                "description": r.get("description", ""),
            })

        SEARCH_CACHE[cache_key] = candidates
        return candidates

    except Exception as e:
        print(f"search_entity_candidates failed for '{query}': {e}")
        return []


def llm_choose_candidate(topic, candidates):
    if not candidates:
        return None

    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. ID: {c['id']} | Label: {c['label']} | Description: {c['description']}"
        )

    prompt = f"""
You are selecting the best Wikidata candidate for a reasoning-based system focused on politics, economics, finance, institutions, and global events.

User topic: "{topic}"

Candidates:
{chr(10).join(lines)}

Rules:
- Choose ONLY one of the numbered candidates above.
- Prefer the candidate that most directly matches the topic itself.
- If the topic is a person's name or title, prefer the actual person over related laws, libraries, articles, places, animals, games, or family names.
- Prefer political, economic, institutional, or globally important real-world entities.
- Avoid cities, animals, entertainment items, families, Wikimedia pages, categories, portals, and libraries unless there is no better option.
- Return ONLY the candidate number.
- Do not explain.
"""

    try:
        output = generate_questions(prompt)

        if isinstance(output, list):
            text = " ".join(str(x) for x in output).strip()
        else:
            text = str(output).strip()

        print("\n--- LLM RAW OUTPUT ---\n")
        print(text)

        match = re.search(r"\b([1-9]|10)\b", text)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]["id"]

    except Exception as e:
        print(f"llm_choose_candidate failed: {e}")

    return None


def get_label(entity_id):
    if not entity_id:
        return None

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
        data = safe_get_json(url, params=params, timeout=20)
        if not data:
            LABEL_CACHE[entity_id] = entity_id
            save_json_cache(LABEL_CACHE_FILE, LABEL_CACHE)
            return entity_id

        label = (
            data.get("entities", {})
            .get(entity_id, {})
            .get("labels", {})
            .get("en", {})
            .get("value", entity_id)
        )

        LABEL_CACHE[entity_id] = label
        save_json_cache(LABEL_CACHE_FILE, LABEL_CACHE)
        return label

    except Exception:
        LABEL_CACHE[entity_id] = entity_id
        save_json_cache(LABEL_CACHE_FILE, LABEL_CACHE)
        return entity_id


def get_entity_data(qid):
    if not qid:
        return None

    if qid in ENTITY_CACHE:
        return ENTITY_CACHE[qid]

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

    try:
        data = safe_get_json(url, timeout=20)
        if not data:
            return None

        ENTITY_CACHE[qid] = data
        return data

    except Exception as e:
        print(f"get_entity_data failed for {qid}: {e}")
        return None


def get_neighbors(qid, limit=5):
    data = get_entity_data(qid)
    if not data:
        return []

    claims = data.get("entities", {}).get(qid, {}).get("claims", {})
    results = []

    for prop, values in claims.items():
        prop_label = get_label(prop)
        if not prop_label:
            continue

        if not is_good_property(prop_label):
            continue

        for v in values:
            try:
                datavalue = v["mainsnak"]["datavalue"]["value"]

                if isinstance(datavalue, dict) and "id" in datavalue:
                    target = datavalue["id"]
                    results.append((prop, target))

                    if len(results) >= limit:
                        return results

            except Exception:
                continue

    return results


def print_entity_claims(qid, limit=20):
    if not qid:
        print("No QID provided.")
        return

    data = get_entity_data(qid)
    if not data:
        print(f"Failed to fetch claims for {qid}")
        return

    event_label = get_label(qid)
    claims = data.get("entities", {}).get(qid, {}).get("claims", {})

    print(f"\n=== RAW CLAIMS FOR {qid} ({event_label}) ===\n")

    count = 0

    for prop, values in claims.items():
        prop_label = get_label(prop)
        if not prop_label:
            continue

        prop_label = prop_label.lower()

        if prop_label in BAD_PROPERTIES:
            continue

        if prop_label not in GOOD_PROPERTIES:
            continue

        for v in values:
            try:
                datavalue = v["mainsnak"]["datavalue"]["value"]

                if isinstance(datavalue, dict) and "id" in datavalue:
                    target = datavalue["id"]
                    target_label = get_label(target)

                    print(f"{event_label} --[{prop_label}]--> {target_label}")

                    count += 1
                    if count >= limit:
                        return

            except Exception:
                continue


def get_two_hop_paths(qid, max_paths=8):
    paths = []
    seen = set()

    event_label = get_label(qid)
    first_hop = get_neighbors(qid, limit=5)

    for prop1, mid in first_hop:
        mid_label = get_label(mid)
        prop1_label = get_label(prop1)

        if not mid_label or not prop1_label:
            continue

        if not is_good_property(prop1_label):
            continue

        second_hop = get_neighbors(mid, limit=3)

        for prop2, end in second_hop:
            prop2_label = get_label(prop2)
            end_label = get_label(end)

            if not prop2_label or not end_label:
                continue

            if not is_good_property(prop2_label):
                continue

            if end == qid or mid == end:
                continue

            key = (
                event_label,
                prop1_label,
                mid_label,
                prop2_label,
                end_label,
            )
            if key in seen:
                continue

            seen.add(key)

            paths.append({
                "event": event_label,
                "relation1": prop1_label,
                "middle": mid_label,
                "relation2": prop2_label,
                "related": end_label,
            })

            if len(paths) >= max_paths:
                return paths

    return paths


def print_paths(paths):
    print("\n=== 2-HOP PATHS ===\n")

    if not paths:
        print("No 2-hop paths found.")
        return

    for i, p in enumerate(paths, 1):
        print(
            f"{i}. {p['event']} --[{p['relation1']}]--> "
            f"{p['middle']} --[{p['relation2']}]--> {p['related']}"
        )


if __name__ == "__main__":
    print(f"\n=== TOPIC: {TOPIC} ===\n")

    candidates = search_entity_candidates(TOPIC)

    print("--- CANDIDATES ---\n")
    for c in candidates:
        print(c)

    qid = llm_choose_candidate(TOPIC, candidates)

    if not qid:
        print("\nNo valid Wikidata entity was chosen.")
    else:
        chosen_label = get_label(qid)
        print("\n--- CHOSEN ENTITY ---\n", qid, "-", chosen_label)

        # Leave this off during normal runs because it triggers lots of label lookups.
        # print_entity_claims(qid)

        paths = get_two_hop_paths(qid, max_paths=8)
        print_paths(paths)