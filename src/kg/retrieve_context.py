from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Set, Tuple

# Local Neo4j Desktop connection
URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "sai.sanju9"   # keep your real password

BAD_RELATIONS: Set[str] = {
    "Has",
    "Relate_To",
    "Operate_In",
    "Control",
    "Raise",
    "Is_Member_Of",
}

BAD_MIDDLES: Set[str] = {
    "Meta Platforms Inc.",
    "Labour Party",
}

JUNK_PREFIXES = (
    "Portal:",
    "Category:",
    "Template:",
    "outline of ",
)

JUNK_EXACT = {
    "Wikimedia portal",
    "Wikimedia outline article",
    "Wikimedia permanent duplicate item",
}


def _is_junk_entity(name: Optional[str]) -> bool:
    if not name:
        return True

    lowered = name.strip().lower()

    if name in JUNK_EXACT:
        return True

    if lowered.startswith("q") and lowered[1:].isdigit():
        return True

    return any(lowered.startswith(prefix.lower()) for prefix in JUNK_PREFIXES)


def get_context(
    event_keyword: str,
    limit: int = 50,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not event_keyword or not event_keyword.strip():
        return []
    
    # AVOID FOR WEAK/NOISY 2-HOP CHAINS WITH SAME ENTITY REPEATED -----------------------------
    #OR toLower(x.name) CONTAINS toLower($kw)
    #OR toLower(x.name) CONTAINS toLower($kw)--------------------------------------------
    # ----line 76

    cypher_topic_anywhere = """
    MATCH (e:Entity)-[r1:REL]->(m:Entity)-[r2:REL]->(x:Entity)
    WHERE e.name IS NOT NULL
      AND m.name IS NOT NULL
      AND x.name IS NOT NULL
      AND r1.rname IS NOT NULL
      AND r2.rname IS NOT NULL
      AND ($split IS NULL OR (r1.split = $split AND r2.split = $split))
      AND NOT r1.rname IN $bad_relations
      AND NOT r2.rname IN $bad_relations
      AND (
        toLower(e.name) CONTAINS toLower($kw)
        OR toLower(m.name) CONTAINS toLower($kw)
      )
      AND toLower(e.name) <> toLower(m.name)
      AND toLower(m.name) <> toLower(x.name)
      AND toLower(e.name) <> toLower(x.name) 
    RETURN
      e.name    AS event_name,
      r1.rname  AS relation1_name,
      m.name    AS middle_name,
      r2.rname  AS relation2_name,
      x.name    AS related_name,
      r1.time   AS time1,
      r2.time   AS time2,
      r1.split  AS split1,
      r2.split  AS split2
    ORDER BY time1 DESC, time2 DESC
    LIMIT $limit
    """

    cypher_single_hop = """
    MATCH (e:Entity)-[r:REL]->(x:Entity)
    WHERE e.name IS NOT NULL
      AND x.name IS NOT NULL
      AND r.rname IS NOT NULL
      AND ($split IS NULL OR r.split = $split)
      AND NOT r.rname IN $bad_relations
      AND (
        toLower(e.name) CONTAINS toLower($kw)
        OR toLower(x.name) CONTAINS toLower($kw)
      )
      AND toLower(e.name) <> toLower(x.name)
    RETURN
      e.name   AS event_name,
      r.rname  AS relation1_name,
      x.name   AS related_name,
      r.time   AS time1,
      r.split  AS split
    ORDER BY time1 DESC
    LIMIT $limit
    """

    results: List[Dict[str, Any]] = []
    seen_chains: Set[Tuple[str, str, str, str, str]] = set()
    seen_facts: Set[Tuple[str, str, str]] = set()

    middle_counts: Dict[str, int] = {}
    MAX_PER_MIDDLE = 6

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    try:
        with driver.session() as session:
            retrieval_limit = limit * 10
            records = session.run(
                cypher_topic_anywhere,
                kw=event_keyword.strip(),
                limit=retrieval_limit,
                split=split,
                bad_relations=list(BAD_RELATIONS),
            )

            for record in records:
                event = record["event_name"]
                relation1 = record["relation1_name"]
                middle = record["middle_name"]
                relation2 = record["relation2_name"]
                related = record["related_name"]
                time1 = record["time1"]
                time2 = record["time2"]
                split1 = record["split1"]
                split2 = record["split2"]

                if (
                    _is_junk_entity(event)
                    or _is_junk_entity(middle)
                    or _is_junk_entity(related)
                ):
                    continue

                # hard filters for weak/noisy patterns
                if relation1 == "Impact" and relation2 == "Impact":
                    continue

                if relation1 == "Produce" or relation2 == "Produce":
                    continue

                if middle in BAD_MIDDLES:
                    continue

                if middle == "Donald Trump" and related == "European Central Bank":
                    continue

                middle_counts.setdefault(middle, 0)
                if middle_counts[middle] >= MAX_PER_MIDDLE:
                    continue

                key = (event, relation1, middle, relation2, related)
                if key in seen_chains:
                    continue
                seen_chains.add(key)

                results.append(
                    {
                        "event": event,
                        "relation1": relation1,
                        "middle": middle,
                        "relation2": relation2,
                        "related": related,
                        "time": time1,
                        "time1": time1,
                        "time2": time2,
                        "split": split1 if split1 == split2 else f"{split1}|{split2}",
                        "context_line": (
                            f"{event} --[{relation1} @ t={time1}]--> "
                            f"{middle} --[{relation2} @ t={time2}]--> "
                            f"{related}"
                        ),
                    }
                )

                middle_counts[middle] += 1

                if len(results) >= retrieval_limit:
                    break

            if len(results) < 3:
                fallback_records = session.run(
                    cypher_single_hop,
                    kw=event_keyword.strip(),
                    limit=limit * 4,
                    split=split,
                    bad_relations=list(BAD_RELATIONS),
                )

                for record in fallback_records:
                    event = record["event_name"]
                    relation1 = record["relation1_name"]
                    related = record["related_name"]
                    time1 = record["time1"]
                    rel_split = record["split"]

                    if _is_junk_entity(event) or _is_junk_entity(related):
                        continue

                    if relation1 == "Produce":
                        continue

                    key = (event, relation1, related)
                    if key in seen_facts:
                        continue
                    seen_facts.add(key)

                    results.append(
                        {
                            "event": event,
                            "relation1": relation1,
                            "middle": None,
                            "relation2": None,
                            "related": related,
                            "time": time1,
                            "time1": time1,
                            "time2": None,
                            "split": rel_split,
                            "context_line": f"{event} --[{relation1} @ t={time1}]--> {related}",
                        }
                    )

                    if len(results) >= limit:
                        break

    finally:
        driver.close()

    return results


# Quick standalone test
if __name__ == "__main__":
    ctx = get_context("inflation", limit=10, split="train")

    print("\n--- CLEAN RETRIEVED CONTEXT ---\n")
    for row in ctx:
        print(row["context_line"])