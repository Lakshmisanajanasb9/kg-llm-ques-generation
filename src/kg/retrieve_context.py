from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

URI = "neo4j://localhost:7687"   # local Neo4j Desktop
USER = "neo4j"
PASSWORD = "sai.sanju9"           # <-- change this


def get_context(
    event_keyword: str,
    limit: int = 25,
    split: Optional[str] = None,   # "train" / "valid" / "test" or None
) -> List[Dict[str, Any]]:
    """
    Returns small, readable KG context for an event/topic keyword.

    Works with your FinDKG import where:
      - nodes are (:Entity {id, name, ...})
      - edges are (:REL {rid, rname, time, split, ...})
    """
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    cypher = """
    MATCH (e:Entity)-[r:REL]->(x:Entity)
    WHERE e.name IS NOT NULL
      AND x.name IS NOT NULL
      AND toLower(e.name) CONTAINS toLower($kw)
      AND ($split IS NULL OR r.split = $split)
    RETURN
      e.id   AS event_id,
      e.name AS event_name,
      r.rid  AS relation_id,
      r.rname AS relation_name,
      x.id   AS related_id,
      x.name AS related_name,
      r.time AS time,
      r.split AS split
    LIMIT $limit
    """

    results: List[Dict[str, Any]] = []

    with driver.session() as session:
        records = session.run(
            cypher,
            kw=event_keyword,
            limit=limit,
            split=split
        )
        for record in records:
            results.append(
                {
                    "event": {
                        "id": record["event_id"],
                        "name": record["event_name"],
                    },
                    "relation": {
                        "id": record["relation_id"],
                        "name": record["relation_name"],
                    },
                    "related": {
                        "id": record["related_id"],
                        "name": record["related_name"],
                    },
                    "time": record["time"],
                    "split": record["split"],
                }
            )

    driver.close()
    return results


if __name__ == "__main__":
    # quick test
    ctx = get_context("interest", limit=10, split="train")
    for row in ctx:
        print(row)
