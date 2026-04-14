from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "sai.sanju9"   # change if needed

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def clear_wiki_subgraph():
    with driver.session() as session:
        session.run("MATCH (n:WikiEntity) DETACH DELETE n")


def insert_wiki_rows(rows):
    with driver.session() as session:
        for r in rows:
            session.run("""
                MERGE (a:WikiEntity {name: $event})
                MERGE (b:WikiEntity {name: $middle})
                MERGE (c:WikiEntity {name: $related})

                MERGE (a)-[:REL {type: $r1}]->(b)
                MERGE (b)-[:REL {type: $r2}]->(c)
            """, {
                "event": r["event"],
                "middle": r["middle"],
                "related": r["related"],
                "r1": r["relation1"],
                "r2": r["relation2"]
            })


def query_wiki_paths(limit=20):
    with driver.session() as session:
        result = session.run(f"""
            MATCH (a:WikiEntity)-[r1]->(b:WikiEntity)-[r2]->(c:WikiEntity)
            RETURN a.name AS event,
                   r1.type AS relation1,
                   b.name AS middle,
                   r2.type AS relation2,
                   c.name AS related
            LIMIT {limit}
        """)

        return [dict(record) for record in result]