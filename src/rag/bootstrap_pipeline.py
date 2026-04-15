import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.generate_questions import (
    clean_questions,
    dedupe_questions_semantic,
    generate_questions,
)
from src.rag.prompt.zero_shot import build_zero_prompt


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def format_path_text(path: dict) -> str:
    return (
        f"{path.get('event')} --{path.get('relation1')}--> {path.get('middle')} "
        f"--{path.get('relation2')}--> {path.get('related')}"
    )


def sbert_grounding_score(question: str, path: dict) -> float:
    path_text = format_path_text(path)

    q_emb = embed_model.encode([question], normalize_embeddings=True)
    p_emb = embed_model.encode([path_text], normalize_embeddings=True)

    return float(cosine_similarity(q_emb, p_emb)[0][0])


def append_bootstrap_rows(csv_path: str, rows: list[dict]) -> None:
    path = Path(csv_path)
    file_exists = path.exists()

    existing_keys = set()
    if file_exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(
                    (
                        row["topic"].strip().lower(),
                        row["path_text"].strip().lower(),
                        row["question"].strip().lower(),
                    )
                )

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topic",
                "path_text",
                "question",
                "grounding_score",
                "pairwise_similarity",
                "final_score",
                "method",
                "accepted",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for row in rows:
            key = (
                row["topic"].strip().lower(),
                row["path_text"].strip().lower(),
                row["question"].strip().lower(),
            )
            if key not in existing_keys:
                writer.writerow(row)
                existing_keys.add(key)


def bootstrap_zero_shot_examples(
    topic: str,
    paths: list[dict],
    csv_path: str = "outputs/bootstrap_examples.csv",
    grounding_fn=None,
    candidates_per_run: int = 2,
    max_examples: int = 5,
):
    rows = []

    for path in paths:
        path_text = format_path_text(path)

        prompt = build_zero_prompt(
            context=path_text,
            topic=topic,
            n=candidates_per_run,
        )

        questions = generate_questions(prompt)
        questions = clean_questions(questions)
        questions = dedupe_questions_semantic(questions, threshold=0.90)

        # keep only the first surviving question per path for cleaner bootstrap
        questions = questions[:1]

        for q in questions:
            score_fn = grounding_fn or sbert_grounding_score
            grounding_score = score_fn(q, path)

            rows.append(
                {
                    "topic": topic,
                    "path_text": path_text,
                    "question": q,
                    "grounding_score": grounding_score,
                    "pairwise_similarity": 0.0,
                    "final_score": grounding_score,
                    "method": "zero_shot",
                    "accepted": grounding_score >= 0.60,
                }
            )

    seen = set()
    unique_rows = []
    for r in rows:
        key = (
            r["topic"].strip().lower(),
            r["path_text"].strip().lower(),
            r["question"].strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    rows = sorted(unique_rows, key=lambda x: x["final_score"], reverse=True)
    rows = rows[:max_examples]

    append_bootstrap_rows(csv_path, rows)
    return rows


def load_top_bootstrap_examples(csv_path: str, topic: str, k: int = 3):
    path = Path(csv_path)
    if not path.exists():
        return []

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["topic"].strip().lower() == topic.strip().lower():
                row["grounding_score"] = float(row["grounding_score"])
                row["pairwise_similarity"] = float(row["pairwise_similarity"])
                row["final_score"] = float(row["final_score"])
                row["accepted"] = str(row["accepted"]).lower() == "true"
                rows.append(row)

    rows = [r for r in rows if r["accepted"]]
    rows.sort(key=lambda x: x["final_score"], reverse=True)
    return rows[:k]