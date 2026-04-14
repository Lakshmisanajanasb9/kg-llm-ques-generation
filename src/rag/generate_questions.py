import json
from pathlib import Path
from src.kg.retrieve_context import get_context
from src.rag.prompt.judgePrompt import build_judge_prompt

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.rag.prompt_template import build_prompt
import requests
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import AgglomerativeClustering

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral:7b-instruct"


def generate_from_model(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()

def _norm_text(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def path_to_text(r):
    event = str(r.get("event") or "").strip()
    relation1 = str(r.get("relation1") or "").strip()
    middle = str(r.get("middle") or "").strip()
    relation2 = str(r.get("relation2") or "").strip()
    related = str(r.get("related") or "").strip()

    time1 = r.get("time1")
    time2 = r.get("time2")

    text = f"{event} [{relation1}] {middle} ; {middle} [{relation2}] {related}"

    if time1 is not None or time2 is not None:
        text += f" ; time1={time1}, time2={time2}"

    return text


def cluster_paths(rows, distance_threshold=0.25):
    """
    Group semantically similar KG paths using embedding-based clustering.

    Smaller distance_threshold -> stricter clustering
    Larger distance_threshold -> broader clusters
    """
    if not rows:
        return {}

    if len(rows) == 1:
        return {0: [rows[0]]}

    texts = [path_to_text(r) for r in rows]
    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None
    )

    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for label, row in zip(labels, rows):
        clusters[int(label)].append(row)

    return dict(clusters)


def score_path(r):
    score = 0

    r1 = _norm_text(r.get("relation1"))
    r2 = _norm_text(r.get("relation2"))

    if r1 and r1 != "rel":
        score += 1
    if r2 and r2 != "rel":
        score += 1

    if r.get("time") is not None:
        score += 1
    if r.get("time1") is not None:
        score += 1
    if r.get("time2") is not None:
        score += 1

    event = str(r.get("event") or "").strip()
    middle = str(r.get("middle") or "").strip()
    related = str(r.get("related") or "").strip()

    if event:
        score += 1
    if middle:
        score += 1
    if related:
        score += 1

    return score


def select_cluster_representatives(rows, max_per_cluster=1, distance_threshold=0.25):
    clusters = cluster_paths(rows, distance_threshold=distance_threshold)
    selected = []

    print("\n--- PATH CLUSTERS ---")
    for cluster_id, cluster_rows in clusters.items():
        cluster_rows = sorted(cluster_rows, key=score_path, reverse=True)

        print(f"Cluster {cluster_id} ({len(cluster_rows)} paths)")
        for r in cluster_rows:
            print("  -", path_to_text(r))

        k = min(max_per_cluster, len(cluster_rows))
        selected.extend(cluster_rows[:k])

    return selected


def dedupe_semantic_paths(rows, threshold=0.88):
    if not rows:
        return []

    texts = [path_to_text(r) for r in rows]
    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sim_matrix = cosine_similarity(embeddings)

    groups = []
    assigned = set()

    for i in range(len(rows)):
        if i in assigned:
            continue

        group = [i]
        assigned.add(i)

        for j in range(i + 1, len(rows)):
            if j in assigned:
                continue

            if sim_matrix[i][j] >= threshold:
                group.append(j)
                assigned.add(j)

        groups.append(group)

    filtered = []
    for group in groups:
        best_idx = max(group, key=lambda idx: score_path(rows[idx]))
        filtered.append(rows[best_idx])

    return filtered


def dedupe_paths(rows):
    seen = set()
    unique = []

    for r in rows:
        key = (
            _norm_text(r.get("event")),
            _norm_text(r.get("relation1")),
            _norm_text(r.get("middle")),
            _norm_text(r.get("relation2")),
            _norm_text(r.get("related")),
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def dedupe_questions_semantic(questions, threshold=0.85):
    if not questions:
        return []

    embeddings = embed_model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sim_matrix = cosine_similarity(embeddings)

    selected = []
    used = set()

    for i in range(len(questions)):
        if i in used:
            continue

        selected.append(questions[i])

        for j in range(i + 1, len(questions)):
            if sim_matrix[i][j] >= threshold:
                used.add(j)

    return selected


def limit_middle_reuse(rows, max_per_middle=1):
    counts = defaultdict(int)
    filtered = []

    for r in rows:
        middle = _norm_text(r.get("middle"))
        if not middle:
            continue

        if counts[middle] < max_per_middle:
            filtered.append(r)
            counts[middle] += 1

    return filtered


def dedupe_middle_related(rows):
    seen = set()
    unique = []

    for r in rows:
        middle = _norm_text(r.get("middle"))
        related = _norm_text(r.get("related"))

        if not middle or not related:
            continue

        key = (middle, related)

        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def remove_bad_paths(rows):
    cleaned = []

    for r in rows:
        event = _norm_text(r.get("event"))
        middle = _norm_text(r.get("middle"))
        related = _norm_text(r.get("related"))

        # skip only completely unusable rows
        if not event and not middle and not related:
            continue

        # skip obvious degenerate loops only if both sides exist
        if event and middle and event == middle:
            continue
        if middle and related and middle == related:
            continue
        if event and related and event == related:
            continue

        cleaned.append(r)

    return cleaned


def select_diverse_paths(rows, final_k=8):
    rows = sorted(rows, key=score_path, reverse=True)

    selected = []
    used_middles = set()
    used_pairs = set()

    for r in rows:
        middle = _norm_text(r.get("middle"))
        related = _norm_text(r.get("related"))

        if not middle or not related:
            continue

        pair = (middle, related)

        if middle in used_middles:
            continue

        if pair in used_pairs:
            continue

        selected.append(r)
        used_middles.add(middle)
        used_pairs.add(pair)

        if len(selected) >= final_k:
            break

    return selected


def clean_questions(questions):
    cleaned = []
    seen = set()

    banned_terms = [
        "belarus", "military",
        "template", "wikimedia", "category", "portal"
    ]

    for q in questions:
        q = q.strip()

        if len(q) < 30:
            continue
        if not q.endswith("?"):
            continue

        q_lower = q.lower()

        if any(term in q_lower for term in banned_terms):
            continue

        if q_lower in seen:
            continue

        seen.add(q_lower)
        cleaned.append(q)

    return cleaned


VALID_RELATIONS = {
    "impact", "affect", "influence",
    "cause", "lead", "result",
    "increase", "decrease",
    "positive_impact_on", "negative_impact_on"
}

def filter_strong_relations(rows):
    filtered = []
    for r in rows:
        r1 = _norm_text(r.get("relation1"))
        r2 = _norm_text(r.get("relation2"))

        if r1 in VALID_RELATIONS and r2 in VALID_RELATIONS:
            filtered.append(r)

    return filtered


'''
#MODEL_NAME = "google/flan-t5-small"
#MODEL_NAME = "google/flan-t5-base"
MODEL_NAME = "google/flan-t5-large"
#MODEL_NAME = "google/flan-t5-xl"

# ✅ Load model once (important improvement)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_questions(prompt: str, max_new_tokens: int = 256) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("q:"):
            q = line[2:].strip()
            if not q.endswith("?"):
                q += "?"
            lines.append(q)

    # fallback
    if len(lines) == 0:
        raw = [ln.strip().lstrip("-").strip() for ln in text.splitlines() if ln.strip()]
        for ln in raw:
            if not ln.endswith("?"):
                ln += "?"
            lines.append(ln)

    return lines
'''

MODEL_NAME = "mistral:7b-instruct"  # change if you want qwen2.5:7b etc.
#MODEL_NAME = "llama3:70b-instruct"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def generate_questions(prompt: str, max_new_tokens: int = 256) -> list[str]:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

    payload = {
      "model": MODEL_NAME,
      "prompt": prompt,
      "stream": False,
      "options": {
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.25,
        "num_predict": 256
      }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    text = response.json()["response"].strip()

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("q:"):
            q = line[2:].strip()
            if not q.endswith("?"):
                q += "?"
            lines.append(q)

    # fallback
    if len(lines) == 0:
        raw = [ln.strip().lstrip("-").strip() for ln in text.splitlines() if ln.strip()]
        for ln in raw:
            if not ln.endswith("?"):
                ln += "?"
            lines.append(ln)

    return lines


def generate_one_per_path(topic, rows, n):
    collected = []
    seen = set()

    for path in rows:
        prompt = build_prompt(topic, [path], n=1)
        qs = clean_questions(generate_questions(prompt))

        for q in qs:
            q_norm = q.strip().lower()
            if q_norm not in seen:
                seen.add(q_norm)
                collected.append(q.strip())

        if len(collected) >= n:
            break

    return collected[:n]



def simplify_context(rows):
    simplified = []

    for r in rows:
        simplified.append(
            {
                "event": r.get("event"),
                "relation1": r.get("relation1"),
                "middle": r.get("middle"),
                "relation2": r.get("relation2"),
                "related": r.get("related"),
                "time": r.get("time1"),
                "time1": r.get("time1"),
                "time2": r.get("time2"),
                "split": r.get("split"),
                "context_line": r.get("context_line"),
            }
        )

    return simplified


if __name__ == "__main__":
    event = "Europe"
    n_questions = 10

    raw_rows = get_context(event, limit=200, split="train")
    context_rows = simplify_context(raw_rows)

    context_rows = dedupe_paths(context_rows)
    context_rows = filter_strong_relations(context_rows)
    context_rows = remove_bad_paths(context_rows)
    context_rows = limit_middle_reuse(context_rows, max_per_middle=2)
    context_rows = dedupe_middle_related(context_rows)
    context_rows = dedupe_semantic_paths(context_rows, threshold=0.90)
    context_rows = select_diverse_paths(context_rows, final_k=20)

    context_rows = select_cluster_representatives(
        context_rows,
        max_per_cluster=1,
        distance_threshold=0.30
    )

    print("\n--- RETRIEVED CONTEXT ---\n")
    for row in context_rows:
        print(row)

    prompt = build_prompt(event, context_rows, n_questions=10)

    questions = generate_questions(prompt)

    #questions = build_judge_prompt(questions, context_rows)

    questions = clean_questions(questions)
    questions = dedupe_questions_semantic(questions, threshold=0.90)

    attempts = 0
    while len(questions) < n_questions and attempts < 3:
        more = generate_questions(prompt)
        more = clean_questions(more)
        questions += more
        questions = dedupe_questions_semantic(questions, threshold=0.90)
        attempts += 1
    
    questions = questions[:n_questions]

    print("\n--- QUESTIONS ---\n")
    for q in questions:
        print(f"- {q}")

    Path("outputs/questions").mkdir(parents=True, exist_ok=True)

    out = {
        "topic": event,
        "model": MODEL_NAME,
        "questions": questions,
        "context": context_rows
    }

    Path("outputs/questions/questions_interest.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8"
    )