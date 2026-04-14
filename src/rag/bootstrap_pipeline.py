import csv
import os
import re
from datetime import datetime
from typing import Callable, Dict, List, Optional


# =========================================================
# Text / parsing helpers
# =========================================================

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_question(text: str) -> str:
    text = normalize_space(text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return text


def parse_questions(raw_text: str) -> List[str]:
    """
    Extract questions from model output.

    Accepts lines such as:
      Q: ...
      1. ...
      - ...
    """
    if not raw_text:
        return []

    questions: List[str] = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^(Q\s*[:.\-]\s*)", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^\d+\s*[.)-]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()

        if not line:
            continue

        if "?" in line:
            line = line[: line.rfind("?") + 1].strip()

        if line.endswith("?"):
            questions.append(line)

    return questions

def ensure_questions(output) -> List[str]:
    """
    Accept either:
    - raw model text (str)
    - list of questions (List[str])

    Return a clean list of question strings.
    """
    if output is None:
        return []

    if isinstance(output, list):
        return [str(x).strip() for x in output if str(x).strip()]

    if isinstance(output, str):
        return parse_questions(output)

    return []


def dedupe_questions(questions: List[str]) -> List[str]:
    seen = set()
    output = []

    for q in questions:
        norm = normalize_question(q)
        if norm in seen:
            continue
        seen.add(norm)
        output.append(q)

    return output


def filter_questions(questions: List[str]) -> List[str]:
    """
    Remove weak / malformed questions.
    """
    filtered = []
    seen = set()

    bad_starts = (
        "is ", "are ", "do ", "does ", "did ",
        "can ", "could ", "will ", "would "
    )

    for q in questions:
        q = normalize_space(q)

        if not q:
            continue
        if not q.endswith("?"):
            continue
        if len(q) < 25:
            continue
        if len(q.split()) < 6:
            continue
        if q.lower().startswith(bad_starts):
            continue

        norm = normalize_question(q)
        if norm in seen:
            continue
        seen.add(norm)
        filtered.append(q)

    return filtered


# =========================================================
# Style detection and scoring
# =========================================================

def detect_question_style(question: str) -> str:
    """
    Label a question as one of:
    - causal
    - mechanism
    - consequence
    - other
    """
    q = question.lower()

    mechanism_patterns = [
        "through what mechanism",
        "by what mechanism",
        "via what process",
        "through which process",
        "what process",
        "what channel",
        "by which channel",
        "how exactly",
    ]

    consequence_patterns = [
        "what broader",
        "what consequence",
        "what consequences",
        "what implications",
        "what effect",
        "what effects",
        "what outcome",
        "what outcomes",
        "what might happen",
    ]

    causal_patterns = [
        "how did",
        "how does",
        "how might",
        "how could",
        "in what way",
        "to what extent",
        "what role",
        "influence",
        "affect",
        "impact",
        "contribute",
        "lead to",
        "result in",
    ]

    for p in mechanism_patterns:
        if p in q:
            return "mechanism"

    for p in consequence_patterns:
        if p in q:
            return "consequence"

    for p in causal_patterns:
        if p in q:
            return "causal"

    return "other"


def score_question(question: str, topic: str = "") -> int:
    q = question.strip()
    q_low = q.lower()
    score = 0

    if q.endswith("?"):
        score += 2
    if len(q.split()) >= 8:
        score += 2
    if 10 <= len(q.split()) <= 24:
        score += 2

    good_terms = [
        "impact", "influence", "effect", "consequence", "mechanism",
        "contribute", "role", "implication", "outcome", "how", "why", "what"
    ]
    for term in good_terms:
        if term in q_low:
            score += 1

    bad_terms = [
        "thing", "stuff", "connected", "related", "linked"
    ]
    for term in bad_terms:
        if term in q_low:
            score -= 2

    if topic and topic.lower() in q_low:
        score += 1

    return score


def select_bootstrap_examples(
    zero_questions: List[str],
    topic: str,
    fallback_examples: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Select one strong example for each style bucket:
    causal, mechanism, consequence.
    """
    if fallback_examples is None:
        fallback_examples = {
            "causal": "How does the initial relationship in the path influence the final entity in the chain?",
            "mechanism": "Through what mechanism does the intermediate entity shape the effect of the first entity on the final entity?",
            "consequence": "What broader economic or geopolitical consequence could emerge from this two-hop relationship?",
        }

    zero_questions = dedupe_questions(zero_questions)

    buckets: Dict[str, List[str]] = {
        "causal": [],
        "mechanism": [],
        "consequence": [],
        "other": [],
    }

    for q in zero_questions:
        style = detect_question_style(q)
        buckets[style].append(q)

    selected: List[str] = []

    for style in ["causal", "mechanism", "consequence"]:
        candidates = buckets[style]
        if candidates:
            best = sorted(
                candidates,
                key=lambda x: score_question(x, topic),
                reverse=True,
            )[0]
            selected.append(best)
        else:
            selected.append(fallback_examples[style])

    return selected


# =========================================================
# Prompt builders
# =========================================================

def build_zero_shot_prompt(context: str, topic: str, n: int = 10) -> str:
    return f"""
You are a financial and geopolitical analyst.

Your task is to generate analytical natural-language questions from knowledge graph paths.

TOPIC: {topic}

KNOWLEDGE GRAPH PATHS:
{context}

Each path follows:
Entity_A -> Relation_1 -> Entity_B -> Relation_2 -> Entity_C

Rules:
- Use only entities and relationships grounded in the paths
- Generate analytical, specific, and meaningful questions
- Avoid vague, generic, or repetitive wording
- Avoid yes/no questions
- Do not invent entities or facts
- Each question must be complete and end with a question mark
- Output exactly {n} questions
- Prefix each line with "Q:"
""".strip()


def build_few_shot_prompt(
    context: str,
    topic: str,
    examples: List[str],
    n: int = 5,
) -> str:
    example_block = "\n".join(
        [f"Example {i + 1}: {q}" for i, q in enumerate(examples)]
    )

    return f"""
You are a financial and geopolitical analyst.

Your task is to generate high-quality analytical natural-language questions from knowledge graph paths.

TOPIC: {topic}

Here are examples of strong questions with different reasoning styles:
{example_block}

Now generate {n} NEW questions from the knowledge graph paths below.

KNOWLEDGE GRAPH PATHS:
{context}

Each path follows:
Entity_A -> Relation_1 -> Entity_B -> Relation_2 -> Entity_C

Rules:
- Follow the style of the examples
- Use only entities and relationships grounded in the paths
- Make questions analytical, specific, and coherent
- Avoid duplicates or near-duplicates
- Avoid yes/no questions
- Do not copy the examples directly
- Do not invent entities or facts
- Each question must end with a question mark
- Output exactly {n} questions
- Prefix each line with "Q:"
""".strip()


# =========================================================
# CSV logging
# =========================================================

def save_bootstrap_run(
    topic: str,
    context: str,
    result: Dict[str, object],
    filename: str = "bootstrap_runs.csv",
) -> None:
    """
    Append one row per run.
    """
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "method",
                "topic",
                "context",
                "zero_questions",
                "examples_used",
                "final_questions",
            ])

        writer.writerow([
            datetime.now().isoformat(),
            "bootstrapped_zero_to_few",
            topic,
            context,
            " || ".join(result.get("zero_questions", [])),
            " || ".join(result.get("examples_used", [])),
            " || ".join(result.get("final_questions", [])),
        ])


# =========================================================
# Main pipeline
# =========================================================

def generate_bootstrap_pipeline(
    context: str,
    topic: str,
    llm_generate_fn: Callable[[str], str],
    n_zero: int = 10,
    n_final: int = 5,
    csv_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Full pipeline:
    1. Zero-shot generation
    2. Parse + filter
    3. Select diverse examples
    4. Few-shot generation
    5. Parse + filter
    6. Optionally log to CSV
    """

    # --------------------------
    # Stage 1: zero-shot
    # --------------------------
    zero_prompt = build_zero_shot_prompt(context=context, topic=topic, n=n_zero)
    zero_raw = llm_generate_fn(zero_prompt)
    zero_questions = ensure_questions(zero_raw)
    zero_questions = filter_questions(zero_questions)

    # --------------------------
    # Example selection
    # --------------------------
    examples_used = select_bootstrap_examples(
        zero_questions=zero_questions,
        topic=topic,
    )

    # --------------------------
    # Stage 2: few-shot
    # --------------------------

    
    few_prompt = build_few_shot_prompt(
        context=context,
        topic=topic,
        examples=examples_used,
        n=n_final,
    )

    few_raw = llm_generate_fn(few_prompt)
    final_questions = ensure_questions(few_raw)
    final_questions = filter_questions(final_questions)

    result: Dict[str, object] = {
        "zero_prompt": zero_prompt,
        "zero_raw": zero_raw,
        "zero_questions": zero_questions,
        "examples_used": examples_used,
        "few_prompt": few_prompt,
        "few_raw": few_raw,
        "final_questions": final_questions,
    }

    if csv_path:
        save_bootstrap_run(
            topic=topic,
            context=context,
            result=result,
            filename=csv_path,
        )
    print("\n===== BOOTSTRAP DEBUG =====")
    print("ZERO QUESTIONS:")
    for q in zero_questions:
        print("-", q)

    print("\nEXAMPLES SELECTED:")
    for q in examples_used:
        print("-", q)
    print("\nFINAL QUESTIONS:")
    for q in final_questions:
        print("-", q)

    return result


def generate_final_questions_only(
    context: str,
    topic: str,
    llm_generate_fn: Callable[[str], str],
    n: int = 5,
    csv_path: Optional[str] = None,
) -> List[str]:
    """
    Convenience wrapper when you only want the final questions.
    """
    result = generate_bootstrap_pipeline(
        context=context,
        topic=topic,
        llm_generate_fn=llm_generate_fn,
        n_zero=max(10, n * 2),
        n_final=n,
        csv_path=csv_path,
    )
    return result["final_questions"]