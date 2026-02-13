import json
from pathlib import Path
from src.kg.retrieve_context import get_context

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.rag.prompt_template import build_prompt


#MODEL_NAME = "google/flan-t5-small"
MODEL_NAME = "google/flan-t5-base"


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


def simplify_context(rows):
    simplified = []
    for r in rows:
        simplified.append({
            "event": r["event"]["name"],
            "relation": r["relation"]["name"],
            "related": r["related"]["name"],
            "time": r["time"]
        })
    return simplified


if __name__ == "__main__":
    event = "interest rates"

    raw_rows = get_context(event, limit=25, split="train")
    context_rows = simplify_context(raw_rows)   # ✅ IMPORTANT

    print("\n--- RETRIEVED CONTEXT ---\n")
    for row in context_rows:
        print(row)

    prompt = build_prompt(event, context_rows, n_questions=5)
    questions = generate_questions(prompt)[:5]

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
