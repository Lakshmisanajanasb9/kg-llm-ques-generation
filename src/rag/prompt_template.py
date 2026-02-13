def build_prompt(event_keyword: str, context_rows: list[dict], n_questions: int = 5) -> str:
    facts = "\n".join(
        [f"- {row.get('event')} --{row.get('relation')}--> {row.get('related')}" for row in context_rows]
    )

    return f"""Task: Generate insightful financial questions.

Rules:
- Use ONLY the facts below.
- Do NOT predict prices.
- Do NOT introduce new entities not in the facts.
- Output exactly {n_questions} questions.
- Each question must start with "Q:".

Topic: {event_keyword}

Facts:
{facts}

Output format (exactly {n_questions} lines):
Q: ...
Q: ...
Q: ...
Q: ...
Q: ...
"""