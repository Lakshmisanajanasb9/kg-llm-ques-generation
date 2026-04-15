def build_zero_prompt(context, topic, n=5):
    return f"""
You are given knowledge graph paths about: {topic}

Context:
{context}

Write exactly {n} full-sentence questions based only on the context.

Rules:
- Use only information from the context.
- Each question must be based on one complete 2-hop path.
- Write clear, natural English.
- Use proper grammar.
- Do not use quotation marks.
- Do not use graph notation in the question.
- Do not mention raw time markers such as t=97.
- Do not use vague phrases like:
  "in what way",
  "over the observed period",
  "during this time frame",
  "across the recorded time frame".
- Do not invent extra facts.
- Do not repeat the same question pattern.
- Every question must be a complete sentence ending with one question mark.
Do NOT mention time unless explicitly written as a real date.
Ignore numeric time indicators such as t=97.
- Avoid near-duplicate questions and avoid repeating the same reasoning pattern.
- Ensure each question is answerable from the context.

Output only the questions.
""".strip()