def build_graph_zero_prompt(topic, context_rows, n=5):
    context_text = "\n".join(
        row.get("context_line", str(row)) for row in context_rows
    )

    return f"""
You are given knowledge graph paths related to the topic: {topic}.

Each line represents a path of entities connected by relations.
Each path encodes a potential real-world cause-and-effect chain.

Context (graph paths):
{context_text}

Task:
Generate {n} high-quality, reasoning-based questions grounded in these paths.

Instructions:
- Treat each path as a causal chain of events or influences.
- Each question must follow ONE clear and coherent path.
- Prefer multi-hop reasoning (2–4 steps).
- Only use entities and relations provided in the context.
- Do NOT combine unrelated paths into a single question.


Quality constraints:
- Only generate questions if the chain represents a realistic and meaningful relationship.
- Ignore paths that seem weak, indirect, or implausible.
- Avoid vague terms like "economic data" — use specific concepts (e.g., inflation, oil prices, stock markets).
- Avoid repetition in structure or wording across questions.
- Do NOT use hedging language (e.g., "might", "could", "is it possible").

Goal:
Produce questions that sound like they could appear in an economics or policy analysis discussion.

Output:
Only return a numbered list of questions.
"""