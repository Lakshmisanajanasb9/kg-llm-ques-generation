def build_zero_prompt(context, topic, n=5):
    return f"""
You are a financial and geopolitical analyst.

You are given structured 2-hop knowledge graph paths about the topic: {topic}.

Context:
{context}

Task:
Generate exactly {n} high-quality natural-language questions using only the context above.

The goal is to turn the structured paths into clear, analytical questions that reflect the relationships between the entities.

Rules:
- Use only entities, relations, and events that appear in the context.
- Each question must be grounded in at least one complete 2-hop path.
- Prefer questions that connect Entity A → Entity B → Entity C, rather than asking about only one isolated relation.
- Write questions in natural English, not graph-style wording.
- Make the questions specific and analytical, not vague or generic.
- Focus on causal, economic, political, or explanatory reasoning where appropriate.
- If time markers appear in the context, express them naturally as "over the observed period", "during this period", or "across the recorded time frame". Do not copy raw forms like "t=97" or invent calendar months/years unless they are explicitly present in the context.
- Do not use placeholders or vague phrases such as:
  "the year mentioned", "the given context", "a specific period", "what event", "in what way", or "how is X related to Y".
- Do not merely restate the KG relation names. Convert them into natural reasoning.
- Do not invent missing details or make unsupported assumptions.
- Avoid near-duplicate questions and avoid repeating the same reasoning pattern.
- Ensure each question is answerable from the context.

Good question style:
- asks about impact, mechanism, consequence, connection, or implication
- combines multiple entities from a path
- sounds natural and precise

Bad question style:
- vague, awkward, repetitive, or copied directly from graph syntax
- unnatural handling of time
- generic factual lookup with little reasoning

Output requirements:
- Output exactly {n} questions
- Number them 1 to {n}
- Output only the questions
"""