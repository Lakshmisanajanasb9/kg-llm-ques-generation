def build_advanced_cot_prompt(context, topic, n=5):
    return f"""
You are a Financial and Geopolitical Analyst specialising in causal reasoning over knowledge graph paths.

Your task is to transform retrieved KG paths into clear, natural, and strictly grounded analytical questions.

TOPIC: {topic}

INPUT PATHS:
{context}

Each path follows:
Entity_A → Relation_1 → Entity_B → Relation_2 → Entity_C

--------------------------------------------------
INTERNAL REASONING FRAMEWORK (DO NOT OUTPUT)
--------------------------------------------------

For each path:

1. ENTITY EXTRACTION
- Identify Entity_A, Entity_B, and Entity_C exactly as written.

2. RELATION INTERPRETATION
- Translate Relation_1 and Relation_2 into simple real-world meaning (e.g., influence, impact, participation).
- Do NOT exaggerate or strengthen the meaning.

3. CHAIN COMPOSITION
- Form the full chain: Entity_A → Entity_B → Entity_C
- Ensure BOTH steps are preserved.

4. VALIDITY CHECK
- Ensure the chain is meaningful and not trivial.
- If weak or unclear, skip the path.

5. QUESTION CONSTRUCTION
- Convert the chain into ONE clear, natural question
- The question must reflect BOTH hops
- The question must remain strictly grounded in the path

6. DIVERSITY CONTROL
- Avoid repeating the same structure across questions
- Use different phrasing styles where possible

--------------------------------------------------
STRICT GROUNDING RULES
--------------------------------------------------

1. Use the exact strings for Entity_A, Entity_B, and Entity_C.
2. Do NOT introduce new entities, facts, interpretations, or assumptions.
3. Do NOT add meaning not explicitly present in the relations.
4. Do NOT reinterpret entities (e.g., person → "financial performance").
5. Do NOT assume direction (positive/negative) unless explicitly stated.
6. Do NOT add concepts like "growth", "profitability", "savings", "instability" unless present.

--------------------------------------------------
TEMPORAL RULES
--------------------------------------------------

- Do NOT output raw time indices (e.g., t=82, Q94).
- Do NOT convert indices into real calendar dates.
- If time is needed, use natural phrasing such as:
  "over the observed period" or "during the recorded period".

--------------------------------------------------
SIMPLICITY & STYLE RULES
--------------------------------------------------

- Use simple, direct question phrasing.
- Prefer natural forms like:
  "How did X affect Y?"
  "How does X influence Y?"
- Avoid complex or academic phrasing such as:
  "Through what mechanism"
  "To what extent"
  "In what way"
- Keep questions concise and clear.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

- Return ONLY the final questions.
- One question per line.
- Do NOT include numbering.
- Each question must be complete and end with a question mark.
- Generate up to {n} questions.
"""