def build_dsp_prompt(context, topic, perspective="economic", n=5):
    return f"""
You are a Financial and Geopolitical Analyst.

Your task is to generate analytical questions from structured 2-hop knowledge graph paths.

TOPIC: {topic}

PERSPECTIVE:
You must adopt a {perspective} perspective when framing the questions.
However, you must remain strictly grounded in the given paths.

INPUT PATHS:
{context}

Each path follows:
Entity_A -> Relation_1 -> Entity_B -> Relation_2 -> Entity_C

--------------------------------------------------
DIRECTIONAL STIMULUS
--------------------------------------------------

Frame each question through a {perspective} lens, but ONLY using information already present in the path.

This means:
- Emphasise the {perspective}-relevant significance of the path
- Do NOT add new outcomes, sectors, actors, mechanisms, dates, or interpretations
- Do NOT infer stronger claims than the path supports
- Do NOT replace a neutral relation with a stronger one unless explicitly stated

Perspective guidance:

IF economic:
- Focus on economic meaning already present in the path, such as inflation, spending, earnings, markets, trade, rates, or economic effects
- Do NOT invent outcomes like profitability, financial performance, savings behaviour, growth, or recession unless explicitly present

IF geopolitical:
- Focus on geopolitical meaning already present in the path, such as countries, leaders, agreements, institutions, or international effects
- Do NOT invent conflict, diplomacy, alliances, or strategic motives unless explicitly present

IF policy:
- Focus on policy or institutional meaning already present in the path, such as regulation, governance, central bank actions, or official decisions
- Do NOT invent regulatory consequences unless explicitly present

IF risk:
- Focus on uncertainty, exposure, or negative consequences only if supported by the path
- Do NOT assume instability or losses unless explicitly present

--------------------------------------------------
TASK
--------------------------------------------------

For each valid path:
1. Read the full 2-hop chain from Entity_A to Entity_C through Entity_B
2. Preserve the meaning of BOTH hops
3. Generate ONE natural, complete question based on that single path
4. Make the question analytical, but strictly grounded

--------------------------------------------------
STRICT RULES
--------------------------------------------------

- Use only entities and relations present in the input
- Use exactly one full path per question
- Reflect BOTH hops: Entity_A -> Entity_B -> Entity_C
- Do NOT skip or collapse the middle entity
- Use exact entity names
- Convert graph relations into natural English, but keep their original meaning
- Do NOT introduce new facts, concepts, comparisons, mechanisms, or assumptions
- Do NOT invent positive or negative direction unless explicitly present in the relation
- Do NOT reinterpret a person, company, or country into a different concept such as "financial performance" or "investor confidence"
- Do NOT ask vague questions like:
  "What is the connection between X and Y?"
  "In what way is X related to Y?"
  "What broader implications does this have?"
- Avoid duplicates and near-duplicates
- If a path is weak, unclear, or unnatural, skip it
- It is better to return fewer strong questions than weak ones
- Do not convert temporal indices or abstract time markers into real calendar years unless those years are explicitly given in the input path.
- If the path contains only abstract time markers, refer to time naturally as "over the observed period", "during the recorded period", or "across the time frame shown in the path".

--------------------------------------------------
QUESTION STYLE
--------------------------------------------------

A strong question:
- is clear, natural, and specific
- sounds like a realistic analytical question
- preserves the full path structure
- highlights the {perspective} angle without inventing content

A weak question:
- is vague, repetitive, or overly broad
- adds unsupported interpretation
- ignores the middle node
- restates the path awkwardly

--------------------------------------------------
OUTPUT
--------------------------------------------------

Return ONLY the questions, one per line.
Do NOT number them.
Each line must be one complete question ending with a question mark.
Generate up to {n} questions.
"""