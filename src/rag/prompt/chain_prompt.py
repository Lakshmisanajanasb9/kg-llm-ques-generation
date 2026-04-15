'''def chain_prompt(context, topic, n=5):
    return f"""
You are a Financial and Geopolitical Analyst specialising in causal reasoning over knowledge graph paths.

Your task is to generate HIGH-QUALITY, ANALYTICAL, and DIVERSE natural-language questions from the given knowledge graph paths.

TOPIC: {topic}

INPUT PATHS:
{context}

Each path follows:
Entity_A → Relation_1 → Entity_B → Relation_2 → Entity_C

--------------------------------------------------
STRICT OUTPUT RULES (MANDATORY)
--------------------------------------------------

- Generate EXACTLY {n} questions
- Number them 1 to {n} (no repeats, no resets)
- Each question must be COMPLETE (no cut-offs)
- Do NOT repeat questions or ideas
- Do NOT invent entities — only use those in the paths

--------------------------------------------------
DIVERSITY CONSTRAINTS (VERY IMPORTANT)
--------------------------------------------------

Each question MUST follow a DIFFERENT reasoning style.

Across all questions, include a mix of:
- Causal reasoning (cause → effect)
- Conditional reasoning (under what conditions)
- Comparative reasoning (A vs B)
- Temporal reasoning (before/after, change over time)
- Multi-entity interaction (A affects B affects C)
- Evaluative reasoning (why is this significant)

DO NOT:
- Start more than 2 questions with the same phrase
- Reuse phrases like:
  "How does", "In what way", "Through what mechanism"
- Overuse words like:
  "impact", "affect", "influence", "contribute"

--------------------------------------------------
QUALITY CONSTRAINTS
--------------------------------------------------

- Ensure each question reflects a REALISTIC and LOGICAL relationship
- Avoid weak or forced links between unrelated entities
- Prefer economically or geopolitically meaningful reasoning
- Each question must clearly use a DIFFERENT path or idea

--------------------------------------------------
INTERNAL REASONING (DO NOT OUTPUT)
--------------------------------------------------

For each path:
1. Identify Entity_A, Entity_B, Entity_C
2. Interpret relations into real-world meaning
3. Form a coherent reasoning chain
4. Discard paths that are vague, generic, or nonsensical
5. Ensure selected questions are diverse in structure and content

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Q1. ...
Q2. ...
...
Q{n}. ...
"""'''

def build_cot_prompt(topic: str, context_rows: list[dict], n: int = 1) -> str:
    facts = "\n".join(
        f"- {row['event']} --{row['relation1']}--> {row['middle']} --{row['relation2']}--> {row['related']}"
        for row in context_rows
    )

    return f"""
You are a financial and geopolitical reasoning assistant.

You are given EXACTLY ONE 2-hop knowledge graph path about the topic: {topic}.

The path has the form:

Event --relation1--> Middle --relation2--> Related

Your task is to reason carefully over the path and generate up to {n} high-quality question(s).

==================================================
INTERNAL REASONING PROCESS
Do not output these steps. Use them silently.
==================================================

Step 1: Extract the path
- Identify the Event, Middle, and Related entities exactly as written.
- Identify relation1 and relation2 exactly as written.

Step 2: Classify the path type
Decide whether the path is mainly:
- CAUSAL / ANALYTICAL
  Examples: impact, affect, influence, increase, decrease, positive_impact_on, negative_impact_on, cause, lead to, result in
- STRUCTURAL / RELATIONAL
  Examples: owned by, owner of, part of, has part(s), follows, followed by, child organization or unit, country, country of citizenship, stock exchange, member of, official religion, studied by, central bank, executive body

Step 3: Interpret the chain
- For causal paths: identify the mechanism linking Event → Middle → Related
- For structural paths: identify the relationship linking Event → Middle → Related
- Preserve the original direction of the path
- Do not reverse the chain

Step 4: Check grounding
Before writing the question, verify:
- the question uses only entities from the path
- the question reflects BOTH hops, not just one
- the question does not invent extra causes, consequences, or entities
- the question type matches the relation type

Step 5: Generate the question
- If the path is causal, write an analytical question
- If the path is structural, write a structural or relational question
- The question must be natural, complete, and grammatically correct
- The question must end with exactly one question mark

==================================================
STRICT RULES
==================================================

- Use ONLY the entities in the path.
- Use Event, Middle, and Related from the SAME path.
- Do NOT combine multiple paths.
- Do NOT invent entities, events, policies, outcomes, dates, or mechanisms.
- Do NOT output reasoning steps, notes, or explanations.
- Do NOT use quotation marks.
- Do NOT output graph notation.
- Do NOT produce fragments or truncated questions.
- If the path is unclear, weak, or semantically invalid, output NOTHING.
- Prefer no question over a bad question.

CRITICAL CONSTRAINT:
Each question MUST be based on exactly ONE 2-hop path.

Do NOT combine:
- multiple middle entities
- multiple related entities
- concepts from different paths

If more than one path is used, the question is invalid and must not be generated.

==================================================
QUESTION STYLE RULES
==================================================

For CAUSAL / ANALYTICAL paths, acceptable styles include:
- How does X affect Y through Z?
- In what way does X influence Y through Z?
- Through what mechanism does X lead to Y through Z?
- What role does X play in shaping Y through Z?
- How might X contribute to Y through Z?

For STRUCTURAL / RELATIONAL paths, acceptable styles include:
- How is X connected to Y through Z?
- What is the relationship between X and Y through Z?
- In what way is X related to Y through Z?
- What is the nature of the connection between X and Y through Z?
- How is X structurally linked to Y through Z?

For STRUCTURAL paths, do NOT use:
- cause
- influence
- impact
- affect
- lead to
- result in
- effect
- decision-making
- market volatility
unless those are explicitly supported by the path.

==================================================
GOOD EXAMPLES
==================================================

Example 1: causal
Path:
- Interest Rate Cuts --impact--> Inflation --affect--> Gold Price
Good question:
Q: How do interest rate cuts influence gold prices through their impact on inflation?

Example 2: causal
Path:
- Inflation --positive_impact_on--> Consumer Spending --impact--> The U.S. Economy
Good question:
Q: In what way does inflation affect the U.S. economy through its positive impact on consumer spending?

Example 3: structural
Path:
- JPMorgan Chase --owned by--> BlackRock --owned by--> Kuwait Investment Authority
Good question:
Q: How is JPMorgan Chase connected to Kuwait Investment Authority through BlackRock?

Example 4: structural
Path:
- Trump administration family separation policy --part of--> immigration policy of the first Donald Trump administration --has part(s)--> Executive Order 13769
Good question:
Q: How is the Trump administration family separation policy related to Executive Order 13769 through the immigration policy of the first Donald Trump administration?

Example 5: structural
Path:
- inflation --studied by--> financial economics --has part(s)--> financial market
Good question:
Q: How is inflation connected to financial markets through its study by financial economics?

==================================================
BAD EXAMPLES
==================================================

Bad:
Q: What is inflation?
Reason: definition question, not multi-hop.

Bad:
Q: How do interest rate cuts affect inflation?
Reason: only uses one hop.

Bad:
Q: How does BlackRock's ownership cause Kuwait Investment Authority to influence JPMorgan Chase?
Reason: invents causality from structural ownership links.

Bad:
Q: Through what mechanism does inflation studied by macroeconomics affect general economics?
Reason: unnatural and misreads the path.

Bad:
Q: Why does JPMorgan Chase's country affiliation with the United States affect lending practices through the Federal Reserve System?
Reason: adds unsupported mechanism.



GOOD:


==================================================
TOPIC
==================================================

{topic}

==================================================
PATH
==================================================

{facts}

==================================================
OUTPUT
==================================================

Q:
""".strip()