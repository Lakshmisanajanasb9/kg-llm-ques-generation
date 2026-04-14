def build_prompt(event_keyword: str, context_rows: list[dict], n: int = 5) -> str:

    facts = "\n".join(
        [
            f"- {row['event']} --{row['relation1']}--> {row['middle']} "
            f"--{row['relation2']}--> {row['related']}"
            for row in context_rows
        ]
    )

    return f"""
You are a financial reasoning assistant generating analytical, multi-hop questions from a knowledge graph.

You behave like a financial analyst who:
- identifies causal chains
- interprets relationships
- avoids weak or meaningless reasoning

--------------------------------
CORE TASK
You are given structured paths of the form:

Event --relation1--> Middle --relation2--> Related

Your job is to:
1. Interpret the meaning of each path
2. Convert it into a meaningful real-world mechanism
3. Generate a high-quality analytical question

--------------------------------
STRICT RULES

- Use ONLY entity names from the Facts
- Each question MUST be based on ONE full 2-hop path
- Each question MUST include Event, Middle, and Related
- Do NOT generate single-hop questions
- Do not output sentence fragments, partial questions, notes, or unfinished thoughts.
- Do NOT invent entities
- Do NOT combine information from multiple paths
- If a question contains any entity not present in the path, DO NOT generate it
- Do NOT answer the question
- Output ONLY questions
- Generate up to {n} questions (fewer is OK if some paths are weak)
- Do not include notes, explanations, or commentary after the question
- Skip weak or repetitive chains instead of rephrasing them
- Avoid vague terms like "economic data" — use specific concepts (e.g., inflation, oil prices, stock markets)
- Each question must be phrased differently
- Each generated question must be a complete and grammatically correct sentence that clearly expresses a full idea and ends with a question mark; any question that is incomplete, cut off, repetitive, or lacks clarity should be discarded.
--------------------------------
REASONING STEP (VERY IMPORTANT)

Before writing each question:
- Interpret the path as a real-world relationship
- Identify the mechanism linking Event → Related through Middle
- Use that interpretation to form the question
- Do NOT just rewrite the path

--------------------------------
VALIDITY FILTER (CRITICAL)

- If a path is unclear, unrealistic, or semantically weak → SKIP it
- If the relationship does not make real-world sense → SKIP it
- Prefer fewer strong questions over many weak ones

--------------------------------
STYLE REQUIREMENTS

- Use varied phrasing:
  - "How does..."
  - "In what way does..."
  - "What role does..."
  - "Through what mechanism..."
- Avoid repeating the same structure
- Questions must sound natural and analytical (like economics exam questions)

--------------------------------
DIVERSITY RULES

- Each question must use a DIFFERENT path
- Avoid rewording the same relationship
- Avoid repeating the same middle entity where possible

--------------------------------
FEW-SHOT EXAMPLES

Facts:
- Interest Rate Cuts --Impact--> Inflation --Affect--> Gold Price
- US Federal Reserve --Set--> Interest Rates --Affect--> Borrowing Costs

GOOD:
Q: How do interest rate cuts influence gold prices through their impact on inflation?
Q: How does the US Federal Reserve affect borrowing costs through its control of interest rates?
Q: How does a decrease involving shareholders affect economic growth through the Shanghai Composite Index?

BAD:
Q: What is inflation?
Q: How do interest rate cuts affect inflation?
Q: Interest Rate Cuts --Impact--> Inflation?
Q: How does inflation affect markets and consumer sentiment?

--------------------------------

Topic: {event_keyword}

Facts:
{facts}

Output:
Q:
"""





'''


def build_prompt(event_keyword: str, context_rows: list[dict], n: int = 5) -> str:

    facts = "\n".join(
        [
            f"- {row['event']} --{row['relation1']}--> {row['middle']} "
            f"--{row['relation2']}--> {row['related']}"
            for row in context_rows
        ]
    )

    return f"""
You are a financial reasoning assistant that generates analytical questions
based strictly on relationships in a knowledge graph.

You are a financial analyst generating precise, multi-hop reasoning questions from a structured economic knowledge graph.

You prioritise:
- clear causal reasoning
- economic relevance
- concise phrasing

You avoid:
- vague language
- repetition
- non-economic entities

Before writing the question:
- Interpret what the path implies (cause, influence, transmission)
- Convert the path into a meaningful real-world mechanism
- Then generate the question

STRICT RULES:
- Use ONLY the exact entity names appearing in Facts.
- Each question MUST use one complete path from the Facts.
- Each question MUST be based on at least one full 2-hop chain:
  Event --relation1--> Middle --relation2--> Related
- Each question MUST include the Event, Middle, and Related entities from the same chain.
- Do NOT generate single-hop questions.
- Do NOT skip the middle entity.
- Each question must be based on a DIFFERENT path in the Facts.
- Do NOT reuse the same Event--Middle--Related chain across several questions.
- If possible, do NOT reuse the same middle entity across questions.
- Questions should reflect the relationships in the Facts exactly.
- Do NOT invent entities.
- Do NOT answer the question.
- Do NOT output multiple choice.
- Do NOT ask definition questions.
- Output exactly {n} questions.
- Each line must start with "Q:".


DIVERSITY RULES:
- Each question must involve a different middle or related entity
- Do NOT generate reworded duplicates
- Each question must be meaningfully different
- Avoid duplicate triples or paths 

REASONING STEP (VERY IMPORTANT):
Before generating the question:
- Interpret the path as a real-world relationship
- Identify the mechanism linking Event → Related through Middle
- Use that interpretation to form a meaningful question
- Do NOT just restate the path directly

STYLE VARIATION:
- Use varied phrasing (not always "How does X affect Y through Z?")
- Use forms like:
  - "In what way does..."
  - "What role does..."
  - "How might..."
  - "Through what mechanism..."

NOISE HANDLING:
- If a path is weak, unclear, or not economically meaningful,
  try to interpret it in the most reasonable way possible
- If interpretation is not possible, skip that path

--------------------------------
EXAMPLES (Few-Shot)
Facts Example:
- Interest Rate Cuts --Impact--> Inflation --Affect--> Gold Price
- AI Regulation --Impact--> Tech Industry --Affect--> Innovation
- Ukraine War --Disrupt--> Energy Supply --Increase--> Oil Prices
- US Federal Reserve --Set--> Interest Rates --Affect--> Borrowing Costs
- Inflation --Reduce--> Purchasing Power --Affect--> Consumer Spending
- Climate Change Policies --Influence--> Energy Sector --Affect--> Renewable Investment
- China Economy Slowdown --Impact--> Global Trade --Affect--> Exports
- Currency Depreciation --Increase--> Export Competitiveness --Boost--> Exports
- Government Spending --Stimulate--> Economy --Affect--> Employment
- Oil Prices --Influence--> Inflation --Affect--> Consumer Spending

GOOD Questions:
Q: How do interest rate cuts influence gold prices through their impact on inflation?
Q: How does AI regulation affect innovation through its impact on the tech industry?
Q: How does the Ukraine war influence oil prices through its disruption of energy supply?
Q: How does the US Federal Reserve affect borrowing costs through its control of interest rates?
Q: How does inflation affect consumer spending through its impact on purchasing power?
Q: How do climate change policies influence renewable investment through their impact on the energy sector?
Q: How does a slowdown in the China economy affect exports through its impact on global trade?
Q: How does currency depreciation boost exports through increased export competitiveness?
Q: How does government spending affect employment through its impact on the economy?
Q: How do oil prices affect consumer spending through their influence on inflation?

BAD Questions (Do NOT generate these):
# Definition / trivial
Q: What is inflation?
Q: What is the Ukraine war?
Q: What is AI regulation?

# Not multi-hop (only 1 relationship)
Q: How do interest rate cuts affect inflation?
Q: How does AI regulation impact the tech industry?
Q: How do oil prices influence inflation?

# Not grounded in full chain (ignores second hop)
Q: How does the US Federal Reserve set interest rates?
Q: How does climate change policy affect the energy sector?

# Hallucination (introduces concepts NOT in Facts)
Q: How do interest rate cuts affect consumer sentiment and stock market confidence?
Q: How does the Ukraine war impact global political stability?
Q: How does AI regulation affect startups and venture capital funding?

# Reusing same pathway (rewording)
Q: How do interest rate cuts influence gold prices via inflation?
Q: In what way do interest rate cuts affect gold prices through inflation?
Q: What links interest rate cuts to gold prices through inflation?

# Wrong format / structure
Q: Interest Rate Cuts --Impact--> Inflation?
Q: Name the entities affected by oil prices.
Q: Which of the following relates to consumer spending?

# Not analytical / too shallow
Q: Are oil prices increasing?
Q: Is inflation bad for the economy?


Facts Example:
- Brexit --Impact--> Economy --Operate_In--> United Kingdom

GOOD Questions:
Q: How does Brexit affect economic activity within the United Kingdom through its impact on the economy?
Q: In what way does the economy act as a transmission channel linking Brexit to outcomes in the United Kingdom?

BAD Questions (Do NOT generate these):
Q: What is Brexit?
Q: When did Brexit happen?
Q: Brexit --Impact--> Economy?
Q: How does Brexit affect global politics?
Q: How does Brexit influence consumer confidence?

--------------------------------

Now generate new questions.

Topic: {event_keyword}

Facts:
{facts}

Output:
Q:
"""'''