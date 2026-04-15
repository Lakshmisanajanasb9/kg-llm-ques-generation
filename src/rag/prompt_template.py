def build_prompt(event_keyword: str, context_rows: list[dict], n: int = 1) -> str:
    facts = "\n".join(
        f"- {row['event']} --{row['relation1']}--> {row['middle']} --{row['relation2']}--> {row['related']}"
        for row in context_rows
    )

    return f"""
You are a financial and geopolitical reasoning assistant.

You are given EXACTLY ONE knowledge graph path about the topic: {event_keyword}.

The path has this form:

Event --relation1--> Middle --relation2--> Related

Your task is to generate up to {n} high-quality question(s) based ONLY on this single path.

--------------------------------------------------
CORE OBJECTIVE
--------------------------------------------------

You must interpret the path correctly and then generate a question that matches the type of relationship in the path.

There are TWO main path types:

1. CAUSAL / ANALYTICAL PATHS
These involve relations like:
- impact
- affect
- influence
- increase
- decrease
- positive_impact_on
- negative_impact_on
- cause
- lead to
- result in

For these, ask an analytical question about mechanism, effect, consequence, or transmission.

2. STRUCTURAL / RELATIONAL PATHS
These involve relations like:
- owned by
- owner of
- part of
- has part(s)
- follows
- followed by
- child organization or unit
- country
- country of citizenship
- stock exchange
- member of
- official religion
- applies to jurisdiction
- headquarters location
- central bank
- executive body
- legislative body

For these, ask a structural or relational question.
Do NOT turn them into strong causal claims.

--------------------------------------------------
STRICT RULES
--------------------------------------------------

- Use ONLY the exact entity names in the path.
- Use Event, Middle, and Related from the SAME path.
- Do NOT invent new entities, events, institutions, causes, or effects.
- Do NOT combine multiple paths.
- Do NOT answer the question.
- Do NOT output explanations, bullets, notes, labels, or commentary.
- Do NOT use quotation marks.
- Output only complete question sentences.
- Every output must end with exactly one question mark.
- If the path is weak, unclear, semantically broken, or unrealistic, output NOTHING.
- Prefer no question over a bad question.

CRITICAL:
The question MUST explicitly reflect BOTH:
- Event → Middle
- Middle → Related

If the second relation is missing in the question → DO NOT generate it.

Do NOT reverse the direction of the path.
The causal or structural flow must follow:

Event → Middle → Related

Do not make Middle the cause of Event.
Do not make Related the cause of Event.

--------------------------------------------------
VERY IMPORTANT INTERPRETATION RULE
--------------------------------------------------

Do NOT blindly convert the path into a causal story.

If the path is STRUCTURAL, your question must remain structural.

That means:
- ask how two entities are connected
- ask what relationship exists
- ask how one entity is linked to another through the middle entity
- ask about institutional, ownership, geographic, membership, or classification relationships

Do NOT use causal language for structural paths.

Avoid words like:
- cause
- influence
- impact
- affect
- lead to
- result in
- effect
- decision-making
- market volatility
- policy outcome

unless the path itself clearly supports causality.

--------------------------------------------------
QUESTION STYLE GUIDANCE
--------------------------------------------------

For CAUSAL paths, acceptable styles include:
- How does...
- In what way does...
- Through what mechanism does...
- What effect does...
- To what extent does...
- What role does...

For STRUCTURAL paths, acceptable styles include:
- How is X related to Y through Z?
- What is the relationship between X and Y through Z?
- In what way is X connected to Y through Z?
- How is X structurally linked to Y through Z?
- What is the nature of the connection between X and Y through Z?
- How is X associated with Y via Z?

For STRUCTURAL paths, prefer:
- related to
- connected to
- linked to
- associated with
- structurally related to
- part of
- ownership relationship
- institutional relationship
- geographic relationship

--------------------------------------------------
GOOD STRUCTURAL EXAMPLES
--------------------------------------------------

Path:
JPMorgan Chase --owned by--> BlackRock --owned by--> Kuwait Investment Authority
GOOD:
Q: How is JPMorgan Chase structurally related to Kuwait Investment Authority through BlackRock?

Path:
JPMorgan Chase --owned by--> The Vanguard Group --owner of--> Mitsubishi UFJ Financial Group
GOOD:
Q: In what way is JPMorgan Chase connected to Mitsubishi UFJ Financial Group through The Vanguard Group?

Path:
JPMorgan Chase --part of--> Dow Jones Industrial Average --stock exchange--> New York Stock Exchange
GOOD:
Q: How is JPMorgan Chase linked to the New York Stock Exchange through its inclusion in the Dow Jones Industrial Average?

Path:
Trump administration family separation policy --part of--> immigration policy of the first Donald Trump administration --has part(s)--> Executive Order 13769
GOOD:
Q: How is the Trump administration family separation policy related to Executive Order 13769 through the immigration policy of the first Donald Trump administration?

Path:
Iran --member of--> United Nations --has part(s)--> United Nations Economic and Social Council
GOOD:
Q: How is Iran connected to the United Nations Economic and Social Council through its membership in the United Nations?

Path:
Iran --official religion--> Islam --has part(s)--> Shahada
GOOD:
Q: How is the Shahada related to Islam, the official religion of Iran?

Path:
BlackRock --stock exchange--> New York Stock Exchange --significant event--> Black Monday
GOOD:
Q: How is BlackRock connected to Black Monday through the New York Stock Exchange?

Path:
JPMorgan Chase --country--> United States --central bank--> Federal Reserve System
GOOD:
Q: How is JPMorgan Chase connected to the Federal Reserve System through its association with the United States?

--------------------------------------------------
BAD STRUCTURAL EXAMPLES
--------------------------------------------------

BAD:
Q: How does BlackRock's ownership cause Kuwait Investment Authority to influence JPMorgan Chase?
Why bad: invents causality from ownership.

BAD:
Q: Through what mechanism does Iran's membership in the United Nations influence its decision-making power in ECOSOC?
Why bad: invents causal influence from membership.

BAD:
Q: What effect does Islam have on recitation of Shahada?
Why bad: turns a structural religious relation into causal impact.

BAD:
Q: Why does JPMorgan Chase's country affiliation with the United States affect lending practices through the Federal Reserve System?
Why bad: adds economic mechanism not present in the path.

--------------------------------------------------
GOOD CAUSAL EXAMPLES
--------------------------------------------------

Path:
Interest Rate Cuts --impact--> Inflation --affect--> Gold Price
GOOD:
Q: How do interest rate cuts influence gold prices through their impact on inflation?

Path:
Germany --impact--> Inflation --positive_impact_on--> Consumer Spending
GOOD:
Q: How does Germany's influence on inflation contribute to consumer spending?

Path:
US Federal Reserve --set--> Interest Rates --affect--> Borrowing Costs
GOOD:
Q: How does the US Federal Reserve affect borrowing costs through its control of interest rates?

Path:
Inflation --positive_impact_on--> Consumer Spending --impact--> The U.S. Economy
GOOD:
Q: In what way does inflation affect the U.S. economy through its positive impact on consumer spending?

--------------------------------------------------
BAD CAUSAL EXAMPLES
--------------------------------------------------

BAD:
Q: What is inflation?
Why bad: definition question, not multi-hop.

BAD:
Q: How do interest rate cuts affect inflation?
Why bad: single-hop only.

BAD:
Q: How does inflation affect markets and consumer sentiment?
Why bad: introduces entities not in the path.

BAD:
Q: Interest Rate Cuts --impact--> Inflation?
Why bad: not a natural-language analytical question.

--------------------------------------------------
FINAL INSTRUCTION
--------------------------------------------------

Now read the single path below and decide whether it is:
- CAUSAL / ANALYTICAL
or
- STRUCTURAL / RELATIONAL

Then generate up to {n} question(s) in the correct style.

Topic: {event_keyword}

Path:
{facts}

Output:
Q:
""".strip()