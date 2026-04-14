def build_few_prompt(event_keyword: str, context_rows: list[dict], n: int = 5) -> str:
    facts = "\n".join(
        [
            f"- {row['event']} --{row['relation1']}--> {row['middle']} --{row['relation2']}--> {row['related']}"
            for row in context_rows
        ]
    )

    return f"""
You are a financial and geopolitical analyst.

You are given 2-hop knowledge-graph paths. Your task is to convert the paths into natural, clear, analytical questions.

Task:
Generate up to {n} questions grounded only in the facts below.

Rules:
- Use only entities and relations that appear in the facts.
- Each question must be based on exactly one complete path.
- Each question should reflect the full 2-hop chain: Event → Middle → Related.
- Do not combine multiple paths into one question.
- Do not invent new entities, dates, mechanisms, sectors, or examples.
- Do not copy graph relations literally if they sound unnatural; rewrite them into natural English.
- Keep the questions concise, natural, and specific.
- Avoid vague phrases such as "what is the relationship", "in what way", "what event", or "how is X related to Y".
- Avoid repeating the same question pattern across outputs.
- Use varied analytical styles where possible, such as impact, mechanism, consequence, or connection.
- If a path is too weak, generic, or unclear to make a good question, skip it.
- Output at most {n} questions.
- Each line must start with "Q:"
- Output only the questions.

Good examples:

Path: JPMorgan Chase --Decrease--> Shares --Impact--> Consumer sentiment
Q: How do declines in JPMorgan Chase shares affect consumer sentiment?

Path: President Barack Obama --Impact--> GDP Growth --Negative_Impact_On--> Unemployment rate
Q: How does Barack Obama's impact on GDP growth influence unemployment rates?

Path: Treasury Inflation-Protected Securities --Introduce--> Inflation expectations --Decrease--> Inflation pressures
Q: How do Treasury Inflation-Protected Securities reduce inflation pressures through their effect on inflation expectations?

Path: Germany --Impact--> Inflation --Impact--> Consumer spending
Q: How does inflation in Germany shape consumer spending?

Bad examples:
Q: How does this affect the wider economy and investor confidence?
Q: What broader mechanism explains this relationship?
Q: How does JPMorgan Chase, Warren Buffett, and Alibaba interact?
Q: What is GDP growth?
Q: In what way is this connected?
Q: What is the role of this factor?

Topic: {event_keyword}

Facts:
{facts}

Output:
"""