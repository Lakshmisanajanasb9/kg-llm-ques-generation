def build_few_prompt(
    event_keyword: str,
    context_rows: list[dict],
    examples: list[dict] | None = None,
    n: int = 5
) -> str:
    facts = "\n".join(
        f"- {row['event']} --{row['relation1']}--> {row['middle']} --{row['relation2']}--> {row['related']}"
        for row in context_rows
    )

    example_block = ""
    if examples:
        formatted_examples = []
        for ex in examples:
            path_text = ex.get("path_text", "").strip()
            question = ex.get("question", "").strip()
            if path_text and question:
                formatted_examples.append(
                    f"Example Path:\n{path_text}\nExample Question:\nQ: {question}"
                )
        if formatted_examples:
            example_block = "\n\nBootstrapped Examples:\n" + "\n\n".join(formatted_examples)

    return f"""
You are a financial and geopolitical analyst.

You are given 2-hop knowledge graph paths about the topic: {event_keyword}.

Your task is to convert the paths into clear, natural, analytical questions.

{example_block}

Instructions:
- Generate up to {n} questions.
- Use only the facts provided below.
- Each question must be based on exactly one complete 2-hop path.
- Each question must reflect the full chain: Event -> Middle -> Related.
- Do not combine multiple paths into one question.
- Do not invent entities, dates, sectors, mechanisms, or outcomes.
- Rewrite graph-style relations into natural English.
- Keep questions clear, specific, and grammatically correct.
- If a path is weak, vague, or unclear, skip it.
- Avoid duplicates and near-duplicates.
- Use varied but natural analytical styles.

Do:
- ask about impact, mechanism, consequence, or connection when supported by the path
- preserve the original direction of the path
- keep the wording concise

Do not:
- ask definition questions
- use vague phrases like:
  - "what is the relationship"
  - "in what way"
  - "what event"
  - "how is X related to Y"
- combine entities from different paths
- output fragments, notes, or explanations

Good examples:

Path:
JPMorgan Chase --Decrease--> Shares --Impact--> Consumer sentiment
Q: How do declines in JPMorgan Chase shares affect consumer sentiment?

Path:
President Barack Obama --Impact--> GDP Growth --Negative_Impact_On--> Unemployment rate
Q: How does Barack Obama's impact on GDP growth influence unemployment rates?

Path:
Treasury Inflation-Protected Securities --Introduce--> Inflation expectations --Decrease--> Inflation pressures
Q: How do Treasury Inflation-Protected Securities reduce inflation pressures through their effect on inflation expectations?

Path:
Germany --Impact--> Inflation --Positive_Impact_On--> Consumer spending
Q: How does Germany influence consumer spending through its impact on inflation?

Bad examples:

Q: What is GDP growth?
Q: What broader mechanism explains this relationship?
Q: How does JPMorgan Chase, Warren Buffett, and Alibaba interact?
Q: In what way is this connected?
Q: What is the role of this factor?

Facts:
{facts}

Output:
Q:
""".strip()