import re


class PromptChainRunner:
    def __init__(self, llm_generate_fn):
        self.llm_generate_fn = llm_generate_fn

    # ---------------- PROMPTS ---------------- #

    def build_prompts(self, context, topic, n=5):

        step1 = f"""
You are given structured knowledge graph context about the topic: {topic}.

Context:
{context}

Task:
Extract ONLY the strongest reasoning chains.

Strict Requirements:
- Use ONLY provided entities and relations.
- Extract ONLY meaningful multi-hop chains (2-hop minimum).
- Prefer causal, influence, or indirect-effect relationships.
- REMOVE:
  - trivial links (e.g. metadata, naming, listing)
  - repetitive patterns
  - weak or unclear relations
- Do NOT invent anything.

Output:
Numbered list ONLY.

Format:
1. A -> relation -> B -> relation -> C
"""

        step2_template = f"""
You are given structured knowledge graph context about the topic: {topic}.

Context:
{context}

Task:
Convert each reasoning chain into a reasoning goal.

Strict Requirements:
- Each goal must describe WHAT reasoning is tested
- Focus on:
  - causality
  - indirect effects
  - propagation through intermediate nodes
- Do NOT invent entities

Input Chains:
{{step1_output}}

Output:
Numbered list ONLY.

Format:
1. Ask how X influences Y through Z.
"""

        step3_template = f"""
You are given structured knowledge graph context about the topic: {topic}.

Context:
{context}

Task:
Generate analytical, reasoning-based questions.

Each question must:
- explain or analyse a relationship
- include "how", "why", or "through what mechanism"
- avoid yes/no questions

Strict Requirements:

- Use only entities, relations, and events present in the context.
- Generate a question only if the underlying reasoning chain is strong, clear, and meaningful.
- Each question must correspond to one distinct reasoning goal.
- Each question must reflect multi-hop reasoning (not simple fact lookup).
- Each question must be answerable using the provided context.

Quality and Validity:
- Each question must be complete, grammatically correct, and natural-sounding.
- Each question must be logically valid and factually plausible.
- Do not reverse causal direction.
- Skip any chain that is weak, unclear, or loosely connected.
- Do not combine unrelated entities or domains.

Content Constraints:
- Do not invent entities, events, relations, sectors, or effects.
- Do not generate vague, generic, or factoid-style questions.
- Ensure diversity: avoid repeating the same reasoning pattern across questions.

Formatting Constraints:
- Do not include path notation, arrows, relation labels, or debug-style text.
- Do not mention timestamps, time indices, or uncertainty phrases (e.g., "if any").
- Do not include brackets or explanatory notes.

Output:
- Output only the final numbered questions.


Input Goals:
{{step2_output}}

Output:
Numbered questions ONLY.
"""

        return step1, step2_template, step3_template

    # ---------------- CLEANING ---------------- #

    def _clean_lines(self, output):
        """Convert raw LLM output into clean unique list"""
        if isinstance(output, list):
            lines = output
        else:
            lines = str(output).split("\n")

        cleaned = []
        seen = set()

        for line in lines:
            line = line.strip()

            # remove numbering like "1." or "1)"
            line = re.sub(r"^\d+[\).\-\s]*", "", line)

            if not line:
                continue

            key = line.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(line)

        return cleaned

    def _to_numbered_text(self, lines):
        return "\n".join(f"{i+1}. {x}" for i, x in enumerate(lines))

    # ---------------- LLM CALL ---------------- #

    def llm_call(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be string, got {type(prompt)}")

        output = self.llm_generate_fn(prompt)
        return self._clean_lines(output)

    # ---------------- MAIN PIPELINE ---------------- #

    def run(self, context, topic, n=5):

        p1, p2_template, p3_template = self.build_prompts(context, topic, n)

        # -------- STEP 1: CHAINS -------- #
        step1 = self.llm_call(p1)

        # ⚠️ guard: if model fails
        if len(step1) == 0:
            return {
                "step1_chains": [],
                "step2_goals": [],
                "final_questions": [],
            }

        step1_text = self._to_numbered_text(step1)

        # -------- STEP 2: GOALS -------- #
        p2 = p2_template.format(step1_output=step1_text)
        step2 = self.llm_call(p2)

        if len(step2) == 0:
            return {
                "step1_chains": step1,
                "step2_goals": [],
                "final_questions": [],
            }

        step2_text = self._to_numbered_text(step2)

        # -------- STEP 3: QUESTIONS -------- #
        p3 = p3_template.format(step2_output=step2_text)
        final_qs = self.llm_call(p3)

        # final safety trim
        final_qs = final_qs[:n]

        return {
            "step1_chains": step1,
            "step2_goals": step2,
            "final_questions": final_qs,
        }