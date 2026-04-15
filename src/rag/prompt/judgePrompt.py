def build_judge_prompt(question, path):
    return f"""
You are evaluating the quality of a question generated from a knowledge graph path.

PATH:
{path}

QUESTION:
{question}

Evaluate the question on the following:

1. Grounding (0–2):
- 2 = fully uses Event, Middle, Related correctly
- 1 = partially grounded
- 0 = not grounded

2. Reasoning Type (0–2):
- 2 = matches path type (causal vs structural)
- 1 = slightly mismatched
- 0 = incorrect reasoning (e.g. fake causality)

3. Clarity (0–1):
- 1 = clear and well-formed
- 0 = unclear or awkward

4. Usefulness (0–2):
- 2 = analytical, meaningful
- 1 = somewhat useful
- 0 = trivial or generic

Return ONLY:
Score: X/7
""" 

'''import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "mistral:7b-instruct"  # change if needed


# -------------------------------
# 1. Build Judge Prompt
# -------------------------------
def build_judge_prompt(questions):
    return f"""
You are an expert evaluator of analytical questions.

Evaluate the following questions and return ONLY the best ones.

Criteria:
- Specific (not vague like "what event")
- Uses entities from the context
- Shows multi-hop reasoning (A → B → C)
- Demonstrates causality (not just description)
- Not repetitive

Questions:
{questions}

Return ONLY a JSON list of the best questions.

Example output:
["question1", "question2"]
"""



# -------------------------------
# 2. Call LLM (Ollama)
# -------------------------------
def call_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    try:
        return response.json()["response"]
    except Exception:
        return ""


# -------------------------------
# 3. Extract Good Questions
# -------------------------------
def extract_good_questions(llm_output):
    try:
        return json.loads(llm_output)
    except Exception:
        return []


# -------------------------------
# 4. Main Judge Function
# -------------------------------
def judge_questions(questions):
    prompt = build_judge_prompt(questions)
    output = call_llm(prompt)
    good_questions = extract_good_questions(output)
    return good_questions'''