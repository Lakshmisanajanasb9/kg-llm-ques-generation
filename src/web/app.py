from flask import Flask, render_template, request

from src.kg.retrieve_context import get_context
from src.rag.generate_questions import generate_questions, simplify_context
from src.rag.prompt_template import build_prompt

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html", questions=None, topic="", n=5)


@app.post("/generate")
def generate():
    topic = request.form.get("topic", "").strip()
    n = int(request.form.get("n", "5"))

    if not topic:
        return render_template("index.html", questions=["Please enter a topic."], topic="", n=n)

    # 1) Retrieve live context from Neo4j
    raw_rows = get_context(topic, limit=25, split="train")

    # 2) Handle "no context" case
    if not raw_rows or len(raw_rows) < 3:
        return render_template(
            "index.html",
            questions=[
                "Not enough knowledge graph context for this topic yet. "
                "Try: interest rates, inflation, oil, gold, OPEC."
            ],
            topic=topic,
            n=n,
        )

    # 3) Simplify to the exact schema your prompt_template expects
    context_rows = simplify_context(raw_rows)

    # (Optional) debug print in terminal logs
    print("\n--- TOPIC ---", topic)
    print("--- CONTEXT (first 5) ---")
    for row in context_rows[:5]:
        print(row)

    # 4) Build prompt and generate
    prompt = build_prompt(topic, context_rows, n_questions=n)
    qs = generate_questions(prompt)[:n]

    return render_template("index.html", questions=qs, topic=topic, n=n)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
