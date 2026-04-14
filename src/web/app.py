import json
import requests
from pathlib import Path
from flask import Flask, render_template, request
from src.kg.path_filter import prepare_paths_for_generation
from src.rag.prompt.zero_shot import build_zero_prompt
from src.rag.prompt.ChainOfThought import build_advanced_cot_prompt 
from src.rag.prompt.graphPrompt import build_graph_zero_prompt
from src.rag.prompt.promptChainRunner import PromptChainRunner
from src.rag.prompt.frewShot import build_few_prompt
from src.rag.prompt.dspPrompt import build_dsp_prompt
from src.rag.prompt.chain_prompt import chain_prompt

from src.rag.bootstrap_pipeline import generate_final_questions_only
from src.rag.generate_questions import (
    generate_questions, 
    generate_one_per_path)  # your existing LLM call

from src.kg.retrieve_context import get_context
from src.kg.neo4j_loader import insert_wiki_rows, query_wiki_paths, clear_wiki_subgraph
from src.rag.prompt_template import build_prompt
from src.rag.generate_questions import (
    generate_questions,
    simplify_context,
    dedupe_paths,
    filter_strong_relations,
    remove_bad_paths,
    dedupe_middle_related,
    limit_middle_reuse,
    dedupe_semantic_paths,
    select_diverse_paths,
    select_cluster_representatives,
    clean_questions,
)

#from src.kg.wikidata_api import get_wikidata_context
from src.kg.wikidata_sparql import (
    get_wikidata_context_sparql,
    format_wikidata_context
)



app = Flask(__name__)

def context_rows_to_text(rows):
    lines = []
    for i, r in enumerate(rows, 1):
        line = r.get("context_line")
        if not line:
            line = (
                f'{r.get("event")} --[{r.get("relation1")}]--> {r.get("middle")} '
                f'--[{r.get("relation2")}]--> {r.get("related")}'
            )
        lines.append(f"{i}. {line}")
    return "\n".join(lines)

def generate_until_enough(prompt, n, clean_fn=None, max_attempts=3):
    collected = []
    seen = set()

    for _ in range(max_attempts):
        new_qs = generate_questions(prompt)

        if clean_fn is not None:
            new_qs = clean_fn(new_qs)

        for q in new_qs:
            q_norm = q.strip().lower()
            if q_norm not in seen:
                seen.add(q_norm)
                collected.append(q.strip())

        if len(collected) >= n:
            break

    return collected[:n]


@app.get("/")
def home():
    return render_template(
        "index.html",
        questions_local=None,
        questions_wiki=None,
        topic="",
        n=5,
    )

''' -= DEBUG WIKI VERSION --
@app.post("/generate")
def generate():
    topic = request.form.get("topic", "").strip()
    n = int(request.form.get("n", "5"))

    raw_local_rows = get_context(topic, limit=200, split="train")
    local_rows = simplify_context(raw_local_rows)
    local_rows = dedupe_paths(local_rows)
    local_rows = remove_bad_paths(local_rows)
    local_rows = dedupe_middle_related(local_rows)
    local_rows = limit_middle_reuse(local_rows, max_per_middle=1)
    local_rows = select_diverse_paths(local_rows, final_k=max(n + 8, 15))

    print(f"\n[DEBUG] topic = {topic}")
    print(f"[DEBUG] local_rows count = {len(local_rows)}")

    try:
        wiki_rows = get_wikidata_context(topic)
    except Exception as e:
        print(f"[DEBUG] get_wikidata_context failed: {e}")
        wiki_rows = []

    print(f"[DEBUG] wiki_rows count = {len(wiki_rows)}")
    if wiki_rows:
        print(f"[DEBUG] first wiki row = {wiki_rows[0]}")
    else:
        print("[DEBUG] wiki_rows is empty")

    if len(local_rows) < 3 and len(wiki_rows) < 3:
        return render_template(
            "index.html",
            questions_local=["Not enough Local KG data"],
            questions_wiki=["Not enough Wikidata data"],
            topic=topic,
            n=n,
        )

    qs_local = []
    qs_wiki = []

    if len(local_rows) >= 3:
        prompt_local = build_prompt(topic, local_rows, n_questions=n)
        qs_local = generate_until_enough(
            prompt_local,
            n,
            clean_fn=clean_questions,
            max_attempts=3,
        )

        if len(qs_local) == 0:
            qs_local = ["Could not generate enough clean Local KG questions"]
    else:
        qs_local = ["Not enough Local KG context"]

    if len(wiki_rows) >= 3:
        try:
            clear_wiki_subgraph()
            insert_wiki_rows(wiki_rows)

            structured_rows = query_wiki_paths(limit=25)
            print(f"[DEBUG] structured_rows before filter = {len(structured_rows)}")

            if structured_rows:
                print(f"[DEBUG] first raw structured row = {structured_rows[0]}")

            structured_rows = prepare_paths_for_generation(
                structured_rows,
                target_k=max(n + 3, 8),
            )
            print(f"[DEBUG] structured_rows after filter = {len(structured_rows)}")

            if structured_rows:
                print(f"[DEBUG] first filtered structured row = {structured_rows[0]}")

            if len(structured_rows) >= 3:
                prompt_wiki = build_zero_prompt(topic, structured_rows, n=n)
                qs_wiki = generate_until_enough(
                    prompt_wiki,
                    n,
                    clean_fn=clean_questions,
                    max_attempts=2,
                )

                if len(qs_wiki) == 0:
                    qs_wiki = ["Could not generate enough Wikidata questions"]
            else:
                qs_wiki = ["Not enough structured Wikidata paths"]

        except Exception as e:
            print(f"[DEBUG] Wikidata structuring failed: {e}")
            qs_wiki = [f"Wikidata pipeline error: {e}"]
    else:
        qs_wiki = ["Not enough Wikidata context"]

    return render_template(
        "index.html",
        questions_local=qs_local,
        questions_wiki=qs_wiki,
        topic=topic,
        n=n,
    )
'''







@app.post("/generate")
def generate():
    topic = request.form.get("topic", "").strip()
    n = int(request.form.get("n", "5"))

    raw_local_rows = get_context(topic, limit=200, split="train")
    local_rows = simplify_context(raw_local_rows)
    
    local_rows = dedupe_paths(local_rows)
    #local_rows = filter_strong_relations(local_rows) # ADDED
    #local_rows = remove_bad_paths(local_rows)
    #local_rows = dedupe_middle_related(local_rows)
    local_rows = limit_middle_reuse(local_rows, max_per_middle=2) # ADDED
    #local_rows = dedupe_semantic_paths(local_rows, threshold=0.90)
    #broken local_rows = select_diverse_paths(local_rows, final_k=max(n + 8, 15))


    '''local_rows = select_cluster_representatives(
        local_rows,
        max_per_cluster=2,
        distance_threshold=0.30,
    )'''

    #wiki_rows = get_wikidata_context(topic)

    wiki_rows = get_wikidata_context_sparql(topic, max_paths=15, sparql_limit=800)
    #wiki_context = format_wikidata_context(wiki_rows)
    
    print("\n--- TOPIC ---", topic)

    print("\n--- LOCAL CONTEXT ---")
    for row in local_rows[:5]:
        print(row)

    print("\n--- WIKIDATA CONTEXT ---")
    if wiki_rows:
        for row in wiki_rows[:5]:
            print(row)
    else:
        print("No wiki rows found")

    qs_local = []
    qs_wiki = []

    local_status = None
    wiki_status = None
    has_local = len(local_rows) >= 3
    has_wiki = len(wiki_rows) > 0   # relaxed for Wikidata


    if not has_local and not has_wiki:
        return render_template(
            "index.html",
            questions_local=[],
            questions_wiki=[],
            local_status="Not enough Local KG data",
            wiki_status="Not enough Wikidata data",
            topic=topic,
            n=n,
        )


    '''
    if len(local_rows) >= 3:
        runner = PromptChainRunner(generate_questions)
        context_text = context_rows_to_text(local_rows)
        
        chain_result = runner.run(
          context=context_text,
          topic=topic,
          n=n
        )

        qs_local = clean_questions(chain_result["final_questions"])[:n]

        if len(qs_local) == 0:
          qs_local = ["Could not generate enough clean Local KG questions"]
    else:
       qs_local = ["Not enough Local KG context"]



    '''
    if len(local_rows) >= 3:

        
        #prompt_local = build_zero_prompt(topic, local_rows, n=n)
        cand = max(n*2,10)
        prompt_local = build_prompt(topic, local_rows, n=cand)
        #prompt_local = chain_prompt(topic, local_rows, n=cand)
        #prompt_local = build_dsp_prompt(topic, local_rows, n=n)
        #prompt_local = build_advanced_cot_prompt(topic, local_rows, n=n)
        #prompt_local = build_few_prompt(topic, local_rows, n=n)
        #prompt_local = build_graph_zero_prompt(topic, local_rows, n=n)

        qs_local = generate_until_enough(
            prompt_local,
            n,
            clean_fn=clean_questions,
            max_attempts=3,
        )

        '''
        context_text = context_rows_to_text(local_rows)
        qs_local = generate_final_questions_only(
            context=context_text,
            topic=topic,
            llm_generate_fn=generate_questions,  #  existing function
            n=n,
            csv_path="bootstrap_runs.csv",       # saves automatically)'''

        if len(qs_local) == 0:
            qs_local = ["Could not generate enough clean Local KG questions"]
    else:
        qs_local = ["Not enough Local KG context"]

    if has_wiki:
        try:
            clear_wiki_subgraph()
            insert_wiki_rows(wiki_rows)

            structured_rows = query_wiki_paths(limit=600)
            structured_rows = limit_middle_reuse(structured_rows, max_per_middle=1)

            # only prepare if there is something to prepare
            '''if structured_rows:
                structured_rows = prepare_paths_for_generation(
                    structured_rows,
                    target_k=max(n + 3, 8),
                )'''

            print("\n--- STRUCTURED WIKIDATA ROWS ---")
            if structured_rows:
                for row in structured_rows[:5]:
                    print(row)
            else:
                print("No structured wiki rows found.")

            
            if len(structured_rows) > 0:
                qs_wiki = generate_one_per_path(topic, structured_rows, n)

                if len(qs_wiki) == 0:
                    wiki_status = "Wikidata generated no usable questions"
            else:
                qs_wiki = generate_one_per_path(topic, wiki_rows, n)
                if len(qs_wiki) == 0:
                    wiki_status = "Wikidata generated no usable questions from raw paths"


            '''
            if len(structured_rows) > 0:   # relaxed from >= 3
                cand = max(n*2,10)
                prompt_wiki = build_prompt(topic, structured_rows, n=cand)
                #prompt_wiki = build_zero_prompt(topic, structured_rows, n=cand)
                qs_wiki = generate_until_enough(
                    prompt_wiki,
                    n,
                    clean_fn=clean_questions,
                    max_attempts=2,
                )

                if len(qs_wiki) == 0:
                    wiki_status = "Wikidata generated no usable questions"
            else:
                # fallback: use raw wikidata rows directly
                #prompt_wiki = build_prompt(topic, wiki_rows, n=n)
                prompt_wiki = build_zero_prompt(topic, wiki_rows, n=n)
                qs_wiki = generate_until_enough(
                    prompt_wiki,
                    n,
                    clean_fn=clean_questions,
                    max_attempts=2,
                )
                
                if len(qs_wiki) == 0:
                    wiki_status = "Wikidata generated no usable questions from raw paths"'''

        except Exception as e:
            print(f"[WIKI ERROR] {e}")
            wiki_status = f"Wikidata pipeline error: {e}"
    else:
        wiki_status = "Not enough Wikidata context"

    return render_template(
        "index.html",
        questions_local=qs_local,
        questions_wiki=qs_wiki,
        local_status=local_status,
        wiki_status=wiki_status,
        topic=topic,
        n=n,
    )

    '''if len(wiki_rows) >= 3:
        clear_wiki_subgraph()
        insert_wiki_rows(wiki_rows)
        structured_rows = query_wiki_paths(limit=25)
        structured_rows = prepare_paths_for_generation(
            structured_rows,
            target_k=max(n + 3, 8),
        )

        if len(structured_rows) >= 3:
            prompt_wiki = build_prompt(topic, structured_rows, n_questions=n)
            #prompt_wiki = build_graph_zero_prompt(topic, structured_rows, n_questions=n)
            qs_wiki = generate_until_enough(
                prompt_wiki,
                n,
                clean_fn=clean_questions,
                max_attempts=2,
            )

            if len(qs_wiki) == 0:
                qs_wiki = ["Could not generate enough Wikidata questions"]
        else:
            qs_wiki = ["Not enough structured Wikidata paths"]
    else:
        qs_wiki = ["Not enough Wikidata context"]

    return render_template(
        "index.html",
        questions_local=qs_local,
        questions_wiki=qs_wiki,
        topic=topic,
        n=n,
    )'''


'''
@app.post("/generate")
def generate():
    topic = request.form.get("topic", "").strip()
    n = int(request.form.get("n", "5"))

    //raw_local_rows = get_context(topic, limit=100, split="train")
    local_rows = simplify_context(raw_local_rows)
    local_rows = dedupe_paths(local_rows)
    local_rows = remove_bad_paths(local_rows)
    local_rows = dedupe_middle_related(local_rows)
    local_rows = limit_middle_reuse(local_rows, max_per_middle=1)
    local_rows = select_diverse_paths(local_rows, final_k=max(n + 3, 8))//

    structured_rows = query_wiki_paths(limit=50)
    structured_rows = prepare_paths_for_generation(structured_rows, target_k=max(n + 3, 8))

    wiki_rows = get_wikidata_context(topic)

    if len(local_rows) < 3 and len(wiki_rows) < 3:
        return render_template(
            "index.html",
            questions_local=["Not enough Local KG data"],
            questions_wiki=["Not enough Wikidata data"],
            topic=topic,
            n=n,
        )

    print("\n--- TOPIC ---", topic)

    print("\n--- LOCAL CONTEXT ---")
    for row in local_rows[:5]:
        print(row)

    print("\n--- WIKIDATA CONTEXT ---")
    for row in wiki_rows[:5]:
        print(row)

    qs_local = []
    qs_wiki = []

    if len(local_rows) >= 3:
        prompt_local = build_prompt(topic, local_rows, n_questions=n)
        qs_local = generate_until_enough(
            prompt_local,
            n,
            clean_fn=clean_questions,
            max_attempts=3,
        )

        if len(qs_local) == 0:
            qs_local = ["Could not generate enough clean Local KG questions"]
    else:
        qs_local = ["Not enough Local KG context"]

    if len(wiki_rows) >= 3:
        clear_wiki_subgraph()
        insert_wiki_rows(wiki_rows)
        structured_rows = query_wiki_paths(limit=25)

        if len(structured_rows) >= 3:
            prompt_wiki = build_prompt(topic, structured_rows, n_questions=n)
            qs_wiki = generate_until_enough(
                prompt_wiki,
                n,
                clean_fn=clean_questions,
                max_attempts=2,
            )

            if len(qs_wiki) == 0:
                qs_wiki = ["Could not generate enough Wikidata questions"]
        else:
            qs_wiki = ["Not enough structured Wikidata paths"]
    else:
        qs_wiki = ["Not enough Wikidata context"]

    return render_template(
        "index.html",
        questions_local=qs_local,
        questions_wiki=qs_wiki,
        topic=topic,
        n=n,
    )

'''
if __name__ == "__main__":
    app.run(debug=True, port=5001)