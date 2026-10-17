"""Ask one question from the command line:  python src/ask.py "How long do EI sickness benefits last?" [--lang en|fr]

Validates input (empty, too long, no letters), detects the language when --lang is omitted, and refuses
without calling the LLM when retrieval finds nothing relevant (threshold tau from results/abstain.json, tuned on dev)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question"); ap.add_argument("--lang", choices=("en", "fr"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--retr", choices=("same", "mixed_rr"), default="mixed_rr")
    ap.add_argument("--prompt", choices=("basic", "fewshot"), default="fewshot")
    a = ap.parse_args(argv)
    from rag import validate_query, REFUSAL, setup_determinism, prompt_sources, messages, load_llm, generate, K
    from audit import lang_of
    try: q = validate_query(a.question)
    except ValueError as e: ap.error(str(e))
    lang = a.lang or lang_of(q)
    if lang == "und": ap.error("could not detect the language; pass --lang en or --lang fr")
    setup_determinism()
    from retrieval import build, search, retrieve, free_reranker
    m, idx = build(); ctx = retrieve(m, idx, q, "mixed", K, rr=True) if a.retr == "mixed_rr" else search(m, idx, q, lang, K); free_reranker()
    tau, key = None, "F_0.5b_mixrr_fewshot" if a.retr == "mixed_rr" else "D_0.5b_same_basic"
    if os.path.exists("results/abstain.json"): tau = json.load(open("results/abstain.json")).get(key, {}).get("tau")
    top = ctx[0].get("rr", ctx[0].get("score"))
    if tau is not None and top < tau:
        print(json.dumps(dict(lang=lang, answer=REFUSAL[lang], abstained="retrieval_gate", top_score=round(top, 3), tau=tau), ensure_ascii=False)); return 0
    tok, lm = load_llm(a.model); ctx = prompt_sources(ctx)
    ans = generate(tok, lm, messages(q, lang, ctx, a.prompt))
    print(json.dumps(dict(lang=lang, answer=ans, sources=[dict(n=i + 1, url=c["url"], cid=c["cid"]) for i, c in enumerate(ctx)]), ensure_ascii=False, indent=1)); return 0
if __name__ == "__main__": sys.exit(main())
