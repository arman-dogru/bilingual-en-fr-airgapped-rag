"""Ask one question from the command line:  python -m src.ask "How long do EI sickness benefits last?" [--lang en|fr]

Works from any directory. Validates input (empty, too long, no letters), detects the language when --lang is
omitted (falls back to English with a warning), always applies the retrieval abstention gate (tau from
results/abstain.json, tuned on dev; exits with an error if it is missing), and runs the NLI citation checker on the
answer: every sentence is reported as supported / unsupported / uncited against its cited chunk. By default
(strict) sentences that are not supported are dropped; --no-strict keeps them and prints a warning to stderr."""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.*")
from .config import path
GATE_KEY = {"mixed_rr": "F_0.5b_mixrr_fewshot", "same": "E_0.5b_same_fewshot"}
def detect_lang(q, warn=lambda m: print(m, file=sys.stderr)):
    from .audit import lang_of
    lang = lang_of(q)
    if lang == "und": warn("warning: could not detect the language; assuming English (pass --lang fr for French)"); lang = "en"
    return lang
def load_tau(retr, abstain_path=None):
    p = abstain_path or path("results", "abstain.json")
    if not os.path.exists(p): raise SystemExit(f"error: {p} not found; run python -m src.abstain first (the abstention gate is mandatory)")
    tau = json.load(open(p)).get(GATE_KEY[retr], {}).get("tau")
    if tau is None: raise SystemExit(f"error: no gate threshold for {GATE_KEY[retr]} in {p}")
    return tau
def check_citations(answer, ctx, entail, th=0.5):
    """Per-sentence verdicts against the cited chunk(s), using the same judge and parser as the audit."""
    from .audit import split_claims, normalize
    out = []
    for t, cites, scored in split_claims(answer):
        r = dict(sentence=t, cites=cites)
        if not scored: r["verdict"] = "not_scored"
        elif not cites: r["verdict"] = "uncited"
        elif any(c < 1 or c > len(ctx) for c in cites): r["verdict"] = "bad_ref"
        else:
            s = max(entail(ctx[c - 1]["prompt_text"], normalize(t)) for c in cites); r["entail"] = round(s, 3)
            r["verdict"] = "supported" if s >= th else "unsupported"
        out.append(r)
    return out
def rebuild(checks):
    """Keep supported / unscored sentences, re-attach their citations, preserve the original end punctuation."""
    parts = []
    for c in checks:
        if c["verdict"] not in ("supported", "not_scored"): continue
        t = c["sentence"]; end = t[-1] if t[-1] in ".!?" else "."
        parts.append(t.rstrip(".!?") + "".join(f" [{x}]" for x in c["cites"]) + end)
    return " ".join(parts)
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question"); ap.add_argument("--lang", choices=("en", "fr"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--retr", choices=("same", "mixed_rr"), default="mixed_rr")
    ap.add_argument("--prompt", choices=("basic", "fewshot"), default="fewshot")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                    help="drop answer sentences the citation checker does not support (default); --no-strict keeps them with a warning")
    a = ap.parse_args(argv)
    from .rag import validate_query, REFUSAL, setup_determinism, prompt_sources, messages, load_llm, generate, K
    try: q = validate_query(a.question)
    except ValueError as e: ap.error(str(e))
    lang = a.lang or detect_lang(q); tau = load_tau(a.retr)
    setup_determinism()
    from .retrieval import build, search, retrieve, free_reranker
    m, idx = build(); ctx = retrieve(m, idx, q, "mixed", K, rr=True) if a.retr == "mixed_rr" else search(m, idx, q, lang, K); free_reranker()
    top = ctx[0]["rr"] if a.retr == "mixed_rr" else ctx[0]["score"]
    if top < tau:
        print(json.dumps(dict(lang=lang, answer=REFUSAL[lang], abstained="retrieval_gate", top_score=round(top, 3), tau=tau), ensure_ascii=False)); return 0
    tok, lm = load_llm(a.model); ctx = prompt_sources(ctx)
    ans = generate(tok, lm, messages(q, lang, ctx, a.prompt))
    from .audit import load_judge, is_refusal
    checks = [] if is_refusal(ans) else check_citations(ans, ctx, load_judge())
    final = ans; ok = all(c["verdict"] in ("supported", "not_scored") for c in checks)
    if a.strict and checks: final = rebuild(checks) or REFUSAL[lang]
    elif not ok: print("WARNING: some answer sentences are not supported by their cited source (see citation_check); "
                       "rerun without --no-strict to drop them.", file=sys.stderr)
    print(json.dumps(dict(lang=lang, answer=final, raw_answer=ans, top_score=round(top, 3), tau=tau,
                          all_supported=ok, strict=a.strict, citation_check=checks,
                          sources=[dict(n=i + 1, url=c["url"], cid=c["cid"]) for i, c in enumerate(ctx)]), ensure_ascii=False, indent=1)); return 0
if __name__ == "__main__": sys.exit(main())
