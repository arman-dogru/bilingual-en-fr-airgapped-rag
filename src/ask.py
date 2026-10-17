"""Ask one question from the command line:  python -m src.ask "How long do EI sickness benefits last?" [--lang en|fr]

Run from the repository root (src/ is a package; PYTHONPATH=<repo> works elsewhere). Validates input (empty, too long, no letters), detects the language when --lang is
omitted (falls back to English with a warning), always applies the retrieval abstention gate (tau from
results/abstain.json, tuned on dev; exits with an error if it is missing), and runs the NLI citation checker on the
answer: every non-refusal sentence is reported as supported / unsupported / uncited / bad_ref against its cited chunk.
By default (strict) the answer is only the citation-supported claim sentences, or the fixed refusal sentence if none survive;
no other model text is echoed. --no-strict prints the raw answer and a WARNING listing every unverified sentence."""
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
    from .audit import split_claims, normalize, sentences, is_refusal_sentence
    out = []
    for t, cites, scored in split_claims(answer):
        r = dict(sentence=t, cites=cites)
        if not scored: r["verdict"] = "refusal" if is_refusal_sentence(t) else "not_scored"
        elif not cites: r["verdict"] = "uncited"
        elif any(c < 1 or c > len(ctx) for c in cites): r["verdict"] = "bad_ref"
        else:
            s = max(entail(ctx[c - 1]["prompt_text"], normalize(t)) for c in cites); r["entail"] = round(s, 3)
            r["verdict"] = "supported" if s >= th else "unsupported"
        out.append(r)
    return out
def rebuild(checks):
    """Strict output: ONLY citation-supported claim sentences, with their citations and original end punctuation.
    Nothing else the model wrote is echoed: not refusal-like sentences (the refusal classifier is for metrics only and is
    never trusted here), not unscored fragments, not uncited/unsupported sentences."""
    parts = []
    for c in checks:
        if c["verdict"] != "supported": continue
        t = c["sentence"]; end = t[-1] if t[-1] in ".!?" else "."
        parts.append(t.rstrip(".!?") + "".join(f" [{x}]" for x in c["cites"]) + end)
    return " ".join(parts)
def final_answer(ans, ctx, entail, lang, strict=True):
    """The one place that turns a generated answer into what the user sees (used by main() and robustness.py).
    Every sentence goes through the citation checker (check_citations); the refusal classifier only labels sentences.
      strict:     answer = the supported claim sentences (rebuild); if there are none, our own constant REFUSAL[lang]
                  (never the model's refusal text).
      not strict: answer = the raw model text; `warnings` lists every sentence that is unsupported / uncited / bad_ref,
                  every sentence classed as a refusal that is not byte-exact one of the fixed refusal sentences, and
                  every other unchecked fragment.
    all_supported: None only when the answer is empty or consists solely of the exact fixed refusal sentence(s);
                   otherwise True iff every sentence is a supported claim."""
    from .rag import REFUSAL
    fixed = set(REFUSAL.values())
    checks = check_citations(ans, ctx, entail)
    warn = [c["sentence"] for c in checks if c["verdict"] != "supported" and not (c["verdict"] == "refusal" and c["sentence"] in fixed)]
    claims_ = [c for c in checks if c["verdict"] == "supported"]
    ok = None if not claims_ and not warn else not warn
    final = (rebuild(checks) or REFUSAL[lang]) if strict else (ans if ans.strip() else REFUSAL[lang])
    return dict(answer=final, checks=checks, all_supported=ok, refused=final == REFUSAL[lang], warnings=[] if strict else warn)
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
    from . import audit
    r = final_answer(ans, ctx, audit.load_judge(), lang, strict=a.strict)
    if r["warnings"]:
        print(f"WARNING: {len(r['warnings'])} answer sentence(s) were not verified against a cited source "
              "(unsupported, uncited, or refusal-like text that is not the fixed refusal sentence); see citation_check. "
              "Rerun without --no-strict to drop them:", file=sys.stderr)
        for w in r["warnings"]: print(f"  - {w}", file=sys.stderr)
    print(json.dumps(dict(lang=lang, answer=r["answer"], raw_answer=ans, top_score=round(top, 3), tau=tau,
                          all_supported=r["all_supported"], strict=a.strict, citation_check=r["checks"],
                          sources=[dict(n=i + 1, url=c["url"], cid=c["cid"]) for i, c in enumerate(ctx)]), ensure_ascii=False, indent=1)); return 0
if __name__ == "__main__": sys.exit(main())
