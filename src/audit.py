"""Citation audit: for each answer sentence, does its cited chunk entail it? Plus answer-language match and abstention.

Per-answer rows are kept for every answer (including zero-claim answers and refusals) so that
compare.py can run paired, question-clustered bootstraps. Judged text == the text shown to the generator.
"""
import json, re, os, sys
import numpy as np
NLI = os.environ.get("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
FR_W = set("le la les des du de et est un une pour que qui vous dans pas sur au aux sont avec ou ne oui non semaines jours "
           "comment quand où combien quel quelle quels quelles qu est-ce je j' mon ma mes puis-je dois-je faut-il il elle on ce cette ces leur par".split())
EN_W = set("the and is of to for you in are a an that with your be can not or on as if yes no weeks days "
           "how what when where who which why do does did i my me much many long get should will it this there from by".split())
def lang_of(t):
    """'en' / 'fr' by stopword vote; 'und' when there is no signal or a tie (no default to EN)."""
    w = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", t.lower()); f = sum(x in FR_W for x in w); e = sum(x in EN_W for x in w)
    if re.search(r"[àâçèêëîïôûùüÿœ]|é", t.lower()): f += 1
    return "fr" if f > e else "en" if e > f else "und"
ATTR = re.compile(r"^(according to|based on|as (stated|mentioned) in|per) (the )?(source|sources|document|information|first link)[^,]*,\s*|^(selon|d'après) (la |les |le )?(source|sources|document|informations?)[^,]*,\s*", re.I)
def normalize(t):
    t = ATTR.sub("", t).strip(); return t[:1].upper() + t[1:] if t else t
CITE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
ABBR = re.compile(r"\b(e\.g|i\.e|etc|p\. ex|c\.-à-d|M|Mme|Dr|no|art)\.\s*$", re.I)
def _nums(s):
    out = []
    for part in re.split(r"\s*,\s*", s):
        if re.search(r"[\-–]", part):
            a, b = [int(x) for x in re.split(r"\s*[\-–]\s*", part)]; out += list(range(a, b + 1)) if 0 < b - a < 10 else [a, b]
        else: out.append(int(part))
    return out
def sentences(ans):
    """Split into sentences; a citation right after the period ("... weeks.[1]") belongs to that sentence."""
    ans = re.sub(r"([.!?])\s*((?:\[\d+(?:\s*[,\-–]\s*\d+)*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", ans)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", ans) if p.strip()]
    merged = []
    for p in parts:  # re-join false splits after abbreviations ("E.g. you ...")
        if merged and ABBR.search(merged[-1]): merged[-1] += " " + p
        else: merged.append(p)
    return merged
REFUSAL_SENTENCES = ("The sources do not contain this information.", "Les sources ne contiennent pas cette information.")
def _norm_s(t): return re.sub(r"[\s«»\"'.]+", " ", CITE.sub("", t)).strip().lower()
_REF_N = {_norm_s(x) for x in REFUSAL_SENTENCES}
_LEAD = re.compile(r"^(no|non|oui|yes|unfortunately|malheureusement|(je suis )?désolée?|(i'?m |i am )?sorry)\s*,?\s*(mais|but)?\s*", re.I)
_REF_PATTERNS = re.compile(
    r"^(les |the |these |ces )?(sources?|documents?|informations?|passages?)( [\w'-]+){0,5} (ne (contiennent|contient|mentionnent|mentionne|fournissent|fournit) pas|(do|does) not (contain|mention|provide|include)|don't (contain|mention|provide))"
    r"|^(je ne (trouve|vois)|je n'ai (pas )?trouvé|i (could not|couldn't|cannot|can't|did not|didn't) find)[^.]{0,40}\b(aucune?|pas d[e']|any|no|specific)\b[^.]{0,30}(information|donnée|data|détail|details?)", re.I)
_CLAUSE = re.compile(r"[,;:()]|\b(but|mais|however|cependant|toutefois|though|although|bien que|whereas|tandis que|yet|except|sauf)\b", re.I)
def is_refusal_sentence(t):
    """True only when the WHOLE sentence is a refusal (M1). Either:
    - the fixed refusal sentence (EN/FR), compared after normalisation; citation markers are ignored only in this exact case
      (e.g. "Les sources ne contiennent pas cette information [1]."), or
    - a paraphrased refusal ("Les sources mentionnées ne contiennent pas…", "Je suis désolé, mais je ne trouve aucune
      information…", "No, the sources do not contain…") that carries no citation, no digit and no trailing clause
      (", but …", "; …", "mais …", "however …"). A hedge followed by a claim is a partial answer, not a refusal.
    There is no fuzzy ratio: "The sources do contain this information" must not match."""
    n = _norm_s(t)
    if n in _REF_N: return True
    if CITE.search(t) or re.search(r"\d", t): return False
    body = _LEAD.sub("", t.strip().rstrip(".!?").strip())
    m = _REF_PATTERNS.search(body)
    if not m: return False
    rest = body[m.end():]
    return not _CLAUSE.search(rest) and len(re.findall(r"\w+", rest)) <= 25
def split_claims(ans):
    """All sentences with their citations: [(text, cites, scored)]. Short / fragment sentences are kept
    (scored=False) so post-hoc attribution never deletes them."""
    out = []
    for s in sentences(ans):
        cites = [n for g in CITE.findall(s) for n in _nums(g)]
        t = re.sub(r"\s*" + CITE.pattern, "", s).strip()
        if not t and out: out[-1][1].extend(cites); continue   # a bare "[1]" attaches to the previous sentence
        if not t: continue
        words = re.findall(r"\w+", t)
        scored = len(words) >= 4 or (len(words) >= 2 and bool(re.search(r"\d", t)))  # "Until age 85." is a claim
        if is_refusal_sentence(s): scored = False   # the fixed refusal sentence is an abstention, not a claim
        out.append([t, cites, scored])
    return out
def claims(ans):
    """Scored claims only: [(text, cites)]."""
    return [(t, c) for t, c, s in split_claims(ans) if s]
def is_refusal(ans):
    """Abstention = empty answer, or an answer whose first sentence is entirely a refusal (is_refusal_sentence).
    Hedges inside partial answers ("...does not provide the date, but ... [1]") are NOT abstentions."""
    if not ans.strip(): return True
    first = sentences(ans)[0] if sentences(ans) else ""
    return is_refusal_sentence(first)
def has_refusal_sentence(ans): return any(is_refusal_sentence(s) for s in sentences(ans))
def split_of(qid): return "unanswerable" if qid.startswith("u") else ("dev" if int(qid[1:]) <= 26 else "test")
def boot_ci(per_q, f, B=2000, seed=0):
    """Cluster bootstrap over questions (EN+FR of a question resampled together). per_q: list of per-question payloads."""
    rng = np.random.default_rng(seed); n = len(per_q); vals = [f([per_q[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])]
def faith(rows):  # pooled over claims: supported / claims; zero-claim answers contribute 0/0
    c = sum(r["n_claims"] for r in rows); return sum(r["n_supported"] for r in rows) / c if c else 0.0
def grounded(rows):
    c = sum(r["n_claims"] for r in rows); return sum(r["n_grounded"] for r in rows) / c if c else 0.0
def load_judge():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from .config import rev, THREADS
    torch.set_num_threads(THREADS)
    try:
        tok = AutoTokenizer.from_pretrained(NLI, revision=rev(NLI)); m = AutoModelForSequenceClassification.from_pretrained(NLI, revision=rev(NLI)).eval()
    except OSError as e: raise SystemExit(f"error: cannot load judge {NLI}: {e}")
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    def entail(p, h):
        x = tok(p, h, truncation="only_first", max_length=512, return_tensors="pt")
        with torch.no_grad(): pr = m(**x).logits.softmax(-1)[0]
        return float(pr[lab["entailment"]])
    return entail
def main():
    inp, out = os.environ.get("IN"), os.environ.get("OUT")
    if not inp or not out: raise SystemExit("error: set IN=<answers json> and OUT=<audit json>")
    if not os.path.exists(inp): raise SystemExit(f"error: {inp} not found")
    D = json.load(open(inp)); A = D["answers"]; entail = load_judge(); norm = os.environ.get("NORMALIZE", "1") == "1"
    rows, answers = [], []
    for a in A:
        ar = dict(id=a["id"], lang=a["lang"], split=split_of(a["id"]), answerable=a.get("answerable", True), refusal=is_refusal(a["answer"]), refusal_sentence=has_refusal_sentence(a["answer"]),
                  lang_detected=lang_of(a["answer"]), n_claims=0, n_supported=0, n_grounded=0, n_uncited=0, n_badref=0)
        for t, cites in claims(a["answer"]):
            r = dict(id=a["id"], lang=a["lang"], split=ar["split"], claim_raw=t, cites=cites); t = normalize(t) if norm else t; r["claim"] = t
            anys = max(entail(src["text"], t) for src in a["sources"]) if a["sources"] else 0.0; r["any_source_entail"] = round(anys, 3)
            ar["n_claims"] += 1; ar["n_grounded"] += anys >= 0.5
            if not cites: r["verdict"] = "uncited"; ar["n_uncited"] += 1
            elif any(c < 1 or c > len(a["sources"]) for c in cites): r["verdict"] = "bad_ref"; ar["n_badref"] += 1
            else:
                s = max(entail(a["sources"][c-1]["text"], t) for c in cites); r["entail"] = round(s, 3)
                r["verdict"] = "supported" if s >= 0.5 else "unsupported"; ar["n_supported"] += s >= 0.5
            rows.append(r)
        answers.append(ar)
    def summary(ans):
        tot = sum(r["n_claims"] for r in ans); cited = tot - sum(r["n_uncited"] + r["n_badref"] for r in ans)
        byq = {}; [byq.setdefault(r["id"], []).append(r) for r in ans]; Q = list(byq.values())
        flat = lambda qs: [r for q in qs for r in q]
        s = dict(answers=len(ans), claims=tot, zero_claim_answers=sum(r["n_claims"] == 0 for r in ans),
                 citation_faithfulness=round(faith(ans), 3), citation_faithfulness_ci95=boot_ci(Q, lambda qs: faith(flat(qs))),
                 faithfulness_of_cited=round(sum(r["n_supported"] for r in ans) / max(cited, 1), 3),
                 uncited_rate=round(sum(r["n_uncited"] for r in ans) / max(tot, 1), 3), bad_ref_rate=round(sum(r["n_badref"] for r in ans) / max(tot, 1), 3),
                 grounded_in_any_source=round(grounded(ans), 3), grounded_ci95=boot_ci(Q, lambda qs: grounded(flat(qs))),
                 answer_language_match=round(sum(r["lang_detected"] == r["lang"] for r in ans) / max(len(ans), 1), 3),
                 refusal_rate=round(sum(r["refusal"] for r in ans) / max(len(ans), 1), 3),
                 refusal_sentence_rate=round(sum(r["refusal_sentence"] for r in ans) / max(len(ans), 1), 3))
        for L in ("en", "fr"): s[f"faithfulness_{L}"] = round(faith([r for r in ans if r["lang"] == L]), 3)
        return s
    rep = dict(nli=NLI, nli_revision=__import__('src.config', fromlist=['rev']).rev(NLI), source_file=inp, normalize=norm, llm=D.get("llm"), env=D.get("env"))
    for sp in ("all_answerable", "dev", "test", "unanswerable"):
        sub = [r for r in answers if (r["answerable"] if sp == "all_answerable" else r["split"] == sp)]
        if sub: rep[sp] = summary(sub)
    print(json.dumps({k: v for k, v in rep.items() if k != "env"}, indent=1))
    json.dump(dict(summary=rep, answers=answers, claims=rows), open(out, "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__": main()
