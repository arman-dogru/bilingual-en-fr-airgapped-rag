"""Judge calibration on data/calibration_set.json (keyed by id + lang + claim text, never by position).

Two scoring modes:
  cited - the headline audit metric: entailment by the claim's cited chunk(s); only items that carry a citation.
  any   - entailment by any retrieved chunk of the answer (the 'grounded' metric); all items.
Fails loudly if a 'generated' claim no longer occurs in its source answer. Reports accuracy, precision,
recall and specificity with Wilson 95% CIs, per item type, raw vs prefix-normalized."""
import json, math, os, re, sys
from .audit import load_judge, normalize, CITE
from .config import path
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [round(c - h, 3), round(c + h, 3)]
def strip(t): return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(CITE.sub("", t).split())).rstrip(".!? ")
_cache = {}
def answers(f):
    if f not in _cache: _cache[f] = {(a["id"], a["lang"]): a for a in json.load(open(path(f)))["answers"]}
    return _cache[f]
def src_answer(it):
    A = answers(it.get("source_answers", "data/calibration_answers.json"))
    return A[(it["sources_from"]["id"], it["sources_from"]["lang"])] if "sources_from" in it else A[(it["id"], it["lang"])]
def check_keys(items):
    bad = []
    for it in items:
        a = answers(it.get("source_answers", "data/calibration_answers.json")).get((it["id"], it["lang"]))
        if a is None: bad.append((it["id"], it["lang"], "no such answer")); continue
        ref = it["claim"] if it["kind"] in ("generated", "source_swap") else it.get("derived_from")
        if ref and strip(ref) not in strip(a["answer"]): bad.append((it["id"], it["lang"], ref[:60]))
    if bad: raise SystemExit(f"error: {len(bad)} calibration items do not match their source answers: {bad[:5]}")
def score(it, entail, norm, mode):
    srcs = src_answer(it)["sources"]; h = normalize(it["claim"]) if norm else it["claim"]
    if mode == "cited":
        c = [x for x in it.get("cites", []) if 1 <= x <= len(srcs)]
        if not c: return None
        return max(entail(srcs[x - 1]["text"], h) for x in c)
    return max(entail(s["text"], h) for s in srcs)
def metrics(pairs, th):
    tp = sum(s >= th and l == 1 for l, s in pairs); fp = sum(s >= th and l == 0 for l, s in pairs)
    fn = sum(s < th and l == 1 for l, s in pairs); tn = len(pairs) - tp - fp - fn
    return dict(n=len(pairs), tp=tp, fp=fp, fn=fn, tn=tn, acc=round((tp+tn)/max(len(pairs),1), 3), acc_ci95=wilson(tp+tn, len(pairs)),
                precision=round(tp/max(tp+fp,1), 3), recall=round(tp/max(tp+fn,1), 3), recall_ci95=wilson(tp, tp+fn),
                specificity=round(tn/max(tn+fp,1), 3), specificity_ci95=wilson(tn, tn+fp))
SUBSETS = {"generated_only": lambda it: it["kind"] == "generated",
           "positives+perturbed": lambda it: it["kind"] in ("generated", "perturbed"),
           "positives+distractor_number": lambda it: it["kind"] == "distractor_number" or (it["kind"] == "generated" and it.get("label_cited", it["label"]) == 1),
           "all": lambda it: True}
def main():
    S = json.load(open(path("data", "calibration_set.json"))); items = S["items"]
    check_keys(items); entail = load_judge(); res = {}; per_item = []
    for norm in (0, 1):
        for mode in ("cited", "any"):
            sc = [score(it, entail, norm, mode) for it in items]
            if norm: per_item.append({i: round(s, 3) if s is not None else None for i, s in enumerate(sc)})
            for th in (0.3, 0.5, 0.7, 0.9):
                for name, f in SUBSETS.items():
                    lab = (lambda it: it.get("label_cited", it["label"])) if mode == "cited" else (lambda it: it["label"])
                    pairs = [(lab(it), s) for it, s in zip(items, sc) if f(it) and s is not None]
                    res[f"{mode}/{'normalized' if norm else 'raw'}@{th}/{name}"] = metrics(pairs, th)
    from collections import Counter
    counts = {f"{k}_{'pos' if l else 'neg'}": v for (k, l), v in Counter((it["kind"], it["label"]) for it in items).items()}
    rows = [dict(id=it["id"], lang=it["lang"], kind=it["kind"], label=it["label"], cited_score=per_item[0][i], any_score=per_item[1][i]) for i, it in enumerate(items)]
    json.dump(dict(counts=counts, results=res, items=rows), open(path("results", "calibration.json"), "w"), ensure_ascii=False, indent=1)
    for k in ("cited/normalized@0.5/generated_only", "cited/normalized@0.5/all", "cited/normalized@0.5/positives+distractor_number",
              "any/normalized@0.5/generated_only", "any/normalized@0.5/all", "any/normalized@0.5/positives+distractor_number"): print(k, res[k])
if __name__ == "__main__": main()
