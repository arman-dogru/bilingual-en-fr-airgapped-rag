"""Judge calibration on data/calibration_set.json (keyed by id + lang + claim text, never by position).

Fails loudly if a 'generated' claim no longer occurs in its source answer. Reports accuracy, precision,
recall (sensitivity) and specificity with Wilson 95% CIs, overall and per negative type, raw vs prefix-normalized."""
import json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit import load_judge, normalize, CITE
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [round(c - h, 3), round(c + h, 3)]
def strip(t): return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(CITE.sub("", t).split())).rstrip(".!? ")
def check_keys(items, A):
    bad = []
    for it in items:
        a = A.get((it["id"], it["lang"]))
        if a is None: bad.append((it["id"], it["lang"], "no such answer")); continue
        if it["kind"] in ("generated", "source_swap") and strip(it["claim"]) not in strip(a["answer"]):
            bad.append((it["id"], it["lang"], it["claim"][:60]))
    if bad: raise SystemExit(f"error: {len(bad)} calibration items do not match their source answers: {bad[:5]}")
def scores(items, A, entail, norm):
    out = []
    for it in items:
        src = A[(it["sources_from"]["id"], it["sources_from"]["lang"])] if "sources_from" in it else A[(it["id"], it["lang"])]
        h = normalize(it["claim"]) if norm else it["claim"]
        out.append(max(entail(s["text"], h) for s in src["sources"]))
    return out
def metrics(items, sc, th):
    tp = sum(s >= th and it["label"] == 1 for it, s in zip(items, sc)); fp = sum(s >= th and it["label"] == 0 for it, s in zip(items, sc))
    fn = sum(s < th and it["label"] == 1 for it, s in zip(items, sc)); tn = len(items) - tp - fp - fn
    r = dict(n=len(items), tp=tp, fp=fp, fn=fn, tn=tn, acc=round((tp+tn)/max(len(items),1), 3), acc_ci95=wilson(tp+tn, len(items)),
             precision=round(tp/max(tp+fp,1), 3), recall=round(tp/max(tp+fn,1), 3), recall_ci95=wilson(tp, tp+fn),
             specificity=round(tn/max(tn+fp,1), 3), specificity_ci95=wilson(tn, tn+fp))
    return r
def main():
    S = json.load(open("data/calibration_set.json")); items = S["items"]
    A = {(a["id"], a["lang"]): a for a in json.load(open("data/calibration_answers.json"))["answers"]}
    check_keys(items, A); entail = load_judge(); res = {}
    subsets = {"generated_only": lambda it: it["kind"] == "generated",
               "positives+perturbed": lambda it: it["kind"] != "source_swap",
               "all": lambda it: True}
    for norm in (0, 1):
        sc = scores(items, A, entail, norm)
        for th in (0.3, 0.5, 0.7, 0.9):
            for name, f in subsets.items():
                idx = [i for i, it in enumerate(items) if f(it)]
                res[f"{'normalized' if norm else 'raw'}@{th}/{name}"] = metrics([items[i] for i in idx], [sc[i] for i in idx], th)
        if norm: per_item = [dict(id=it["id"], lang=it["lang"], kind=it["kind"], label=it["label"], score=round(s, 3)) for it, s in zip(items, sc)]
    from collections import Counter
    counts = {f"{k}_{'pos' if l else 'neg'}": v for (k, l), v in Counter((it["kind"], it["label"]) for it in items).items()}
    json.dump(dict(counts=counts, results=res, items=per_item), open("results/calibration.json", "w"), ensure_ascii=False, indent=1)
    for k in ("normalized@0.5/generated_only", "normalized@0.5/positives+perturbed", "normalized@0.5/all", "raw@0.5/all"): print(k, res[k])
if __name__ == "__main__": main()
