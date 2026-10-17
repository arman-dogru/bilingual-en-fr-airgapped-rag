"""Abstention on unanswerable / out-of-corpus questions (data/unanswerable.json, 25 x EN/FR).

Two abstention mechanisms, evaluated separately and combined:
  llm   - the generator itself says the sources do not contain the answer (audit.is_refusal)
  gate  - refuse before generation when the top retrieval score is below a threshold tau.
          tau is chosen on the DEV split only (q01-q26 + u01-u12, max balanced accuracy) and reported on TEST (q27-q52 + u13-u25).
Positive class = 'should abstain' (unanswerable)."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit import is_refusal
def top_score(a): s = a["sources"][0]; return s["rr"] if s.get("rr") is not None else s["score"]
def is_dev(a): return a["id"].startswith("u") and int(a["id"][1:]) <= 12 or a["id"].startswith("q") and int(a["id"][1:]) <= 26
def prf(rows, pred):
    tp = sum(pred(a) and not a.get("answerable", True) for a in rows); fp = sum(pred(a) and a.get("answerable", True) for a in rows)
    fn = sum(not pred(a) and not a.get("answerable", True) for a in rows); tn = len(rows) - tp - fp - fn
    return dict(precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                false_refusal_rate=round(fp / max(fp + tn, 1), 3), answered_unanswerable=fn, n_unanswerable=tp + fn, n_answerable=fp + tn)
def bal(rows, pred):
    r = prf(rows, pred); return (r["recall"] + 1 - r["false_refusal_rate"]) / 2
def main():
    out = {}
    for f in sorted(glob.glob("results/answers_[A-H]_*.json")):
        A = json.load(open(f))["answers"]; name = os.path.basename(f)[8:-5]
        if not any(not a.get("answerable", True) for a in A): continue
        dev = [a for a in A if is_dev(a)]; test = [a for a in A if not is_dev(a)]
        cands = sorted({top_score(a) for a in dev})
        tau = max(cands, key=lambda t: (bal(dev, lambda a: top_score(a) < t), -t))
        gate = lambda a: top_score(a) < tau; llm = lambda a: is_refusal(a["answer"])
        out[name] = dict(tau=round(tau, 4), score="reranker logit" if A[0]["sources"][0].get("rr") is not None else "e5 cosine",
                         test=dict(llm=prf(test, llm), gate=prf(test, gate), gate_or_llm=prf(test, lambda a: gate(a) or llm(a))),
                         dev=dict(llm=prf(dev, llm), gate=prf(dev, gate)))
    json.dump(out, open("results/abstain.json", "w"), indent=1)
    for k, v in out.items(): print(k, "tau", v["tau"], "test", {m: (r["precision"], r["recall"], r["false_refusal_rate"]) for m, r in v["test"].items()})
if __name__ == "__main__": main()
