"""Paired, question-clustered bootstrap for config deltas (citation faithfulness and grounding).
Both configs are evaluated on the same resampled questions (EN+FR of a question move together);
answers with zero claims stay in the resample (they contribute 0/0 to the pooled ratio)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from audit import faith, grounded
PAIRS = [("B", "A", "few-shot vs basic prompt (1.5B)"), ("E", "D", "few-shot vs basic prompt (0.5B)"),
         ("F", "E", "mixed-language + rerank vs same-language retrieval (0.5B)"), ("G", "F", "post-hoc attribution (0.5B)"),
         ("H", "B", "post-hoc attribution (1.5B)"), ("B", "E", "1.5B vs 0.5B, same retrieval + prompt"),
         ("G", "B", "0.5B+post-hoc vs 1.5B without (not like-for-like)"), ("G", "H", "0.5B vs 1.5B, both post-hoc")]
def load(letter):
    import glob
    f = glob.glob(f"results/audit_{letter}_*.json"); assert len(f) == 1, f"expected one audit for config {letter}, got {f}"
    return json.load(open(f[0]))
def paired(a, b, split, metric, B=5000, seed=0):
    ra = {}; rb = {}
    for x in a["answers"]:
        if x["split"] in split: ra.setdefault(x["id"], []).append(x)
    for x in b["answers"]:
        if x["split"] in split: rb.setdefault(x["id"], []).append(x)
    qs = sorted(set(ra) & set(rb)); assert qs and len(qs) == len(ra) == len(rb), "configs were run on different questions"
    f = faith if metric == "faithfulness" else grounded
    flat = lambda d, idx: [r for i in idx for r in d[qs[i]]]
    rng = np.random.default_rng(seed); n = len(qs); deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n); deltas.append(f(flat(ra, idx)) - f(flat(rb, idx)))
    d = f(flat(ra, range(n))) - f(flat(rb, range(n))); lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0))
    return dict(delta=round(d, 3), ci95=[round(float(lo), 3), round(float(hi), 3)], p_two_sided=round(float(min(p, 1.0)), 4), n_questions=n,
                significant=bool(lo > 0 or hi < 0))
def main():
    out = []
    for x, y, what in PAIRS:
        try: a, b = load(x), load(y)
        except AssertionError as e: print("skip", x, y, e); continue
        for split in (("dev", "test"), ("test",)):
            for metric in ("faithfulness", "grounded"):
                out.append(dict(comparison=f"{x}-{y}", what=what, split="+".join(split), metric=metric, **paired(a, b, split, metric)))
    json.dump(out, open("results/compare.json", "w"), indent=1)
    for r in out:
        if r["metric"] == "faithfulness": print(r["comparison"], r["split"], r["delta"], r["ci95"], "sig" if r["significant"] else "")
if __name__ == "__main__": main()
