"""Chunk EN/FR corpus, embed with multilingual-e5-small, evaluate recall@k for 4 directions."""
import json, re, numpy as np, os
from sentence_transformers import SentenceTransformer
EMB = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
def chunks(corpus, lang, win=3):
    out = []
    for d in corpus:
        ps = d[lang]["paras"]
        for i in range(0, max(1, len(ps)), win - 1):
            t = " ".join(ps[i:i + win])
            if t: out.append(dict(cid=f"{d['doc_id']}-{lang}-{i}", doc=d["doc_id"], lang=lang, title=d[lang]["title"], url=d[lang]["url"], text=t))
    return out
def build(model=None):
    corpus = json.load(open("data/corpus.json")); m = model or SentenceTransformer(EMB, device="cpu")
    idx = {}
    for lang in ("en", "fr"):
        cs = chunks(corpus, lang)
        E = m.encode(["passage: " + c["title"] + ". " + c["text"] for c in cs], normalize_embeddings=True, batch_size=32)
        idx[lang] = (cs, E)
    return m, idx
def search(m, idx, q, lang, k=5):
    cs, E = idx[lang]; v = m.encode(["query: " + q], normalize_embeddings=True)[0]
    s = E @ v; o = np.argsort(-s)[:k]; return [dict(cs[i], score=float(s[i])) for i in o]
def hit(c, item): return c["doc"] == item["doc"] and re.search(item["ev"], c["text"])
if __name__ == "__main__":
    qa = json.load(open("data/qa.json")); m, idx = build(); res = {}; misses = []
    for ql in ("en", "fr"):
        for cl in ("en", "fr"):
            r = {k: 0 for k in (1, 3, 5)}; dr = {k: 0 for k in (1, 3, 5)}
            for it in qa:
                top = search(m, idx, it[ql], cl, 5)
                for k in r:
                    r[k] += any(hit(c, it) for c in top[:k]); dr[k] += any(c["doc"] == it["doc"] for c in top[:k])
                if not any(hit(c, it) for c in top[:5]): misses.append(dict(dir=f"{ql}->{cl}", q=it["id"], got=[c["doc"] for c in top]))
            n = len(qa); res[f"{ql.upper()}->{cl.upper()}"] = {**{f"passage_R@{k}": round(r[k]/n, 3) for k in r}, **{f"doc_R@{k}": round(dr[k]/n, 3) for k in dr}}
    print(json.dumps(res, indent=1)); os.makedirs("results", exist_ok=True)
    json.dump(dict(model=EMB, n_questions=len(qa), n_chunks={l: len(idx[l][0]) for l in idx}, recall=res, misses=misses), open("results/retrieval.json", "w"), indent=1)
