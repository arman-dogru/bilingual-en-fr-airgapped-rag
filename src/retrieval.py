"""Chunk EN/FR corpus, embed with multilingual-e5-small, evaluate recall@k for 4 directions."""
import json, re, numpy as np, os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentence_transformers import SentenceTransformer
from config import rev
CORPUS = os.environ.get("CORPUS", "data/corpus.json")
CORPUS_SHA256 = "523784e7807d5e93173cbcbbc787fb813af92119aeaab2e0f6bf7d3f1c28318a"
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
    raw = open(CORPUS, "rb").read()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256 and os.environ.get("ALLOW_CORPUS_CHANGE") != "1":
        raise SystemExit(f"error: {CORPUS} sha256 differs from the snapshot the results were produced on (set ALLOW_CORPUS_CHANGE=1 to override)")
    corpus = json.loads(raw); m = model or SentenceTransformer(EMB, device="cpu", revision=rev(EMB))
    idx = {}
    for lang in ("en", "fr"):
        cs = chunks(corpus, lang)
        E = m.encode(["passage: " + c["title"] + ". " + c["text"] for c in cs], normalize_embeddings=True, batch_size=32)
        idx[lang] = (cs, E)
    idx["mixed"] = (idx["en"][0] + idx["fr"][0], np.vstack([idx["en"][1], idx["fr"][1]]))
    return m, idx
RR = os.environ.get("RERANKER", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"); _rr = None
def rerank(q, cands, k):
    global _rr
    if _rr is None:
        from sentence_transformers import CrossEncoder; _rr = CrossEncoder(RR, device="cpu", revision=rev(RR))
    s = _rr.predict([(q, c["title"] + ". " + c["text"][:1000]) for c in cands], batch_size=16)
    o = np.argsort(-s)[:k]; return [dict(cands[i], rr=float(s[i])) for i in o]
def free_reranker():
    global _rr; _rr = None
def retrieve(m, idx, q, lang, k=5, rr=False, pool=20):
    if not rr: return search(m, idx, q, lang, k)
    return rerank(q, search(m, idx, q, lang, pool), k)
def search(m, idx, q, lang, k=5):
    cs, E = idx[lang]; v = m.encode(["query: " + q], normalize_embeddings=True)[0]
    s = E @ v; o = np.argsort(-s)[:k]; return [dict(cs[i], score=float(s[i])) for i in o]
def hit(c, item): return c["doc"] == item["doc"] and re.search(item["ev"], c["text"])
if __name__ == "__main__":
    qa = json.load(open("data/qa.json")); m, idx = build(); res = {}; misses = []
    for rr in (False, True):
        for ql in ("en", "fr"):
            for cl in ("en", "fr", "mixed"):
                r = {k: 0 for k in (1, 3, 5)}; dr = {k: 0 for k in (1, 3, 5)}
                for it in qa:
                    top = retrieve(m, idx, it[ql], cl, 5, rr)
                    for k in r:
                        r[k] += bool(any(hit(c, it) for c in top[:k])); dr[k] += any(c["doc"] == it["doc"] for c in top[:k])
                    if not any(hit(c, it) for c in top): misses.append(dict(rr=rr, dir=f"{ql}->{cl}", q=it["id"], got=[c["cid"] for c in top]))
                n = len(qa); key = f"{ql.upper()}->{cl.upper()}{' +rerank' if rr else ''}"
                res[key] = {**{f"passage_R@{k}": round(r[k]/n, 3) for k in r}, **{f"doc_R@{k}": round(dr[k]/n, 3) for k in dr}}
                print("RES", key, res[key], flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(dict(model=EMB, reranker=RR, n_questions=len(qa), n_chunks={l: len(idx[l][0]) for l in idx}, recall=res, misses=misses), open("results/retrieval.json", "w"), indent=1)
