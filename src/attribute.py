"""Post-hoc citation attribution: give every uncited answer sentence the retrieved source
with the highest embedding similarity (e5). Uses a different model than the NLI judge to avoid circularity."""
import json, os, re, sys
sys.path.insert(0, "src"); from audit import claims
from sentence_transformers import SentenceTransformer
A = json.load(open(os.environ["IN"])); m = SentenceTransformer(os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small"), device="cpu")
for a in A["answers"]:
    S = m.encode(["passage: " + s["text"] for s in a["sources"]], normalize_embeddings=True); parts = []
    for t, c in claims(a["answer"]):
        if not c:
            v = m.encode(["query: " + t], normalize_embeddings=True)[0]; c = [int((S @ v).argmax()) + 1]
        parts.append(t.rstrip(".") + " " + "".join(f"[{x}]" for x in c) + ".")
    a["answer_raw"] = a["answer"]; a["answer"] = " ".join(parts)
A["post_hoc_attribution"] = True; json.dump(A, open(os.environ["OUT"], "w"), ensure_ascii=False, indent=1)
