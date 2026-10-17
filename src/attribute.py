"""Post-hoc citation attribution: give every uncited answer sentence the retrieved source with the highest
embedding similarity (e5). Uses a different model than the NLI judge to avoid circularity.
Sentences that are not scored (short ones like "Yes.") are kept verbatim, so no answer text is deleted."""
import json, os, sys
from .audit import split_claims, is_refusal
from .config import rev
def main():
    inp, out = os.environ.get("IN"), os.environ.get("OUT")
    if not inp or not out: raise SystemExit("error: set IN=<answers json> and OUT=<answers json>")
    if not os.path.exists(inp): raise SystemExit(f"error: {inp} not found")
    from sentence_transformers import SentenceTransformer
    emb = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
    A = json.load(open(inp)); m = SentenceTransformer(emb, device="cpu", revision=rev(emb))
    for a in A["answers"]:
        a["answer_raw"] = a["answer"]
        if is_refusal(a["answer"]) or not a["sources"]: continue   # never attach citations to a refusal
        S = m.encode(["passage: " + s["text"] for s in a["sources"]], normalize_embeddings=True); parts = []
        for t, c, scored in split_claims(a["answer"]):
            if not c and scored:
                v = m.encode(["query: " + t], normalize_embeddings=True)[0]; c = [int((S @ v).argmax()) + 1]
            end = t[-1] if t[-1] in ".!?" else "."
            parts.append(t.rstrip(".!?") + (" " + "".join(f"[{x}]" for x in c) if c else "") + end)
        a["answer"] = " ".join(parts)
    A["post_hoc_attribution"] = True; json.dump(A, open(out, "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__": main()
