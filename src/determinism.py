"""Run-to-run drift: fraction of answers that are byte-identical between a config and its rerun(s)."""
import json, os, glob
def load(p): return {(a["id"], a["lang"]): a["answer"] for a in json.load(open(p))["answers"]}
def main():
    out = {}
    for p in sorted(glob.glob("results/answers_rerun_*.json")):
        name = os.path.basename(p)[len("answers_rerun_"):-5]; base = name.replace("_threads8", "")
        ref = f"results/answers_{base}.json"
        if not os.path.exists(ref): continue
        a, b = load(ref), load(p); keys = sorted(set(a) & set(b))
        out[name] = dict(reference=os.path.basename(ref), n=len(keys), identical=sum(a[k] == b[k] for k in keys),
                         match_rate=round(sum(a[k] == b[k] for k in keys) / max(len(keys), 1), 3),
                         threads=json.load(open(p))["env"]["threads"], reference_threads=json.load(open(ref))["env"]["threads"])
    json.dump(out, open("results/determinism.json", "w"), indent=1); print(json.dumps(out, indent=1))
if __name__ == "__main__": main()
