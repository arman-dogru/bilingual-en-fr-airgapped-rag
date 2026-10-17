"""Agreement of NLI judge (raw vs prefix-normalized claims) with hand labels, across thresholds."""
import json
L = {int(k): v for k, v in json.load(open("data/calibration_labels.json"))["labels"].items() if v is not None}
out = {}
for n in (0, 1):
    C = json.load(open(f"results/calib_norm{n}.json"))["claims"]
    for th in (0.3, 0.5, 0.7, 0.9):
        tp = sum(C[i]["any_source_entail"] >= th and L[i] for i in L); fp = sum(C[i]["any_source_entail"] >= th and not L[i] for i in L)
        fn = sum(C[i]["any_source_entail"] < th and L[i] for i in L); tn = len(L) - tp - fp - fn
        out[f"{'normalized' if n else 'raw'}@{th}"] = dict(acc=round((tp+tn)/len(L), 3), precision=round(tp/max(tp+fp,1), 3), recall=round(tp/max(tp+fn,1), 3), tp=tp, fp=fp, fn=fn, tn=tn)
print(json.dumps(out, indent=1)); json.dump(dict(n_labeled=len(L), results=out), open("results/calibration.json", "w"), indent=1)
