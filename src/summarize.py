"""Build markdown tables (results/tables.md) from retrieval.json, audit_*.json and calibration.json."""
import json, glob, math, os
def wilson(p, n, z=1.96):
    d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return f"[{c-h:.2f}, {c+h:.2f}]"
R = json.load(open("results/retrieval.json")); n = R["n_questions"]; L = [f"## Retrieval (n={n} questions per direction; 95% Wilson CI on passage R@1)\n",
     "| Direction | Reranker | passage R@1 | 95% CI | R@3 | R@5 | doc R@1 | doc R@5 |", "|---|---|---|---|---|---|---|---|"]
for k, v in R["recall"].items():
    d, rr = k.replace(" +rerank", ""), "yes" if "rerank" in k else "no"
    L.append(f"| {d} | {rr} | {v['passage_R@1']:.3f} | {wilson(v['passage_R@1'], n)} | {v['passage_R@3']:.3f} | {v['passage_R@5']:.3f} | {v['doc_R@1']:.3f} | {v['doc_R@5']:.3f} |")
L += ["", "## Generation and citation audit (104 answers = 52 EN + 52 FR; 95% bootstrap CI over answers)\n",
      "| Config | Claims | Uncited | Citation faithfulness | 95% CI | Faithful when cited | Grounded (any source) | 95% CI | Lang match | EN / FR faithfulness |", "|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob("results/audit_[A-Z]_*.json")):
    s = json.load(open(f))["summary"]; name = os.path.basename(f)[6:-5]
    L.append(f"| {name} | {s['claims']} | {s['uncited_rate']:.2f} | {s['citation_faithfulness']:.3f} | {s['citation_faithfulness_ci95']} | {s['faithfulness_of_cited']:.3f} | {s['grounded_in_any_source']:.3f} | {s['grounded_ci95']} | {s['answer_language_match']:.2f} | {s['faithfulness_en']:.2f} / {s['faithfulness_fr']:.2f} |")
C = json.load(open("results/calibration.json")); L += ["", f"## Judge calibration (n={C['n_labeled']} hand-labelled claims)\n", "| Setting | Acc | Precision | Recall |", "|---|---|---|---|"]
for k, v in C["results"].items(): L.append(f"| {k} | {v['acc']} | {v['precision']} | {v['recall']} |")
open("results/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
