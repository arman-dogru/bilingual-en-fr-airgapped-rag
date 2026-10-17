"""Build results/tables.md from an explicit list of configs (no globbing): retrieval, generation (dev / test),
paired deltas, determinism, abstention and judge calibration."""
import json, math, os
CONFIGS = ["A_1.5b_same_basic", "B_1.5b_same_fewshot", "H_1.5b_same_fewshot_posthoc",
           "D_0.5b_same_basic", "E_0.5b_same_fewshot", "F_0.5b_mixrr_fewshot", "G_0.5b_mixrr_fewshot_posthoc"]
def wilson(p, n, z=1.96):
    d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return f"[{c-h:.2f}, {c+h:.2f}]"
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def main():
    L = []; R = load("results/retrieval.json")
    if R:
        n = R["n_questions"]; L += [f"## Retrieval (n={n} questions per direction; 95% Wilson CI on passage R@1)\n",
             "| Direction | Reranker | passage R@1 | 95% CI | R@3 | R@5 | doc R@1 | doc R@5 |", "|---|---|---|---|---|---|---|---|"]
        for k, v in R["recall"].items():
            L.append(f"| {k.replace(' +rerank', '')} | {'yes' if 'rerank' in k else 'no'} | {v['passage_R@1']:.3f} | {wilson(v['passage_R@1'], n)} | {v['passage_R@3']:.3f} | {v['passage_R@5']:.3f} | {v['doc_R@1']:.3f} | {v['doc_R@5']:.3f} |")
    for split, title in (("test", "TEST split (q27-q52, never used for development)"), ("dev", "DEV split (q01-q26, used to develop prompts/parser/threshold)")):
        L += ["", f"## Generation and citation audit - {title}; 95% CI = bootstrap clustered by question\n",
              "| Config | Answers | Claims | Uncited | Citation faithfulness | 95% CI | Faithful when cited | Grounded (any source) | 95% CI | Lang match | EN / FR |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in CONFIGS:
            a = load(f"results/audit_{c}.json")
            if not a or split not in a["summary"]: L.append(f"| {c} | not run | | | | | | | | | |"); continue
            s = a["summary"][split]
            L.append(f"| {c} | {s['answers']} | {s['claims']} | {s['uncited_rate']:.2f} | {s['citation_faithfulness']:.3f} | {s['citation_faithfulness_ci95']} | {s['faithfulness_of_cited']:.3f} | {s['grounded_in_any_source']:.3f} | {s['grounded_ci95']} | {s['answer_language_match']:.2f} | {s['faithfulness_en']:.2f} / {s['faithfulness_fr']:.2f} |")
    C = load("results/compare.json")
    if C:
        L += ["", "## Paired deltas (citation faithfulness; paired bootstrap over questions, 5000 draws)\n", "| Comparison | What | Split | Delta | 95% CI | p | CI excludes 0 |", "|---|---|---|---|---|---|---|"]
        for r in C:
            if r["metric"] == "faithfulness": L.append(f"| {r['comparison']} | {r['what']} | {r['split']} | {r['delta']:+.3f} | {r['ci95']} | {r['p_two_sided']} | {'yes' if r['significant'] else 'no'} |")
    D = load("results/determinism.json")
    if D:
        L += ["", "## Run-to-run determinism (byte-identical answers on dev+test)\n", "| Rerun | Threads (ref -> rerun) | Identical | Match rate |", "|---|---|---|---|"]
        for k, v in D.items(): L.append(f"| {k} | {v['reference_threads']} -> {v['threads']} | {v['identical']}/{v['n']} | {v['match_rate']:.3f} |")
    A = load("results/abstain.json")
    if A:
        L += ["", "## Abstention on TEST (13 unanswerable x 2 langs vs 26 answerable x 2 langs; tau tuned on dev)\n", "| Config | Mechanism | Precision | Recall | False-refusal rate | Unanswerable answered |", "|---|---|---|---|---|---|"]
        for k, v in A.items():
            for m, r in v["test"].items(): L.append(f"| {k} | {m}{' (tau=%.3f, %s)' % (v['tau'], v['score']) if m != 'llm' else ''} | {r['precision']:.2f} | {r['recall']:.2f} | {r['false_refusal_rate']:.2f} | {r['answered_unanswerable']}/{r['n_unanswerable']} |")
    K = load("results/calibration.json")
    if K:
        L += ["", f"## NLI judge calibration (labels by the Developer agent, single annotator, dev questions; counts {K['counts']})\n",
              "| Setting | n | Acc [95% CI] | Precision | Recall [95% CI] | Specificity [95% CI] |", "|---|---|---|---|---|---|"]
        for k, v in K["results"].items():
            if "@0.5" in k or k.endswith("/all"):
                L.append(f"| {k} | {v['n']} | {v['acc']} {v['acc_ci95']} | {v['precision']} | {v['recall']} {v['recall_ci95']} | {v['specificity']} {v['specificity_ci95']} |")
    open("results/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
if __name__ == "__main__": main()
