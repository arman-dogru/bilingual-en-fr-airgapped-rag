"""Build results/tables.md from an explicit list of configs (no globbing): retrieval, generation (dev / test),
paired deltas, determinism, abstention, judge calibration, gate-threshold stability and robustness."""
import json, math, os
CONFIGS = ["A_1.5b_same_basic", "B_1.5b_same_fewshot", "H_1.5b_same_fewshot_posthoc",
           "D_0.5b_same_basic", "E_0.5b_same_fewshot", "F_0.5b_mixrr_fewshot", "G_0.5b_mixrr_fewshot_posthoc"]
def wilson(p, n, z=1.96):
    p = round(p * n) / n  # recall is stored rounded; recover the exact count k/n first
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
    if A:
        L += ["", "## Gate threshold stability (1000 bootstrap resamples of DEV; percentiles 2.5 / 50 / 97.5)\n",
              "| Gate | Configs | tau (point) | tau pct | TEST recall pct | TEST false-refusal pct | TEST-oracle tau |", "|---|---|---|---|---|---|---|"]
        seen = {}
        for k, v in A.items():
            if "tau_stability" in v: seen.setdefault(v["score"], []).append(k)
        for score, ks in seen.items():
            v = A[ks[0]]; t = v["tau_stability"]; f = lambda xs: " / ".join(f"{x:.3f}" for x in xs)
            L.append(f"| {score} | {', '.join(c[0] for c in ks)} | {v['tau']:.3f} | {f(t['tau_p2_5_50_97_5'])} | {f(t['test_recall_p2_5_50_97_5'])} | {f(t['test_false_refusal_p2_5_50_97_5'])} | {t['test_oracle_tau']:.3f} |")
    RB = load("results/robustness.json")
    if RB:
        S = RB["summary"]; f = lambda d: f"{d['rate']:.2f} ({d['k']}/{d['n']}) {d['ci95']}"
        L += ["", f"## Robustness ({RB['config']}, tau={S['tau']:.3f}; Wilson 95% CI, rows clustered by question so CIs are optimistic)\n",
              "### In-domain unanswerable questions (data/unanswerable_indomain.json)\n",
              "| Set | n | Gate recall | LLM refusal recall | Gate-or-LLM recall | top reranker score - tau (min, max) |", "|---|---|---|---|---|---|"]
        for k, name in (("indomain_unanswerable", "v01-v20 (off-target)"), ("indomain_near_miss", "w01-w08 (near-miss)")):
            if k in S: u = S[k]; L.append(f"| {name} | {u['n']} | {f(u['gate_recall'])} | {f(u['llm_recall'])} | {f(u['gate_or_llm_recall'])} | {u['top_rr_minus_tau']} |")
        L += ["", "### Prompt injection (data/injection.json; success = injected marker in the answer; strict = src.ask final answer)\n",
              "| Site / template | n (questions) | Success raw | After gate | After gate + strict | Gated | Strict refuses non-hijacked |", "|---|---|---|---|---|---|---|"]
        for k, r in S["injection"].items():
            L.append(f"| {k} | {r['n']} ({r['questions']}) | {f(r['success_raw'])} | {f(r['success_after_gate'])} | {f(r['success_after_gate_and_strict'])} | {f(r['gated'])} | {f(r['strict_refuses_nonhijacked'])} |")
        if "strict_cost_test_F" in S:
            t = S["strict_cost_test_F"]
            L += ["", "### Cost of strict mode on TEST answers (config F, committed answers)\n", "| n | Gated | Raw refusal (ungated) | Strict -> refusal (ungated) | Strict leaves answer unchanged (ungated) | Answered after gate + strict |", "|---|---|---|---|---|---|",
                  f"| {t['n']} | {f(t['gated'])} | {f(t['raw_refusal_ungated'])} | {f(t['strict_refused_ungated'])} | {f(t['strict_unchanged_ungated'])} | {f(t['answered_after_gate_and_strict'])} |"]
    open("results/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
if __name__ == "__main__": main()
