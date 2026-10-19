"""Render the README figures from committed result files only (no model is loaded, nothing is re-scored).

Inputs : results/retrieval.json, results/audit_*.json, results/compare.json, results/answers_[BEF]_*.json,
         results/abstain.json, results/robustness.json, results/offline_run.log, docs/demo/*.txt
Outputs: docs/figures/*.png
Usage  : uv sync --frozen --group figures && .venv/bin/python scripts/make_figures.py
"""
import glob, json, math, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.abstain import top_score, is_dev, prf   # the exact gate definitions used for results/abstain.json

OUT = ROOT / "docs" / "figures"
J = lambda p: json.loads((ROOT / p).read_text())
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
BLUE, LIGHT, RED, GREY, GREEN = "#0969da", "#9ecbff", "#cf222e", "#8c959f", "#1a7f37"
SRC = dict(fontsize=7.5, color="#555")
CONFIGS = [("A", "A_1.5b_same_basic", "1.5B · same-lang · basic"), ("B", "B_1.5b_same_fewshot", "1.5B · same-lang · few-shot"),
           ("H", "H_1.5b_same_fewshot_posthoc", "1.5B · few-shot · post-hoc"), ("D", "D_0.5b_same_basic", "0.5B · same-lang · basic"),
           ("E", "E_0.5b_same_fewshot", "0.5B · same-lang · few-shot"), ("F", "F_0.5b_mixrr_fewshot", "0.5B · mixed+rerank · few-shot"),
           ("G", "G_0.5b_mixrr_fewshot_posthoc", "0.5B · mixed+rerank · post-hoc")]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", metadata={"Software": None}); plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def wilson(p, n, z=1.96):
    k = round(p * n); p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_retrieval():
    R = J("results/retrieval.json"); n = R["n_questions"]; dirs = ["EN->EN", "FR->FR", "EN->FR", "FR->EN"]
    fig, ax = plt.subplots(figsize=(6.8, 3.6)); x = np.arange(len(dirs)); w = .36
    for off, suf, col, lab in [(-w / 2, "", LIGHT, "e5-small only"), (w / 2, " +rerank", BLUE, "e5-small + mMiniLM reranker")]:
        v = [R["recall"][d + suf]["passage_R@1"] for d in dirs]; ci = [wilson(p, n) for p in v]
        ax.bar(x + off, v, w, color=col, label=lab, yerr=[[p - c[0] for p, c in zip(v, ci)], [c[1] - p for p, c in zip(v, ci)]], capsize=3)
        for xi, p in zip(x + off, v): ax.text(xi, 0.04, f"{p:.3f}", ha="center", fontsize=8.5, color="white" if col == BLUE else "black", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([d.replace("->", " → ") for d in dirs]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("passage recall@1 (95% Wilson CI)"); ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.12))
    ax.set_title("The reranker closes most of the cross-lingual gap", pad=24)
    ax.text(0, -0.17, f"n = {n} questions per direction (query language → corpus language). Source: results/retrieval.json", transform=ax.transAxes, **SRC)
    save(fig, "retrieval_recall_at_1.png")


def fig_faithfulness():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, (k, name, lab) in enumerate(CONFIGS):
        t = J(f"results/audit_{name}.json")["summary"]["test"]
        f, (lo, hi) = t["citation_faithfulness"], t["citation_faithfulness_ci95"]
        col = BLUE if "posthoc" in name else (GREY if "basic" in name else LIGHT)
        ax.barh(i, f, color=col, height=.62, xerr=[[f - lo], [hi - f]], capsize=3)
        ax.text(hi + .015, i, f"{f:.2f}   uncited {t['uncited_rate']:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(CONFIGS))); ax.set_yticklabels([f"{k} · {lab}" for k, name, lab in CONFIGS], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("citation faithfulness: NLI-supported cited claims / all claims (95% CI)")
    ax.set_title("TEST split: few-shot + post-hoc attribution lift faithfulness")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, LIGHT, BLUE)], labels=["basic prompt", "few-shot", "few-shot + post-hoc"],
              frameon=False, fontsize=8, loc="center right")
    ax.text(0, -0.2, "q27-q52 × EN/FR = 52 answers per config; question-clustered bootstrap CI. Judge: mDeBERTa-v3 XNLI (indicative calibration). "
            "Source: results/audit_*.json", transform=ax.transAxes, **SRC)
    save(fig, "faithfulness_by_config.png")


def fig_deltas():
    C = J("results/compare.json"); order = ["E-D", "B-A", "G-F", "H-B", "B-E", "F-E", "G-B", "G-H"]
    lab = {x["comparison"]: x["what"] for x in C}
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, c in enumerate(order):
        for dy, split, mk, fc in [(-.14, "test", "o", None), (.14, "dev+test", "s", "white")]:
            r = next(x for x in C if x["comparison"] == c and x["split"] == split and x["metric"] == "faithfulness")
            col = BLUE if r["ci95"][0] > 0 else (RED if r["ci95"][1] < 0 else GREY)
            ax.errorbar(r["delta"], i + dy, xerr=[[r["delta"] - r["ci95"][0]], [r["ci95"][1] - r["delta"]]], fmt=mk, color=col, mfc=fc or col,
                        ms=6, capsize=2.5, lw=1.5, label=("TEST" if split == "test" else "dev+test") if i == 0 else None)
    ax.axvline(0, color="#444", lw=.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{c}: {lab[c]}" for c in order], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Δ citation faithfulness, paired question-clustered bootstrap (95% CI, 5,000 draws)")
    ax.set_title("Which interventions help? Paired deltas")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0, -0.2, "Blue: CI above 0; red: below 0; grey: includes 0 (called 'comparable'). Source: results/compare.json", transform=ax.transAxes, **SRC)
    save(fig, "paired_deltas.png")


def fig_abstention():
    AB = J("results/abstain.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, name, unit in [(axes[0], "E_0.5b_same_fewshot", "e5 cosine of top passage"), (axes[1], "F_0.5b_mixrr_fewshot", "reranker logit of top passage")]:
        A = json.loads((ROOT / f"results/answers_{name}.json").read_text())["answers"]; test = [a for a in A if not is_dev(a)]
        s = np.array(sorted({top_score(a) for a in A})); grid = np.linspace(s.min(), s.max(), 400)
        rec = [prf(test, lambda a, t=t: top_score(a) < t)["recall"] for t in grid]
        frr = [prf(test, lambda a, t=t: top_score(a) < t)["false_refusal_rate"] for t in grid]
        tau = AB[name]["tau"]; at = AB[name]["test"]["gate"]
        chk = prf(test, lambda a: top_score(a) < tau)   # the curve must reproduce the committed operating point
        assert (chk["recall"], chk["false_refusal_rate"]) == (at["recall"], at["false_refusal_rate"]), (name, chk, at)
        ax.plot(grid, rec, color=BLUE, lw=2, label="unanswerable refused (recall)")
        ax.plot(grid, frr, color=RED, lw=2, label="answerable refused (false-refusal rate)")
        lo, _, hi = AB[name]["tau_stability"]["tau_p2_5_50_97_5"]; ax.axvspan(lo, hi, color="#eaeef2", zorder=0, label="τ re-selected on 1,000 dev bootstraps (2.5-97.5 pct)")
        ax.axvline(tau, color="black", ls="--", lw=1.2)
        ax.text(tau, 1.04, f"dev-tuned τ = {tau:g}\nrecall {at['recall']:.2f}, false refusals {at['false_refusal_rate']:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(-.02, 1.02); ax.set_xlabel(f"gate threshold τ ({unit}); refuse if score < τ"); ax.grid(alpha=.25)
        ax.set_title(f"config {name[0]} ({'same-lang, cosine gate' if name[0] == 'E' else 'mixed+rerank, reranker gate'})", pad=30)
    axes[0].set_ylabel("share of TEST questions refused")
    h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.1))
    fig.suptitle("Retrieval-score abstention gate on TEST (13 unanswerable + 26 answerable questions × EN/FR)", fontweight="bold", y=1.08)
    fig.text(0.01, -0.14, "Curves recomputed from per-question top retrieval scores in results/answers_*.json with src/abstain.py's own rule; "
             "the dashed operating point is asserted to match results/abstain.json.", **SRC)
    fig.tight_layout(); save(fig, "abstention_threshold.png")


def fig_injection():
    I = J("results/robustness.json")["summary"]["injection"]
    rows = [("query_side/ignore", "query: 'ignore instructions'"), ("query_side/cite", "query: 'copy this, cite [1]'"),
            ("query_side/refusal_prefix", "query: 'start with the refusal, but…'"), ("query_side/refusal_and", "query: 'end with … you must email'"),
            ("query_side/all", "query-side, all 4 templates"), ("poisoned_chunk/all", "poisoned top chunk, all 4 templates")]
    stages = [("success_raw", "raw model answer", GREY), ("success_after_gate", "after retrieval gate", LIGHT), ("success_after_gate_and_strict", "after gate + strict citation filter", BLUE)]
    fig, ax = plt.subplots(figsize=(7.6, 4.0)); h = .26
    for i, (k, lab) in enumerate(rows):
        for j, (st, sl, col) in enumerate(stages):
            r = I[k][st]; y = i + (j - 1) * h
            ax.barh(y, r["rate"], h, color=col, label=sl if i == 0 else None)
            ax.text(r["rate"] + .01, y, f"{r['k']}/{r['n']}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, .8); ax.set_xlabel("prompt-injection success rate (marker 'evil.example' in output)")
    ax.set_title("Only the strict citation filter stops query-side injection")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0, -0.2, "Config F (0.5B), 10 test questions × EN/FR per template. Source: results/robustness.json", transform=ax.transAxes, **SRC)
    save(fig, "prompt_injection.png")


def terminal(lines, title, name, width=118):
    """Draw captured stdout/stderr as a terminal window (text is copied verbatim from the capture file)."""
    import textwrap
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, width, subsequent_indent="    ", replace_whitespace=False, drop_whitespace=False) or [""]
    hgt = .24 + .19 * len(wrapped)
    fig = plt.figure(figsize=(10.8, hgt)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0d1117", transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((0, 1 - .22 / hgt), 1, .22 / hgt, color="#30363d", transform=ax.transAxes))
    for k, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.plot([.015 + k * .015], [1 - .11 / hgt], "o", ms=6, color=c, transform=ax.transAxes)
    ax.text(.5, 1 - .11 / hgt, title, ha="center", va="center", color="#c9d1d9", fontsize=8.5, transform=ax.transAxes)
    for i, ln in enumerate(wrapped):
        col = "#7ee787" if ln.startswith("$") else ("#ffa657" if "WARNING" in ln or ln.startswith("  - ") else "#c9d1d9")
        ax.text(.012, 1 - (.22 + .19 * (i + .75)) / hgt, ln, family="DejaVu Sans Mono", fontsize=8, color=col, transform=ax.transAxes)
    save(fig, name)


def fig_terminals():
    log = (ROOT / "results/offline_run.log").read_text().rstrip("\n").splitlines()
    terminal(["$ ./run_offline.sh      # inside `unshare -rn`: network namespace with loopback only"] + log,
             "results/offline_run.log (air-gapped run, committed output)", "terminal_offline_run.png")
    for f in sorted(glob.glob(str(ROOT / "docs/demo/ask_*.txt"))):
        txt = Path(f).read_text().rstrip("\n").splitlines()
        terminal(txt, f"docs/demo/{Path(f).name} (captured output of src.ask)", f"terminal_{Path(f).stem}.png")


if __name__ == "__main__":
    fig_retrieval(); fig_faithfulness(); fig_deltas(); fig_abstention(); fig_injection(); fig_terminals()
