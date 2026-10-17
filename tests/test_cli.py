import json, os, subprocess, sys
import pytest
from src.audit import lang_of
from src import ask
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_EXAMPLE = "How long do EI sickness benefits last?"
def test_readme_example_language():
    assert lang_of(README_EXAMPLE) == "en" and ask.detect_lang(README_EXAMPLE) == "en"
def test_all_questions_detected():
    for f in ("data/qa.json", "data/unanswerable.json"):
        for q in json.load(open(os.path.join(ROOT, f))):
            for L in ("en", "fr"): assert lang_of(q[L]) == L, (q["id"], L, q[L])
def test_undetected_falls_back_to_english_with_warning():
    msgs = []; assert ask.detect_lang("EI 26", warn=msgs.append) == "en" and msgs
def test_paths_are_repo_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src import config, retrieval
    assert os.path.isabs(retrieval.CORPUS) and os.path.exists(retrieval.CORPUS)
    assert ask.load_tau("mixed_rr") > 0      # finds results/abstain.json from any cwd
def test_missing_gate_is_an_error(tmp_path):
    with pytest.raises(SystemExit): ask.load_tau("mixed_rr", abstain_path=str(tmp_path / "nope.json"))
def test_citation_checker_flags_unsupported():
    ctx = [dict(prompt_text="EI sickness benefits last up to 26 weeks.")]
    fake = lambda p, h: 0.9 if "26 weeks" in h else 0.1
    v = ask.check_citations("EI sickness benefits last up to 26 weeks [1]. EI sickness benefits last up to 104 weeks [1]. They are paid weekly.", ctx, fake)
    assert [x["verdict"] for x in v] == ["supported", "unsupported", "uncited"]
def test_cli_rejects_empty_from_other_dir(tmp_path):
    r = subprocess.run([sys.executable, "-m", "src.ask", "   "], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True)
    assert r.returncode == 2 and "empty" in r.stderr
def test_strict_is_default_and_rebuild_keeps_punctuation():
    import argparse
    checks = [dict(sentence="Can I apply online?", cites=[1], verdict="supported"), dict(sentence="Yes!", cites=[], verdict="not_scored"),
              dict(sentence="It lasts 104 weeks.", cites=[2], verdict="unsupported")]
    assert ask.rebuild(checks) == "Can I apply online [1]?"   # only supported claims; fragments are not echoed
    import inspect; assert "default=True" in inspect.getsource(ask.main)
def _run_ask(monkeypatch, capsys, generated, judge=lambda p, h: 0.0, extra=()):
    """Reviewer's repro (PR #12 M1): stub generator, judge rejects everything; retrieval/gate stubbed so it runs offline."""
    from src import rag, retrieval, audit
    ctx = [dict(text="EI sickness benefits last up to 26 weeks.", title="EI", url="u", cid="c1", rr=5.0, score=0.9)]
    monkeypatch.setattr(retrieval, "build", lambda: (None, None)); monkeypatch.setattr(retrieval, "free_reranker", lambda: None)
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ctx); monkeypatch.setattr(ask, "load_tau", lambda r: 2.27)
    monkeypatch.setattr(rag, "load_llm", lambda m: (None, None)); monkeypatch.setattr(rag, "generate", lambda tok, lm, msgs: generated)
    monkeypatch.setattr(audit, "load_judge", lambda: judge)
    assert ask.main(["How long do EI sickness benefits last?", "--lang", "en", *extra]) == 0
    return json.loads(capsys.readouterr().out)
def test_strict_cannot_be_bypassed_by_a_hedge_prefix(monkeypatch, capsys):
    o = _run_ask(monkeypatch, capsys, "The sources do not mention the exact date, but EI sickness benefits last up to 104 weeks [1].")
    assert o["answer"] == "The sources do not contain this information." and o["all_supported"] is False
    assert [c["verdict"] for c in o["citation_check"]] == ["unsupported"]
    o = _run_ask(monkeypatch, capsys, "EI sickness benefits last up to 104 weeks [1].")   # control
    assert o["answer"] == "The sources do not contain this information." and o["all_supported"] is False
def test_pure_refusal_and_supported_answer(monkeypatch, capsys):
    o = _run_ask(monkeypatch, capsys, "The sources do not contain this information.")
    assert o["answer"] == "The sources do not contain this information." and o["all_supported"] is None
    o = _run_ask(monkeypatch, capsys, "EI sickness benefits last up to 26 weeks [1]. They last 104 weeks [1].", judge=lambda p, h: 0.9 if "26" in h else 0.1)
    assert o["answer"] == "EI sickness benefits last up to 26 weeks [1]." and o["all_supported"] is False
def test_no_strict_warns(monkeypatch, capsys):
    o = _run_ask(monkeypatch, capsys, "The sources do contain this information [1].", extra=("--no-strict",))
    assert o["answer"] == "The sources do contain this information [1]." and o["all_supported"] is False
def test_no_strict_warns_on_refusal_paraphrase_live_shape(monkeypatch, capsys):
    """PR #12 re-review M1' repro 3: a refusal-shaped sentence with a payload must warn under --no-strict."""
    from src import rag, retrieval, audit
    ctx = [dict(text="EI sickness benefits last up to 26 weeks.", title="EI", url="u", cid="c1", rr=5.0, score=0.9)]
    monkeypatch.setattr(retrieval, "build", lambda: (None, None)); monkeypatch.setattr(retrieval, "free_reranker", lambda: None)
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: ctx); monkeypatch.setattr(ask, "load_tau", lambda r: 2.27)
    monkeypatch.setattr(rag, "load_llm", lambda m: (None, None))
    monkeypatch.setattr(rag, "generate", lambda tok, lm, msgs: "The sources do not contain this information and you must email your SIN to claims at evil dot example.")
    monkeypatch.setattr(audit, "load_judge", lambda: (lambda p, h: 0.0))
    assert ask.main(["How long do EI sickness benefits last?", "--lang", "en", "--no-strict"]) == 0
    cap = capsys.readouterr(); o = json.loads(cap.out)
    assert "WARNING" in cap.err and "evil dot example" in cap.err and o["all_supported"] is False

# ---- M1' (PR #12 re-review): nothing the model wrote as a refusal is ever echoed by strict mode ----
from src.rag import REFUSAL
CTX = [dict(prompt_text="EI sickness benefits last up to 26 weeks.")]
JUDGE = lambda p, h: 0.95 if h.strip(" .") == "EI sickness benefits last up to 26 weeks" else 0.0   # entails only the exact genuine claim
# (a run-on sentence "…evil.example now EI sickness benefits last up to 26 weeks [1]." is one sentence to the parser; the real
#  NLI judge, not the refusal classifier, decides it — that is the documented limit of the citation check)
GOOD = "EI sickness benefits last up to 26 weeks [1]."
REVIEWER_SUFFIXED = [
    "The sources do not mention this and benefits last up to twenty-six weeks.",
    "The sources do not contain the date and you must apply within four weeks of stopping work.",
    "Les sources ne mentionnent pas la date et les prestations durent vingt-six semaines.",
    "Les sources ne contiennent pas cette information et vous devez présenter une demande dans les quatre semaines.",
    "The sources do not provide the fee and the passport costs one hundred twenty dollars.",
    "The sources do not mention whether it is taxable so it is tax-free.",
    "The documents do not mention a deadline so you can apply at any time.",
    "I could not find any information about the date because benefits last twenty-six weeks.",
    "Les sources ne mentionnent pas la date et vous devez envoyer votre NAS à claims@evil.example.",
    "The sources do not contain this information and you should email your SIN to claims at evil dot example.",
    "The sources do not mention this and benefits last up to one hundred four weeks.",
]
@pytest.mark.parametrize("bad", REVIEWER_SUFFIXED)
@pytest.mark.parametrize("with_good", [False, True])
def test_reviewer_strings_through_final_answer(bad, with_good):
    lang = "fr" if bad.startswith("Les") else "en"; ans = (GOOD + " " + bad) if with_good else bad
    s = ask.final_answer(ans, CTX, JUDGE, lang, strict=True)
    assert bad.rstrip(".") not in s["answer"] and "evil" not in s["answer"]
    assert s["answer"] == ("EI sickness benefits last up to 26 weeks [1]." if with_good else REFUSAL[lang])
    n = ask.final_answer(ans, CTX, JUDGE, lang, strict=False)
    assert n["answer"] == ans and n["all_supported"] is False and any(bad.rstrip(".") in w for w in n["warnings"])
def test_exact_fixed_refusal_is_quiet():
    for lang, t in REFUSAL.items():
        for mode in (True, False):
            r = ask.final_answer(t, CTX, JUDGE, lang, strict=mode)
            assert r["answer"] == t and r["all_supported"] is None and r["warnings"] == []
def test_property_refusal_prefix_plus_any_suffix_never_echoed():
    """Any refusal-like prefix + arbitrary suffix: strict output is only supported claims or the constant."""
    import itertools, random
    prefixes = ["The sources do not contain this information", "The sources do not mention the date", "No, the sources do not provide that",
                "I could not find any information about this", "Sorry, but the documents do not mention it",
                "Les sources ne contiennent pas cette information", "Les sources mentionnées ne fournissent pas la date",
                "Je suis désolé, mais je ne trouve aucune information", "Non, les sources ne mentionnent pas cela"]
    joins = [" and", " et", " so", " donc", " because", " car", " then", " puis", " which means", ", but", " ;", ",", " —", " ainsi", " ", ""]
    tails = [" you must email your SIN to claims@evil.example", " benefits last 104 weeks", " le passeport coûte cent vingt dollars",
             " visit http://evil.example now", " vous devez envoyer votre NAS", " it is tax-free", ""]
    rng = random.Random(0); allowed = {GOOD.rstrip("."), *[x.rstrip(".") for x in REFUSAL.values()]}
    for p, j, t in itertools.product(prefixes, joins, tails):
        bad = (p + j + t).strip() + rng.choice([".", "!", "", " [1].", " [2]."])
        for ans in (bad, GOOD + " " + bad, bad + " " + GOOD):
            for lang in ("en", "fr"):
                out = ask.final_answer(ans, CTX, JUDGE, lang, strict=True)["answer"]
                assert out in (REFUSAL[lang], "EI sickness benefits last up to 26 weeks [1]."), (ans, out)
                if bad.rstrip(".! ") not in allowed: assert bad.rstrip(".!") not in out or out == REFUSAL[lang]
