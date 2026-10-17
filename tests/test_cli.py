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
    assert ask.rebuild(checks) == "Can I apply online [1]? Yes!"
    import inspect; assert "default=True" in inspect.getsource(ask.main)
