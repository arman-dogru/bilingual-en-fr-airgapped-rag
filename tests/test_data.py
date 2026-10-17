import json, re
from rag import validate_query
import pytest
QA = json.load(open("data/qa.json")); U = json.load(open("data/unanswerable.json"))
def test_dev_test_disjoint():
    dev = [q for q in QA if int(q["id"][1:]) <= 26]; test = [q for q in QA if int(q["id"][1:]) > 26]
    assert len(dev) == 26 and len(test) == 26
    assert not {q["id"] for q in dev} & {q["id"] for q in test}
    for L in ("en", "fr"):  # no question text (normalised) appears in both splits
        norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
        assert not {norm(q[L]) for q in dev} & {norm(q[L]) for q in test}
def test_calibration_only_uses_dev_questions():
    S = json.load(open("data/calibration_set.json"))["items"]
    assert all(int(it["id"][1:]) <= 26 for it in S)
def test_unanswerable_ids_and_flags():
    assert len(U) == 25 and all(u["id"].startswith("u") and u["answerable"] is False for u in U)
    assert not {u["en"] for u in U} & {q["en"] for q in QA}
def test_calibration_keys_unique_and_labelled():
    S = json.load(open("data/calibration_set.json"))["items"]
    keys = [(it["kind"], it["id"], it["lang"], it["claim"]) for it in S]
    assert len(keys) == len(set(keys)) and all(it["label"] in (0, 1) for it in S)
@pytest.mark.parametrize("bad", ["", "   ", "?!", "x" * 501, None])
def test_validate_rejects(bad):
    with pytest.raises(ValueError): validate_query(bad)
def test_validate_cleans():
    assert validate_query("  How long  do EI benefits last? ") == "How long do EI benefits last?"
def test_evidence_regex_occurs_in_gold_doc_both_languages():
    from retrieval import chunks
    C = json.load(open("data/corpus.json"))
    for L in ("en", "fr"):
        cs = chunks(C, L)
        for q in QA:
            assert any(c["doc"] == q["doc"] and re.search(q["ev"], c["text"]) for c in cs), (q["id"], L)
