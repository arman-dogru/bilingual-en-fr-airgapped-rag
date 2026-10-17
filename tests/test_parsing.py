from audit import split_claims, claims, lang_of, is_refusal, split_of
def test_multi_citation():
    assert claims("EI lasts 26 weeks [1, 2].") == [("EI lasts 26 weeks.", [1, 2])]
def test_citation_range():
    assert claims("See the page for details on this [1-3].")[0][1] == [1, 2, 3]
def test_period_before_citation_not_merged():
    c = claims("The benefit lasts up to 26 weeks.[1] You must apply online today.[2]")
    assert [x[1] for x in c] == [[1], [2]] and len(c) == 2
def test_short_sentences_kept_in_text():
    s = split_claims("Yes [1].")
    assert s == [["Yes.", [1], False]]
def test_short_numeric_claim_scored():
    assert claims("Until age 85 [1].") == [("Until age 85.", [1])]
    assert claims("Il faut 1 095 jours [1]. Oui [2].") == [("Il faut 1 095 jours.", [1])]
def test_abbreviation_not_split():
    assert claims("E.g. you can apply online today [1].") == [("E.g. you can apply online today.", [1])]
def test_bare_citation_attaches_to_previous():
    assert claims("EI sickness lasts 26 weeks.\n[1]") == [("EI sickness lasts 26 weeks.", [1])]
def test_lang_of_not_biased_to_en():
    assert lang_of("Oui.") == "fr" and lang_of("26 semaines [1].") == "fr"
    assert lang_of("12 [1].") == "und"
def test_refusal():
    assert is_refusal("The sources do not contain this information.") and is_refusal("Les sources ne contiennent pas cette information.")
    assert is_refusal("") and not is_refusal("EI lasts 26 weeks [1].")
def test_split_names():
    assert split_of("q01") == "dev" and split_of("q26") == "dev" and split_of("q27") == "test" and split_of("u03") == "unanswerable"
def test_refusal_sentence_not_a_claim():
    assert claims("The sources do not contain this information.") == []
    assert claims("Les sources ne contiennent pas cette information.") == []
    assert claims("EI lasts 26 weeks [1]. The sources do not contain this information.") == [("EI lasts 26 weeks.", [1])]
def test_hedge_is_not_abstention():
    assert not is_refusal("Yes, you can keep contributing [1]. The document does not provide the date.")
    assert is_refusal("The sources do not contain this information.") and is_refusal("Les sources ne contiennent pas cette information.")
