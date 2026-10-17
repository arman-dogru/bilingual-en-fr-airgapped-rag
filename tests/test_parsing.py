from src.audit import split_claims, claims, lang_of, is_refusal, split_of
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
def test_near_verbatim_refusals():
    for t in ["Les sources mentionnées ne contiennent pas cette information.", "No, the sources do not contain this information.",
              "Je suis désolé, mais je ne trouve aucune information sur l'Everest dans les sources fournies.",
              "The sources do not contain this information [1]."]:
        assert is_refusal(t) and claims(t) == [], t
    for t in ["No, you cannot get EI if you lost your job through your own fault [1].", "One parent cannot receive more than 61 weeks [1]."]:
        assert not is_refusal(t), t
HEDGED_PARTIAL = [  # PR #12 review M1: hedge-first or affirmative sentences carrying a claim are NOT refusals
    "The sources do contain this information [1].",
    "The sources do contain this information.",
    "The sources do not mention the exact date, but EI sickness benefits last up to 26 weeks [1].",
    "The document does not provide a phone number, but you can apply online through My Service Canada Account [2].",
    "Les sources ne mentionnent pas la date exacte, mais les prestations durent jusqu'à 26 semaines [1].",
    "I could not find any specific information on the exact date, but payments start in July 2025 [1].",
    "Sorry, but the sources do not provide that; EI pays 55% of earnings up to $729 per week [1].",
    "Je ne trouve aucune information sur la date, cependant les paiements commencent en juillet [1].",
    "Les sources ne fournissent pas le numéro; vous pouvez faire une demande en ligne [1].",
]
# Conjunction-joined refusal paraphrases (PR #12 re-review M1') may still be *classified* as refusals for the abstention
# metric; that classification is never trusted user-side (see tests/test_cli.py::test_reviewer_strings_through_final_answer).
def test_hedged_partial_answers_are_not_refusals():
    for t in HEDGED_PARTIAL:
        assert not is_refusal(t) and len(claims(t)) == 1, t
