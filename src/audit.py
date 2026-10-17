"""Citation audit: for each answer sentence, does its cited chunk entail it? Plus answer-language match."""
import json, re, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
NLI = os.environ.get("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
FR_W = set("le la les des du de et est un une pour que qui vous dans pas sur au aux sont avec ou ne".split())
EN_W = set("the and is of to for you in are a an that with your be can not or on as if".split())
def lang_of(t):
    w = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", t.lower()); f = sum(x in FR_W for x in w); e = sum(x in EN_W for x in w)
    return "fr" if f > e else "en"
def claims(ans):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", ans) if len(s.strip()) > 3]
    out = []
    for s in sents:
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", s)]
        out.append((re.sub(r"\s*\[\d+\]", "", s).strip(), cites))
    # a citation on the next fragment alone (e.g. "[1]") attaches to the previous sentence
    merged = []
    for t, c in out:
        if not t and merged: merged[-1][1].extend(c)
        elif t: merged.append([t, c])
    return merged
def main():
    A = json.load(open(os.environ.get("IN","results/answers.json")))["answers"]
    tok = AutoTokenizer.from_pretrained(NLI); m = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    def entail(p, h):
        x = tok(p, h, truncation="only_first", max_length=512, return_tensors="pt")
        with torch.no_grad(): pr = m(**x).logits.softmax(-1)[0]
        return float(pr[lab["entailment"]])
    rows = []; tot = sup = uncited = badref = 0; lm = 0; refusals = 0; grounded = 0
    for a in A:
        lm += lang_of(a["answer"]) == a["lang"]
        if re.search(r"(do not|don't) contain|ne contiennent pas|pas d'information", a["answer"], re.I): refusals += 1
        for t, cites in claims(a["answer"]):
            tot += 1; r = dict(id=a["id"], lang=a["lang"], claim=t, cites=cites)
            anys = max(entail(src["text"], t) for src in a["sources"]); r["any_source_entail"] = round(anys, 3); grounded += anys >= 0.5
            if not cites: uncited += 1; r["verdict"] = "uncited"
            elif any(c < 1 or c > len(a["sources"]) for c in cites): badref += 1; r["verdict"] = "bad_ref"
            else:
                s = max(entail(a["sources"][c-1]["text"], t) for c in cites); r["entail"] = round(s, 3)
                r["verdict"] = "supported" if s >= 0.5 else "unsupported"; sup += s >= 0.5
            rows.append(r)
    rep = dict(nli=NLI, answers=len(A), claims=tot, citation_faithfulness=round(sup/max(tot,1), 3),
               faithfulness_of_cited=round(sup/max(tot-uncited-badref,1), 3), uncited_rate=round(uncited/max(tot,1), 3),
               bad_ref_rate=round(badref/max(tot,1), 3), answer_language_match=round(lm/len(A), 3), refusals=refusals, grounded_in_any_source=round(grounded/max(tot,1), 3))
    for L in ("en", "fr"):
        rr = [r for r in rows if r["lang"] == L]; rep[f"faithfulness_{L}"] = round(sum(r["verdict"]=="supported" for r in rr)/max(len(rr),1), 3)
    print(json.dumps(rep, indent=1)); json.dump(dict(summary=rep, claims=rows), open(os.environ.get("OUT","results/audit.json"), "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__": main()
