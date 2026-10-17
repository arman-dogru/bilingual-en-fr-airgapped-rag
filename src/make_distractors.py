"""Build distractor-number calibration negatives from dev answers: replace a number that is in the cited chunk with a
different number that occurs in another retrieved chunk of the same answer (label 0 against the cited chunk)."""
import json,re,sys
from .audit import claims
NUM=re.compile(r"(?<![\d,.$])(?<!\d )(\d{1,4})(?![\d,])(?! \d{3})")
def ok(n,m): n,m=int(n),int(m); return m!=n and ((n<1000 and 1<m<1000) or (n>=1900 and m>=1900))
has=lambda x,t: re.search(rf"(?<!\d)(?<!\d ){x}(?!\d)(?! \d{{3}})",t)
U=[];done=set()
for f in ["results/answers_B_1.5b_same_fewshot.json","results/answers_A_1.5b_same_basic.json","results/answers_E_0.5b_same_fewshot.json"]:
  for a in json.load(open(f))["answers"]:
    if not a["id"].startswith("q") or int(a["id"][1:])>26 or (a["id"],a["lang"]) in done: continue
    for t,c in claims(a["answer"]):
        if (a["id"],a["lang"]) in done or len(c)!=1 or c[0]>len(a["sources"]): continue
        cited=a["sources"][c[0]-1]["text"]
        for n in NUM.findall(t):
            if not has(n,cited): continue
            pick=None
            for j,s in enumerate(a["sources"]):
                if j==c[0]-1 or has(n,s["text"]): continue
                ms=[m for m in NUM.findall(s["text"]) if ok(n,m) and not has(m,cited)]
                if ms: pick=(ms[0],j+1); break
            if pick:
                m=pick[0]; neg=re.sub(rf"(?<!\d)(?<!\d ){n}(?!\d)(?! \d{{3}})",m,t,count=1)
                U.append(dict(kind="distractor_number",id=a["id"],lang=a["lang"],claim=neg,label=0,cites=c,source_answers="results/"+f.split('/')[1],
                    derived_from=t,note=f"number {n} (in cited chunk [{c[0]}]) replaced by {m}, which occurs in retrieved chunk [{pick[1]}] but not in the cited chunk"))
                done.add((a["id"],a["lang"])); break
print(len(U))
for x in U: print(x["id"],x["lang"],x["source_answers"][16:18],"|",x["claim"][:120])
json.dump(U, open(sys.argv[1] if len(sys.argv) > 1 else "distractors.json", "w"), ensure_ascii=False, indent=1)
