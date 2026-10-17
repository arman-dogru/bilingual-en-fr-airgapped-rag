"""RAG answers with inline [n] citations using a small local LLM on CPU."""
import json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "src"); from retrieval import build, search
LLM = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"); K = 3
SYS = {"en": "Answer the question in English using ONLY the numbered sources. After every sentence, cite the source(s) it uses like [1]. If the sources do not contain the answer, say so. Keep it to 1-3 sentences.",
       "fr": "Réponds à la question en français en utilisant UNIQUEMENT les sources numérotées. Après chaque phrase, cite la ou les sources utilisées comme [1]. Si les sources ne contiennent pas la réponse, dis-le. Réponds en 1 à 3 phrases."}
def main():
    torch.set_num_threads(os.cpu_count()); qa = json.load(open("data/qa.json"))
    m, idx = build(); tok = AutoTokenizer.from_pretrained(LLM); lm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.bfloat16); lm.eval()
    out = []
    for it in qa:
        for lang in ("en", "fr"):
            ctx = search(m, idx, it[lang], lang, K)
            src = "\n".join(f"[{i+1}] ({c['title']}) {c['text'][:700]}" for i, c in enumerate(ctx))
            msgs = [{"role": "system", "content": SYS[lang]}, {"role": "user", "content": f"Sources:\n{src}\n\nQuestion: {it[lang]}"}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            with torch.no_grad(): g = lm.generate(**ids, max_new_tokens=120, do_sample=False)
            ans = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            out.append(dict(id=it["id"], lang=lang, q=it[lang], answer=ans, sources=[dict(cid=c["cid"], doc=c["doc"], url=c["url"], text=c["text"]) for c in ctx]))
            print(it["id"], lang, ans[:120].replace("\n", " "), flush=True)
    os.makedirs("results", exist_ok=True); json.dump(dict(llm=LLM, answers=out), open(os.environ.get("OUT","results/answers.json"), "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__": main()
