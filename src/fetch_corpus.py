"""Fetch EN/FR page pairs from Canada.ca (Open Government Licence - Canada)."""
import json, re, sys, time, requests
from bs4 import BeautifulSoup
EN = """services/benefits/ei/ei-regular-benefit/eligibility.html
services/benefits/ei/ei-maternity-parental.html
services/benefits/ei/ei-sickness.html
services/benefits/ei/caregiving.html
services/benefits/publicpensions/cpp.html
services/benefits/publicpensions/cpp/old-age-security.html
services/benefits/publicpensions/cpp/old-age-security/guaranteed-income-supplement.html
services/child-family-services/benefits/child-benefit.html
revenue-agency/services/child-family-benefits/canada-child-benefit-overview.html
revenue-agency/services/child-family-benefits/goods-services-tax-harmonized-sales-tax-gst-hst-credit.html
revenue-agency/services/tax/individuals/topics/rrsps-related-plans.html
revenue-agency/services/tax/individuals/topics/tax-free-savings-account.html
services/benefits/dental/dental-care-plan.html
immigration-refugees-citizenship/services/canadian-passports.html
immigration-refugees-citizenship/services/canadian-citizenship/become-canadian-citizen/eligibility.html
immigration-refugees-citizenship/services/study-canada/study-permit.html
immigration-refugees-citizenship/services/work-canada/permit.html
immigration-refugees-citizenship/services/immigrate-canada/express-entry.html
services/jobs/workplace/federal-labour-standards/vacation.html
services/jobs/workplace/federal-labour-standards/holidays.html
services/benefits/education/student-aid.html
services/benefits/disability/canada-disability-benefit.html
health-canada/services/drugs-health-products/medical-use-cannabis.html
services/benefits/housing/first-home-savings-account.html
revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/deductions-credits-expenses/canada-workers-benefit.html
services/benefits/ei/ei-fishing.html""".split()
def get(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "bilingual-rag-research/0.1"}); r.raise_for_status(); return r.text
def text(html):
    s = BeautifulSoup(html, "html.parser"); main = s.find("main") or s.body
    for t in main.select("script,style,nav,.pagedetails,#wb-dtmd,aside"): t.decompose()
    title = (s.find("h1") or s.title).get_text(" ", strip=True)
    paras = [re.sub(r"\s+", " ", e.get_text(" ", strip=True)) for e in main.find_all(["h2","h3","p","li","td"])]
    return title, [p for p in paras if len(p) > 25]
out = []
for i, path in enumerate(EN):
    try:
        url = f"https://www.canada.ca/en/{path}"; h = get(url)
        m = re.search(r'lang="fr"[^>]*href="([^"]+)"', h) or re.search(r'href="([^"]+)"[^>]*lang="fr"', h)
        fr = m.group(1); fr = fr if fr.startswith("http") else "https://www.canada.ca" + fr
        hf = get(fr); te, pe = text(h); tf, pf = text(hf)
        out.append(dict(doc_id=f"d{i:02d}", en=dict(url=url, title=te, paras=pe), fr=dict(url=fr, title=tf, paras=pf),
                        licence="Open Government Licence - Canada"))
        print("ok", i, te[:60], len(pe), len(pf), flush=True)
    except Exception as e: print("FAIL", path, e, flush=True)
    time.sleep(0.5)
json.dump(out, open("data/corpus.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "pairs")
