"""Clean the Adaption export into the final training set.

Rules (owner OK 20 Nov):
  seed rows ('none')      -> keep OUR original prompt + completion (verified), drop Adaption's rewrite.
  'similar' scheme rows   -> keep only if full checklist score, not unsafe, no Bangladesh text, answer <= MAXW words,
                             and every money amount / age number matches the official card of the scheme it names.
                             Rows about schemes we have no card for are kept only if they state no money amounts.
  general rows            -> keep if checklist >= 80%, not unsafe, not a canned "I only answer WB schemes" refusal
                             (a small number of refusals kept on purpose), answer <= MAXW words; capped to ~40% of total.
Outputs data/clean_train.jsonl, data/clean_train_upload.csv, data/clean_report.txt
"""
import csv, json, random, re, sys
from collections import Counter

random.seed(7)
SRC = 'data/adaption_export/wb_gov_scheme_qa_bn.jsonl'
MAXW = 550  # scheme rows; general rows use MAXW_GEN
MAXW_GEN = 450
cards = json.load(open('facts/cards_bn.json', encoding='utf-8'))
cards.pop('_meta', None)

BD = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
def nums(s):
    s = s.translate(BD)
    return {int(x.replace(',', '')) for x in re.findall(r'\d[\d,]*', s) if x.replace(',', '').isdigit()}

# names / aliases used to spot which scheme a row talks about
ALIAS = {
    'aywb': ['অন্নপূর্ণা'], 'lbs-wb': ['লক্ষ্মীর ভান্ডার', 'লক্ষ্মীর ভাণ্ডার'], 'kbs': ['কৃষক বন্ধু', 'কৃষকবন্ধু'],
    'wbkanyashree': ['কন্যাশ্রী', 'কন্যশ্রী'], 'wbrupashree': ['রূপশ্রী'], 'wbsaboojsathi': ['সবুজ সাথী', 'সবুজসাথী'],
    'oapwb': ['বার্ধক্য ভাতা', 'বার্ধক্য পেনশন'], 'wpwb': ['বিধবা ভাতা', 'বিধবা পেনশন'], 'jjoaps': ['জয় জোহার'],
    'obs': ['তপশিলি বন্ধু', 'তফসিলি বন্ধু', 'তপসিলি বন্ধু'], 'bay': ['বাংলার আবাস'], 'yuvasree': ['যুবশ্রী'],
    'wbsccs': ['স্টুডেন্ট ক্রেডিট কার্ড'], 'pm-kisan': ['কিষাণ', 'কিসান', 'PM-KISAN'], 'pmjdy': ['জন ধন', 'জনধন'],
    'pmuy': ['উজ্জ্বলা'], 'pmmvy': ['মাতৃ বন্দনা', 'মাতৃবন্দনা'], 'apy': ['অটল পেনশন'],
    'pmsby': ['সুরক্ষা বিমা', 'সুরক্ষা বীমা'], 'pmjjby': ['জীবন জ্যোতি'],
}
allowed = {}
for slug, c in cards.items():
    txt = ' '.join([c.get('what', ''), c.get('where', ''), c.get('note', '')] + c.get('who', []) + c.get('docs', []))
    allowed[slug] = nums(txt)
# harmless numbers that appear everywhere (helplines, years, list numbers, Aadhaar digits)
SAFE = {1930, 1100, 2024, 2025, 2026, 12, 10, 18} | set(range(0, 10))

MONEY = re.compile(r'([\d০-৯][\d০-৯,\.]*)\s*(?:লক্ষ|লাখ|হাজার)?\s*টাকা')
AGE = re.compile(r'([\d০-৯]+)\s*(?:বছর|বৎসর)')
BANGLADESH = re.compile(r'বাংলাদেশ|Bangladesh|ঢাকা|উপজেলা|ইউনিয়ন পরিষদ')
REFUSAL = re.compile(r'(আমার কাজের এলাকার বাইরে|উত্তর দিতে পারছি না|একমাত্র (কাজ|উদ্দেশ্য))')
WRONG = re.compile(r'কন্যারত্ন')  # known invented rename
POLITICS = re.compile(r'বিজেপি|তৃণমূল|সিপিএম|সিপিআই|কংগ্রেস|BJP|TMC|CPI\(M\)|Congress|ক্ষমতাসীন দল|বিরোধী দল|নির্বাচনে|রাজনৈতিক প্রভাব|দলীয় প্রভাব')

def to_int(s):
    s = s.translate(BD).replace(',', '')
    try:
        return int(float(s))
    except ValueError:
        return None

def facts_ok(text, prompt):
    named = [s for s, al in ALIAS.items() if any(a in text for a in al)]
    money = [to_int(m) for m in MONEY.findall(text)]
    ages = [to_int(m) for m in AGE.findall(text)]
    money = [m for m in money if m is not None]
    ages = [a for a in ages if a is not None]
    if not named:
        return (not money), 'unknown scheme with amounts'
    ok = set().union(*[allowed[s] for s in named]) | SAFE | nums(prompt)
    # amounts written as "৩ হাজার" or "১.২ লক্ষ" -> accept scaled forms too
    ok |= {x * 1000 for x in ok} | {x * 100000 for x in ok}
    bad = [m for m in money + ages if m not in ok]
    return (not bad), f'unverified numbers {bad[:4]}'

def words(s):
    return len(s.split())

stats, drop = Counter(), Counter()
seed, scheme, general = [], [], []
for line in open(SRC, encoding='utf-8'):
    r = json.loads(line)
    src = r.get('expand_row_source') or 'none'
    stats[src] += 1
    if src == 'none':
        seed.append({'prompt': r['prompt'], 'completion': r['completion'], 'src': 'seed_verified'})
        continue
    p = r.get('enhanced_prompt') or r['prompt']
    a = r.get('enhanced_completion') or r['completion']
    tot, mx = r.get('checklist_total_score'), r.get('checklist_max_score')
    score = (tot / mx) if tot is not None and mx else 0
    if r.get('response_safety_issues') or r.get('prompt_safety_issues'):
        drop[src, 'unsafe'] += 1; continue
    if BANGLADESH.search(a) or BANGLADESH.search(p):
        drop[src, 'bangladesh'] += 1; continue
    if POLITICS.search(a) or POLITICS.search(p):
        drop[src, 'politics'] += 1; continue
    if WRONG.search(a):
        drop[src, 'known wrong fact'] += 1; continue
    lim = MAXW if src == 'similar' else MAXW_GEN
    if words(a) > lim:
        drop[src, f'answer > {lim} words'] += 1; continue
    if src == 'similar':
        if score < 1:
            drop[src, 'checklist < 100%'] += 1; continue
        ok, why = facts_ok(a, p)
        if not ok:
            drop[src, 'unverified numbers' if 'unverified' in why else why] += 1; continue
        scheme.append({'prompt': p, 'completion': a, 'src': 'adaption_scheme'})
    else:
        if score < 0.8:
            drop[src, 'checklist < 80%'] += 1; continue
        if REFUSAL.search(a):
            r_ = {'prompt': p, 'completion': a, 'src': 'adaption_refusal'}
            general.append(r_); continue
        # general rows must not claim WB scheme facts we cannot check
        ok, why = facts_ok(a, p)
        if not ok and any(any(al in a for al in v) for v in ALIAS.values()):
            drop[src, 'unverified scheme numbers'] += 1; continue
        general.append({'prompt': p, 'completion': a, 'src': 'adaption_general'})

# keep only a few refusals (teach staying on topic without making it the main habit)
refusals = [g for g in general if g['src'] == 'adaption_refusal']
gen_ok = [g for g in general if g['src'] == 'adaption_general']
random.shuffle(refusals); random.shuffle(gen_ok)
refusals = refusals[:300]
domain_n = len(seed) + len(scheme)
cap = int(domain_n * 0.40 / 0.60)  # general = 40% of total
gen_ok = gen_ok[:max(0, cap - len(refusals))]
final = seed + scheme + refusals + gen_ok
random.shuffle(final)

with open('data/clean_train.jsonl', 'w', encoding='utf-8') as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open('data/clean_train_upload.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['prompt', 'completion'])
    for r in final:
        w.writerow([r['prompt'], r['completion']])

BN = re.compile(r'[ঀ-৿]'); LAT = re.compile(r'[A-Za-z]')
def bnr(s):
    b, l = len(BN.findall(s)), len(LAT.findall(s)); return b / (b + l) if b + l else 0
out = [f'INPUT rows: {dict(stats)}', 'DROPPED:']
out += [f'  {k[0]:17s} {k[1]:28s} {v}' for k, v in sorted(drop.items())]
kinds = Counter(r['src'] for r in final)
out += [f'KEPT: {len(final)} rows -> {dict(kinds)}']
aw = sorted(words(r['completion']) for r in final)
out += [f'answer words median {aw[len(aw)//2]}, p90 {aw[int(len(aw)*.9)]}',
        f'Bengali-script answers (>=50%): {sum(bnr(r["completion"])>=.5 for r in final)} / {len(final)}']
open('data/clean_report.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\n'.join(out))
