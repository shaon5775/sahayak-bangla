"""Score the combined system on the same 290 held-out questions: router (rules) for eligibility questions,
round-2 model answers (results/eval_outputs_r2.jsonl) for everything else. Same scoring as the Kaggle eval.
python src/eval_hybrid.py -> results/eval_hybrid_summary.json"""
import collections, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
from router import route
import sacrebleu

rows = [json.loads(l) for l in open(os.path.join(ROOT, 'results', 'eval_outputs_r2.jsonl'), encoding='utf-8')]
r1 = [json.loads(l) for l in open(os.path.join(ROOT, 'results', 'eval_outputs.jsonl'), encoding='utf-8')]
paths = collections.Counter()
hyb = []
for r in rows:
    out = route(r['prompt'], model_answer=r['ours_r2'])
    paths[out['path']] += 1
    hyb.append(out['answer'])

BN = re.compile(r'[ঀ-৿]'); LAT = re.compile(r'[A-Za-z]')
D = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
def bn_ratio(s):
    b, l = len(BN.findall(s)), len(LAT.findall(s)); return b / (b + l) if b + l else 0.0
def nums(s):
    return {int(x.replace(',', '')) for x in re.findall(r'\d[\d,]*', s.translate(D)) if x.replace(',', '').isdigit()}
NO = re.compile(r'কঠিন|মিলছে না|মেলে না|পাবেন না|পাওয়া যাবে না|পাওয়া যায় না|যোগ্য নন|যোগ্য নয়|সম্ভব নয়|^না[,।\s]')
YES = re.compile(r'মিলছে|যোগ্য|পাবেন|পেতে পারেন|হ্যাঁ|আবেদন করতে পারেন')
def verdict(a):
    h = a[:400]
    return 'no' if NO.search(h) else 'yes' if YES.search(h) else 'unclear'
SAFE = re.compile(r'OTP|ওটিপি|১৯৩০|1930|টাকা দেবেন না|টাকা চাইলে|দালাল')
def score(outs):
    s = collections.defaultdict(list)
    for r, a in zip(rows, outs):
        gold = r['gold']
        s['bengali_answer'].append(bn_ratio(a) >= 0.6)
        if 'banglish' in r['kind']: s['bengali_on_banglish_q'].append(bn_ratio(a) >= 0.6)
        if r.get('verdict') in ('yes', 'no'): s['verdict_acc'].append(verdict(a) == r['verdict'])
        gn = {n for n in nums(gold) if n >= 100}
        if gn: s['key_number_recall'].append(len(gn & nums(a)) / len(gn))
        an = {n for n in nums(a) if n >= 100 and n != 1930}
        allowed = nums(gold) | nums(r['prompt']) | {2024, 2025, 2026}
        if an: s['unsupported_number_rate'].append(len(an - allowed) / len(an))
        s['safety_reminder'].append(bool(SAFE.search(a)))
        s['words'].append(len(a.split()))
    out = {k: round(sum(v) / len(v), 3) for k, v in s.items() if k != 'words'}
    out['median_words'] = sorted(s['words'])[len(s['words']) // 2]
    out['chrF'] = round(sacrebleu.corpus_chrf(outs, [[r['gold'] for r in rows]]).score, 1)
    return out

res = {'paths': dict(paths), 'base_qwen': score([r['base_qwen'] for r in rows]), 'r1_gemma': score([r['ours'] for r in r1]),
       'r2_model': score([r['ours_r2'] for r in rows]), 'r2_plus_rules': score(hyb)}
json.dump(res, open(os.path.join(ROOT, 'results', 'eval_hybrid_summary.json'), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(ROOT, 'results', 'eval_outputs_hybrid.jsonl'), 'w', encoding='utf-8') as f:
    for r, a in zip(rows, hyb):
        f.write(json.dumps(dict(r, hybrid=a), ensure_ascii=False) + '\n')
print('paths', dict(paths))
keys = list(res['r2_plus_rules'])
print(f"{'metric':26s}{'base_qwen':>10s}{'R1':>8s}{'R2':>8s}{'R2+rules':>10s}")
for k in keys:
    print(f"{k:26s}{str(res['base_qwen'].get(k)):>10s}{str(res['r1_gemma'].get(k)):>8s}{str(res['r2_model'].get(k)):>8s}{str(res['r2_plus_rules'][k]):>10s}")
