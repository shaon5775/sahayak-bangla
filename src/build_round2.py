"""Round 2 training set (20 Nov 2026), built after the round-1 hand-check of 50 answers.

Round-1 problems -> fixes:
  wrong money amounts / documents / offices  -> verified rows (from facts/cards_bn.json) become ~75% of the set
                                                (round 1: 23%), with many more question wordings per fact.
  eligible people told "hard to get"          -> 150 persona rows per scheme (round 1: 45).
  Banglish questions misread                  -> many more Banglish wordings + common spelling variants of names.
  HTML / markdown junk, long formal answers   -> Adaption rows: HTML and ** removed, <= 250 words, capped at ~25%.
The held-out test set (data/test_heldout.jsonl) is NOT rebuilt; any training prompt equal to a test prompt is dropped,
and none of the test-only question wordings is used.

python src/build_round2.py -> data/round2_train.jsonl, data/round2_train_upload.csv, data/round2_report.txt
"""
import csv, json, os, random, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
import build_seed as B

# ---------- extra TRAIN-ONLY wordings (deliberately different from the test wordings in B.Q_TEST / BANGLISH_Q_TEST)
EXTRA_Q = {
    'overview': ['{n} সম্পর্কে বিস্তারিত বলুন।', '{n} কী ধরনের প্রকল্প?', '{n} জিনিসটা আসলে কী?'],
    'amount': ['{n}-এর টাকার পরিমাণ কত?', '{n}-এ সরকার কত টাকা দেয়?', '{n}-এ মাসে কত টাকা দেয়?'],
    'docs': ['{n}-এর জন্য কী কী কাগজপত্র লাগে?', '{n}-এ কোন কোন নথি দরকার?', '{n}-এর ফর্ম ভরতে কোন কাগজ লাগে?'],
    'where': ['{n}-এর ফর্ম কোথায় জমা দেব?', '{n}-এ আবেদন কোথা থেকে হয়?', '{n}-এর আবেদনপত্র কোথায় জমা হয়?'],
    'who': ['{n} কাদের জন্য?', '{n}-এর নিয়মে কারা পড়ে?', 'কারা {n} পাওয়ার যোগ্য?'],
}
EXTRA_BQ = {
    'amount': ['{b} e mase koto dey?', '{b} te ki ki subidha pabo?', '{b} er taka koto kore dey?'],
    'docs': ['{b} apply korte ki ki lagbe?', '{b} er kagojpotro ki ki?', '{b} er jonno documents ki?'],
    'where': ['{b} er apply kothay hoy?', '{b} er office kothay?', '{b} er form kothay pai?'],
    'who': ['{b} ke pay?', '{b} er niyom ki, ke joggo?', '{b} kader jonno?'],
    'overview': ['{b} ki jinis?', '{b} somporke bolo'],
}
ALT_NAMES = {   # common other spellings people type
    'lbs-wb': ['lakshmir bhandar'], 'kbs': ['krishak bandhu'], 'wbkanyashree': ['kanyasree'], 'wbrupashree': ['rupasree'],
    'wbsaboojsathi': ['sabuj sathi'], 'oapwb': ['bardhakya bhata'], 'wpwb': ['widow pension'], 'obs': ['taposili bandhu'],
    'bay': ['bangla awas yojana'], 'yuvasree': ['yubashree'], 'pm-kisan': ['pm kishan'], 'pmjdy': ['jandhan'],
    'pmuy': ['ujjwala yojana'], 'pmmvy': ['matru vandana'], 'apy': ['atal pension yojana'], 'aywb': ['annapurna bhandar'],
    'pmsby': ['suraksha bima yojana'], 'pmjjby': ['jivan jyoti bima'], 'jjoaps': ['joy johar'], 'wbsccs': ['student credit']}

for k, v in EXTRA_Q.items():
    B.Q[k] = B.Q[k] + v
B.BANGLISH_Q = {k: B.BANGLISH_Q.get(k, []) + EXTRA_BQ[k] for k in EXTRA_BQ}

test = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'test_heldout.jsonl'), encoding='utf-8')]
test_prompts = {x['prompt'] for x in test}

# ---------- 1. verified rows
tmp = os.path.join(ROOT, 'data', '_r2_seed_tmp.jsonl')
seed = B.build(seed=21, n_persona_per_scheme=150, n_problem=600, out_path=tmp)
os.remove(tmp)
# Banglish rows for the alternative spellings (same verified answers)
r = random.Random(5)
names = [c['name'] for c in B.CARDS.values()]
FN = {'overview': B.ans_overview, 'amount': B.ans_amount, 'docs': B.ans_docs, 'where': B.ans_where, 'who': B.ans_who}
for slug, alts in ALT_NAMES.items():
    c = B.CARDS[slug]
    for b in alts:
        for kind, tmpl in B.BANGLISH_Q.items():
            for t in tmpl:
                seed.append(dict(prompt=t.format(b=b), completion=B.fix_gen(FN[kind](c, r), names),
                                 kind=kind + '_banglish', scheme=slug))
verified, seen = [], set()
for x in seed:
    if x['prompt'] in seen or x['prompt'] in test_prompts:
        continue
    seen.add(x['prompt'])
    verified.append({'prompt': x['prompt'], 'completion': x['completion'], 'src': 'seed_verified',
                     'kind': x['kind'], 'verdict': x.get('verdict')})

# ---------- 2. Adaption rows from the round-1 clean set (already filtered), cleaned harder
TAG = re.compile(r'</?[a-zA-Z][^>]*>')
BOLD = re.compile(r'\*\*|__')
FOREIGN = re.compile(r'প্রেটোরিয়া|দক্ষিণ আফ্রিকা|আমেরিকা|ইউরোপ|চীন|জাপান|তুলনা')
def tidy(s):
    s = BOLD.sub('', TAG.sub('', s))
    s = re.sub(r'^\s*\*\s+', '• ', s, flags=re.M)        # markdown bullets -> same bullet as verified rows
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

r1 = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'clean_train.jsonl'), encoding='utf-8')]
drop = Counter()
adapt = []
for x in r1:
    if x['src'] == 'seed_verified':
        continue
    a, p = tidy(x['completion']), tidy(x['prompt'])
    if len(a.split()) > 250:
        drop[x['src'], 'answer > 250 words'] += 1; continue
    if FOREIGN.search(p) or FOREIGN.search(a):
        drop[x['src'], 'foreign comparison'] += 1; continue
    if p in test_prompts:
        drop[x['src'], 'equals a test prompt'] += 1; continue
    adapt.append({'prompt': p, 'completion': a, 'src': x['src']})

random.seed(9)
random.shuffle(adapt)
cap = int(len(verified) * 0.25 / 0.75)                    # Adaption rows = ~25% of the total
refus = [x for x in adapt if x['src'] == 'adaption_refusal'][:80]
rest = [x for x in adapt if x['src'] != 'adaption_refusal'][:max(0, cap - len(refus))]
final = verified + refus + rest
random.shuffle(final)

out_j = os.path.join(ROOT, 'data', 'round2_train.jsonl')
with open(out_j, 'w', encoding='utf-8') as f:
    for x in final:
        f.write(json.dumps(x, ensure_ascii=False) + '\n')
with open(os.path.join(ROOT, 'data', 'round2_train_upload.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['prompt', 'completion'])
    for x in final:
        w.writerow([x['prompt'], x['completion']])

ver = [x for x in final if x['src'] == 'seed_verified']
rep = [f'ROUND 2 rows: {len(final)}  by source: {dict(Counter(x["src"] for x in final))}',
       f'verified share: {len(ver) / len(final):.0%}',
       'verified kinds: ' + str(dict(Counter(x['kind'] for x in ver))),
       'persona verdicts: ' + str(dict(Counter(x['verdict'] for x in ver if x['kind'] == 'persona'))),
       'Adaption rows dropped: ' + str({f'{a} / {b}': n for (a, b), n in drop.items()}),
       'HTML tags left: ' + str(sum(bool(TAG.search(x['completion'])) for x in final)),
       'training prompts equal to a test prompt: ' + str(sum(x['prompt'] in test_prompts for x in final)),
       'answer words median: ' + str(sorted(len(x['completion'].split()) for x in final)[len(final) // 2])]
open(os.path.join(ROOT, 'data', 'round2_report.txt'), 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('\n'.join(rep))
