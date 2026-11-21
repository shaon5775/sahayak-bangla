"""Build the public release folder (GitHub repo) for the Adaption Dataset Challenge submission.
python src/make_release.py -> release/sahayak-bangla/"""
import gzip, json, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(ROOT, 'release', 'sahayak-bangla')
J = lambda *p: os.path.join(ROOT, *p)

def rows(path):
    with open(path, encoding='utf-8') as f:
        for l in f:
            if l.strip(): yield json.loads(l)

def write(path, it, gz=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    op = gzip.open if gz else open
    n = 0
    with op(path, 'wt', encoding='utf-8') as f:
        for r in it:
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
    print(f'{os.path.relpath(path, OUT):55s} {n:6d} rows  {os.path.getsize(path)/1e6:6.1f} MB')

for d in ('data', 'facts', 'src', 'kaggle', 'results', 'charts', 'demo'):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

# ---- datasets (original -> Adaption-improved -> our cleaned -> final)
write(os.path.join(OUT, 'data', '01_seed_original.jsonl'),
      ({k: r[k] for k in ('prompt', 'completion', 'kind', 'scheme')} for r in rows(J('data', 'seed_train.jsonl'))))
KEEP = ('prompt', 'completion', 'enhanced_prompt', 'enhanced_completion', 'expand_row_source',
        'checklist_total_score', 'checklist_max_score', 'prompt_safety_issues', 'response_safety_issues')
raw = [{k: r.get(k) for k in KEEP} for r in rows(J('data', 'adaption_export', 'wb_gov_scheme_qa_bn.jsonl'))]
PART = 3000  # GitHub web upload limit -> several parts of < 10 MB each
for i in range(0, len(raw), PART):
    write(os.path.join(OUT, 'data', '02_adaption_improved_raw', f'part{i // PART + 1}.jsonl.gz'), raw[i:i + PART], gz=True)
write(os.path.join(OUT, 'data', '03_cleaned_round1.jsonl.gz'), rows(J('data', 'clean_train.jsonl')), gz=True)
write(os.path.join(OUT, 'data', '04_final_round2.jsonl'), rows(J('data', 'round2_train.jsonl')))
write(os.path.join(OUT, 'data', 'test_heldout.jsonl'),
      ({k: r[k] for k in ('prompt', 'completion', 'kind', 'scheme') if k in r} | ({'verdict': r['verdict']} if 'verdict' in r else {})
       for r in rows(J('data', 'test_heldout.jsonl'))))
write(os.path.join(OUT, 'data', 'freeform_eligibility_test.jsonl'), rows(J('data', 'freeform_eligibility_test.jsonl')))
for f in ('clean_report.txt', 'round2_report.txt', 'stats.txt'):
    shutil.copy(J('data', f), os.path.join(OUT, 'data', f))

# ---- facts, code, results, charts
for f in ('cards_bn.json', 'schemes_en.json', 'myscheme_raw.json', 'target_schemes.txt'):
    shutil.copy(J('facts', f), os.path.join(OUT, 'facts', f))
for f in ('parse_myscheme.py', 'rules.py', 'build_seed.py', 'clean_export.py', 'build_round2.py', 'router.py',
          'eval_hybrid.py', 'make_release.py'):
    shutil.copy(J('src', f), os.path.join(OUT, 'src', f))
for f in ('eval_template.py', 'eval_r2_template.py'):
    shutil.copy(J('kaggle', f), os.path.join(OUT, 'kaggle', f))
for f in ('eval_summary.json', 'eval_hybrid_summary.json', 'eval_outputs.jsonl', 'eval_outputs_r2.jsonl',
          'eval_outputs_hybrid.jsonl', 'handcheck_50.csv', 'handcheck_50_r2.csv', 'handcheck_ids.json'):
    shutil.copy(J('results', f), os.path.join(OUT, 'results', f))
for f in os.listdir(J('writeup', 'charts')):
    shutil.copy(J('writeup', 'charts', f), os.path.join(OUT, 'charts', f))
print('done ->', OUT)
