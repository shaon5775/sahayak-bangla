"""Split the raw myScheme page texts into sections per scheme -> facts/schemes_en.json"""
import json, re, os
HERE = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(os.path.join(HERE, '..', 'facts', 'myscheme_raw.json'), encoding='utf-8'))
SECTIONS = ['Details', 'Benefits', 'Eligibility', 'Exclusions', 'Application Process', 'Documents Required',
            'Frequently Asked Questions', 'Sources And References']
out = {}
for slug, txt in raw.items():
    if len(txt) < 2000:
        continue
    lines = [l.strip() for l in txt.split('\n') if l.strip()]
    # the page repeats the section names once as a menu at the top; real content starts at the 2nd "Details"
    idx = [i for i, l in enumerate(lines) if l == 'Details']
    start = idx[1] if len(idx) > 1 else (idx[0] if idx else 0)
    title = lines[start - 2] if start >= 2 else slug
    tags = lines[start - 1] if start >= 1 else ''
    cur, sec = None, {}
    for l in lines[start:]:
        if l in SECTIONS:
            cur = l; sec.setdefault(cur, []); continue
        if l in ('Feedback', 'Check Eligibility', 'Apply Now') or l.startswith('Was this helpful'):
            continue
        if cur:
            sec[cur].append(l)
    out[slug] = {'title': title, 'tags': tags, **{k: '\n'.join(v) for k, v in sec.items()}}
json.dump(out, open(os.path.join(HERE, '..', 'facts', 'schemes_en.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for s, d in out.items():
    print(s, '|', d['title'][:45], '|', {k: len(v) for k, v in d.items() if k not in ('title', 'tags')})
