"""Sahayak Bangla — offline demo (no GPU, no internet, Python 3.9+ standard library only).

    python demo/ask.py "আমার বয়স ৬২ বছর, অন্য কোনো পেনশন নেই। বার্ধক্য ভাতা পাব কি?"
    python demo/ask.py            # interactive: type questions, empty line to quit

Eligibility questions ("will I get scheme X?") are decided by the official rules (src/rules.py) and answered from
the checked fact cards — exactly the path the full system uses. If a needed fact is missing, it asks.
Other questions (documents, problems, scams, …) need the fine-tuned model; offline we show the checked fact card
for the scheme instead. The full system with the model runs in demo/sahayak_demo_kaggle.py (GPU)."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from router import route  # noqa: E402

CARDS = json.load(open(os.path.join(ROOT, 'facts', 'cards_bn.json'), encoding='utf-8'))


def card_text(slug):
    c = CARDS[slug]
    lst = lambda v: '\n'.join('• ' + x for x in v) if isinstance(v, list) else str(v)
    parts = [c['name'], '', 'কী পাবেন: ' + c.get('what', ''), '', 'কারা পাবেন:\n' + lst(c.get('who', [])),
             '', 'কাগজপত্র:\n' + lst(c.get('docs', [])), '', 'কোথায়: ' + c.get('where', '')]
    if c.get('note'): parts += ['', c['note']]
    parts += ['', 'আবেদন সম্পূর্ণ বিনামূল্যে; কেউ টাকা বা OTP চাইলে দেবেন না। সন্দেহ হলে ১৯৩০ নম্বরে ফোন করুন।',
              f"(তথ্য যাচাই: {CARDS['_meta']['checked']} — সর্বশেষ নিয়ম অফিসে মিলিয়ে নিন।)"]
    return '\n'.join(parts)


def answer(q):
    r = route(q)
    if r['path'] in ('rules', 'ask'):
        return f"[{r['path']}: {r['verdict']}]  facts read: {r['persona']}\n\n{r['answer']}"
    if r['scheme']:
        return '[model path — offline demo shows the checked fact card]\n\n' + card_text(r['scheme'])
    return ('[model path — no scheme name found; the fine-tuned model answers this in the GPU demo]\n\n'
            'এই প্রশ্নের উত্তর দিতে প্রকল্পের নামটা বলুন, অথবা BDO অফিস / দুয়ারে সরকার ক্যাম্পে জিজ্ঞেস করুন।')


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    if len(sys.argv) > 1:
        for q in sys.argv[1:]: print('প্রশ্ন:', q, '\n'); print(answer(q)); print('\n' + '-' * 60)
    else:
        while True:
            q = input('\nপ্রশ্ন (empty = quit): ').strip()
            if not q: break
            print(); print(answer(q))
