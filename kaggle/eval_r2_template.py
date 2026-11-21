# Sahayak Bangla — ROUND 2 held-out evaluation: base Qwen3.5-4B vs base + our round-2 AutoScientist LoRA.
# Same 290 test rows and same scoring as round 1 (kaggle/eval_template.py). Runs on Kaggle (GPU T4; fp16 fits one T4).
# Needs Kaggle Secret ADAPTION_API_KEY; internet ON (base model downloaded from Hugging Face).
import os, sys, subprocess, json, re, glob, gzip, base64, tarfile, time, collections, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
LIMIT = int(os.environ.get('EVAL_LIMIT', '0')) or None
DATASET_ID = '015e308c-adad-46d5-990e-ed52a767f6a1'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers', 'peft', 'accelerate',
                'adaption', 'zstandard', 'sacrebleu'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'])

# ---------- 1. data
TEST_B64 = "__TEST_B64__"
rows = [json.loads(l) for l in gzip.decompress(base64.b64decode(TEST_B64)).decode('utf-8').splitlines() if l.strip()]
if LIMIT: rows = rows[:LIMIT]
print('test rows', len(rows))

# ---------- 2. round-2 adapter from Adaption
CK = '/kaggle/ours_r2'
if not os.path.exists(f'{CK}/adapter_config.json'):
    from kaggle_secrets import UserSecretsClient
    os.environ['ADAPTION_API_KEY'] = UserSecretsClient().get_secret('ADAPTION_API_KEY')
    from adaption import Adaption
    import zstandard
    client = Adaption()
    items = list(getattr(client.autoscientist.list(), 'data', None) or client.autoscientist.list())
    for r in items:
        print('run', r.id, r.status, getattr(r, 'dataset_id', None), getattr(r, 'model', None), getattr(r, 'created_at', None))
    ok = [r for r in items if r.status == 'succeeded']
    mine = [r for r in ok if getattr(r, 'dataset_id', None) == DATASET_ID] or ok
    run = sorted(mine, key=lambda r: str(getattr(r, 'created_at', '')))[-1]
    print('USING run', run.id, getattr(run, 'model', None), 'best_win_rate', getattr(run, 'best_win_rate', None))
    with client.autoscientist.with_streaming_response.download(run.id) as resp:
        resp.stream_to_file('/kaggle/ckpt2.tar.zst')
    with open('/kaggle/ckpt2.tar.zst', 'rb') as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd, \
            tarfile.open(fileobj=rd, mode='r|') as t:
        t.extractall(CK)
    os.remove('/kaggle/ckpt2.tar.zst')
# adapter may sit in a sub-folder
cfgs = glob.glob(f'{CK}/**/adapter_config.json', recursive=True)
AD = os.path.dirname(cfgs[0])
acfg = json.load(open(cfgs[0]))
print('adapter dir', AD, '| files', os.listdir(AD))
print('adapter config:', {k: acfg.get(k) for k in ('base_model_name_or_path', 'r', 'lora_alpha', 'target_modules')})
BASE = acfg['base_model_name_or_path']
if 'qwen3.5-4b' in BASE.lower(): BASE = 'Qwen/Qwen3.5-4B'

# ---------- 3. base model, fp16 on GPU 0
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map='cuda:0')
except Exception as e:
    print('CausalLM load failed -> image-text class:', repr(e)[:300])
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.float16, device_map='cuda:0')
model.eval()
print('model class', type(model).__name__)
tok_base = AutoTokenizer.from_pretrained(BASE)
try:
    tok_ours = AutoTokenizer.from_pretrained(AD)
except Exception as e:
    print('no tokenizer in adapter dir -> base tokenizer', repr(e)[:200]); tok_ours = tok_base
for t in (tok_base, tok_ours):
    t.padding_side = 'left'
    if t.pad_token is None: t.pad_token = t.eos_token
print('ours chat template head:', (tok_ours.chat_template or '')[:300])

THINK = re.compile(r'<think>.*?</think>', re.S)
def fmt(tok, p):
    msgs = [{'role': 'user', 'content': p}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def _gen_batch(m, tok, batch, max_new):
    enc = tok([fmt(tok, p) for p in batch], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
    try:
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                       pad_token_id=tok.pad_token_id)
    except torch.cuda.OutOfMemoryError:
        del enc; gc.collect(); torch.cuda.empty_cache()
        if len(batch) == 1: raise
        h = len(batch) // 2
        return _gen_batch(m, tok, batch[:h], max_new) + _gen_batch(m, tok, batch[h:], max_new)
    res = [THINK.sub('', tok.decode(r, skip_special_tokens=True)).strip() for r in g[:, enc['input_ids'].shape[1]:]]
    del g, enc; torch.cuda.empty_cache()
    return res

def generate(m, tok, prompts, bs=8, max_new=400):
    out = []
    for i in range(0, len(prompts), bs):
        out += _gen_batch(m, tok, prompts[i:i + bs], max_new)
        print(f'  {min(i + bs, len(prompts))}/{len(prompts)}  {time.strftime("%H:%M:%S")}', flush=True)
    return out

prompts = [r['prompt'] for r in rows]

# ---------- 4. attach our round-2 adapter (v4: remap key names; checkpoint was saved from the multimodal
#            wrapper, e.g. '...model.language_model.layers.N...', our text model uses '...model.layers.N...')
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
model = PeftModel.from_pretrained(model, AD)
akeys = load_file(glob.glob(f'{AD}/adapter_model*.safetensors')[0])
print('file key sample:', list(akeys)[:3])
want = {n.replace('.default', '') for n, _ in model.named_parameters() if 'lora_' in n}
print('model key sample:', sorted(want)[:3])
def remap(k):
    for a, b in (('model.language_model.', 'model.'), ('language_model.model.', 'model.'),
                 ('model.model.language_model.', 'model.model.')):
        if k.replace(a, b) in want: return k.replace(a, b)
    return k
fixed = {remap(k): v for k, v in akeys.items()}
hit = sum(k in want for k in fixed)
print('keys matched after remap', hit, '/', len(fixed))
res = set_peft_model_state_dict(model, fixed)
print('unexpected after load:', len(getattr(res, 'unexpected_keys', []) or []))
n_lora = sum(1 for n, _ in model.named_parameters() if 'lora_' in n)
nz = sum(1 for n, p in model.named_parameters() if 'lora_B' in n and p.abs().sum() > 0)
print('adapter tensors in file', len(akeys), '| lora params in model', n_lora, '| non-zero lora_B', nz)
assert hit == len(fixed) and nz > 0, 'adapter did not attach'
model = model.to(torch.float16)
del akeys, fixed
print('sample ours prompt:', fmt(tok_ours, prompts[0])[-200:])
t0 = time.time(); print('== OUR round-2 model'); our_out = generate(model, tok_ours, prompts)
print('ours minutes', round((time.time() - t0) / 60, 1))

# ---------- 5. base = same model with the adapter switched off
print('sample base prompt:', fmt(tok_base, prompts[0])[-200:])
t0 = time.time(); print('== BASE Qwen3.5-4B')
with model.disable_adapter():
    base_out = generate(model, tok_base, prompts)
print('base minutes', round((time.time() - t0) / 60, 1))

# ---------- 5. scoring (identical to round 1)
import sacrebleu
BN = re.compile(r'[ঀ-৿]'); LAT = re.compile(r'[A-Za-z]')
D = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
def bn_ratio(s):
    b, l = len(BN.findall(s)), len(LAT.findall(s)); return b / (b + l) if b + l else 0.0
def nums(s):
    return {int(x.replace(',', '')) for x in re.findall(r'\d[\d,]*', s.translate(D)) if x.replace(',', '').isdigit()}
NO = re.compile(r'কঠিন|মিলছে না|মেলে না|পাবেন না|পাওয়া যাবে না|পাওয়া যায় না|যোগ্য নন|যোগ্য নয়|সম্ভব নয়|^না[,।\s]')
YES = re.compile(r'মিলছে|যোগ্য|পাবেন|পেতে পারেন|হ্যাঁ|আবেদন করতে পারেন')
def verdict(a):
    head = a[:400]
    if NO.search(head): return 'no'
    if YES.search(head): return 'yes'
    return 'unclear'
SAFE = re.compile(r'OTP|ওটিপি|১৯৩০|1930|টাকা দেবেন না|টাকা চাইলে|দালাল')
def score(outs):
    s = collections.defaultdict(list)
    for r, a in zip(rows, outs):
        gold = r['completion']
        s['bengali_answer'].append(bn_ratio(a) >= 0.6)
        if 'banglish' in r['kind']: s['bengali_on_banglish_q'].append(bn_ratio(a) >= 0.6)
        if r.get('verdict') in ('yes', 'no'): s['verdict_acc'].append(verdict(a) == r['verdict'])
        gn = {n for n in nums(gold) if n >= 100}
        if gn: s['key_number_recall'].append(len(gn & nums(a)) / len(gn))
        an = {n for n in nums(a) if n >= 100 and n not in (1930,)}
        allowed = nums(gold) | nums(r['prompt']) | {2024, 2025, 2026}
        if an: s['unsupported_number_rate'].append(len(an - allowed) / len(an))
        s['safety_reminder'].append(bool(SAFE.search(a)))
        s['words'].append(len(a.split()))
    out = {k: round(sum(v) / len(v), 3) for k, v in s.items() if k != 'words'}
    out['median_words'] = sorted(s['words'])[len(s['words']) // 2]
    out['chrF'] = round(sacrebleu.corpus_chrf(outs, [[r['completion'] for r in rows]]).score, 1)
    out['n_verdict'] = len(s['verdict_acc'])
    return out
sb, so = score(base_out), score(our_out)
print('\n==== RESULTS ROUND 2 (held-out, n=%d) ====' % len(rows))
print(f"{'metric':28s}{'base_qwen':>12s}{'ours_r2':>10s}")
for k in so: print(f"{k:28s}{str(sb[k]):>12s}{str(so[k]):>10s}")
by_kind = {}
for kind in sorted({r['kind'] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r['kind'] == kind]
    by_kind[kind] = {'n': len(idx),
        'chrF_base': round(sacrebleu.corpus_chrf([base_out[i] for i in idx], [[rows[i]['completion'] for i in idx]]).score, 1),
        'chrF_ours': round(sacrebleu.corpus_chrf([our_out[i] for i in idx], [[rows[i]['completion'] for i in idx]]).score, 1)}
print(json.dumps(by_kind, ensure_ascii=False))
os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/eval_outputs_r2.jsonl', 'w', encoding='utf-8') as f:
    for r, a, b in zip(rows, base_out, our_out):
        f.write(json.dumps({'kind': r['kind'], 'scheme': r['scheme'], 'verdict': r.get('verdict'), 'prompt': r['prompt'],
                            'gold': r['completion'], 'base_qwen': a, 'ours_r2': b}, ensure_ascii=False) + '\n')
json.dump({'base_qwen': sb, 'ours_r2': so, 'by_kind': by_kind, 'n': len(rows)},
          open('/kaggle/working/eval_summary_r2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for i in [0, 1, 2]:
    print('\nQ:', rows[i]['prompt'][:200]); print('BASE:', base_out[i][:300]); print('OURS:', our_out[i][:300])
