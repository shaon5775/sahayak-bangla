# Sahayak Bangla — held-out evaluation: base Gemma 3 4B PT vs base + our AutoScientist LoRA adapter.
# Runs on Kaggle (GPU T4 x2). Needs Kaggle Secret ADAPTION_API_KEY and accepted Gemma 3 licence.
import os, sys, subprocess, json, re, glob, gzip, base64, tarfile, time, collections, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   # v2: less fragmentation (v1 ran out of GPU memory)
LIMIT = int(os.environ.get('EVAL_LIMIT', '0')) or None   # 0 = all 290 rows
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'adaption', 'zstandard', 'sacrebleu'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'])  # old torchao breaks peft import

# ---------- 1. data (290 held-out rows, wordings never used in training) ----------
TEST_B64 = "__TEST_B64__"
rows = [json.loads(l) for l in gzip.decompress(base64.b64decode(TEST_B64)).decode('utf-8').splitlines() if l.strip()]
if LIMIT: rows = rows[:LIMIT]
print('test rows', len(rows))

# ---------- 2. our adapter from Adaption (downloaded inside Kaggle) ----------
CK = '/kaggle/ours_ckpt'
if not os.path.exists(f'{CK}/adapter_model.safetensors'):
    from kaggle_secrets import UserSecretsClient
    os.environ['ADAPTION_API_KEY'] = UserSecretsClient().get_secret('ADAPTION_API_KEY')
    from adaption import Adaption
    import zstandard
    client = Adaption()
    items = list(getattr(client.autoscientist.list(), 'data', None) or client.autoscientist.list())
    run = [r for r in items if r.status == 'succeeded'][0]
    print('run', run.id, 'best_win_rate', getattr(run, 'best_win_rate', None))
    with client.autoscientist.with_streaming_response.download(run.id) as resp:
        resp.stream_to_file('/kaggle/ckpt.tar.zst')
    with open('/kaggle/ckpt.tar.zst', 'rb') as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd, \
            tarfile.open(fileobj=rd, mode='r|') as t:
        t.extractall(CK)
    os.remove('/kaggle/ckpt.tar.zst')

# ---------- 3. base Gemma 3 4B PT (text part) ----------
import torch, kagglehub
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Gemma3ForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
BASE = kagglehub.model_download('google/gemma-3/transformers/gemma-3-4b-pt')
tcfg = AutoConfig.from_pretrained(BASE).text_config
model = Gemma3ForCausalLM(tcfg).to(torch.float32)
sd = {}
for shard in sorted(glob.glob(f'{BASE}/*.safetensors')):
    with safe_open(shard, 'pt') as f:
        for k in f.keys():
            nk = None
            if k.startswith('language_model.model.'): nk = 'model.' + k[len('language_model.model.'):]
            elif k.startswith('model.language_model.'): nk = 'model.' + k[len('model.language_model.'):]
            elif k.startswith('language_model.lm_head.'): nk = 'lm_head.' + k[len('language_model.lm_head.'):]
            if nk: sd[nk] = f.get_tensor(k).to(torch.float32)
res = model.load_state_dict(sd, strict=False)
del sd
model.tie_weights()
miss = [k for k in res.missing_keys if not k.startswith('lm_head')]
print('missing (non lm_head):', len(miss), miss[:5], '| unexpected:', len(res.unexpected_keys))
assert not miss, 'base weights did not load'
model.eval()
dmap = infer_auto_device_map(model, max_memory={0: '8GiB', 1: '12GiB', 'cpu': '20GiB'},
                             no_split_module_classes=['Gemma3DecoderLayer'])
model = dispatch_model(model, device_map=dmap)
tok = AutoTokenizer.from_pretrained(CK)
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = '<pad>'

def fmt(p):  # same format the adapter was trained with (Llama-2 style template in the checkpoint)
    return tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False)

@torch.no_grad()
def _gen_batch(m, batch, max_new):
    texts = [fmt(p) for p in batch]
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
    try:
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    except torch.cuda.OutOfMemoryError:            # v2: split the batch instead of crashing
        del enc; gc.collect(); torch.cuda.empty_cache()
        if len(batch) == 1: raise
        h = len(batch) // 2
        print(f'    OOM at batch {len(batch)} -> splitting', flush=True)
        return _gen_batch(m, batch[:h], max_new) + _gen_batch(m, batch[h:], max_new)
    res = [tok.decode(row, skip_special_tokens=True).strip() for row in g[:, enc['input_ids'].shape[1]:]]
    del g, enc; torch.cuda.empty_cache()
    return res

def generate(m, prompts, bs=8, max_new=400):
    out = []
    for i in range(0, len(prompts), bs):
        out += _gen_batch(m, prompts[i:i + bs], max_new)
        print(f'  {min(i + bs, len(prompts))}/{len(prompts)}  {time.strftime("%H:%M:%S")}', flush=True)
    return out

prompts = [r['prompt'] for r in rows]
t0 = time.time()
print('== BASE model'); base_out = generate(model, prompts)
print('base minutes', round((time.time() - t0) / 60, 1))
gc.collect(); torch.cuda.empty_cache()

# ---------- 4. same model + our LoRA adapter ----------
from peft import PeftModel
from safetensors.torch import load_file
model = PeftModel.from_pretrained(model, CK)
akeys = load_file(f'{CK}/adapter_model.safetensors')
n_lora_model = sum(1 for n, _ in model.named_parameters() if 'lora_' in n)
print('adapter tensors in file', len(akeys), '| lora params in model', n_lora_model)
assert n_lora_model >= len(akeys) > 0, 'adapter did not attach'
nz = sum(1 for n, p in model.named_parameters() if 'lora_B' in n and p.abs().sum() > 0)
print('non-zero lora_B tensors', nz); assert nz > 0
del akeys
t0 = time.time()
print('== OUR model'); our_out = generate(model, prompts)
print('ours minutes', round((time.time() - t0) / 60, 1))

# ---------- 5. scoring ----------
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
        if r.get('verdict') in ('yes', 'no'):
            s['verdict_acc'].append(verdict(a) == r['verdict'])
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
print('\n==== RESULTS (held-out, n=%d) ====' % len(rows))
print(f"{'metric':28s}{'base':>10s}{'ours':>10s}")
for k in so:
    print(f"{k:28s}{str(sb[k]):>10s}{str(so[k]):>10s}")
by_kind = {}
for kind in sorted({r['kind'] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r['kind'] == kind]
    by_kind[kind] = {
        'n': len(idx),
        'chrF_base': round(sacrebleu.corpus_chrf([base_out[i] for i in idx], [[rows[i]['completion'] for i in idx]]).score, 1),
        'chrF_ours': round(sacrebleu.corpus_chrf([our_out[i] for i in idx], [[rows[i]['completion'] for i in idx]]).score, 1)}
print(json.dumps(by_kind, ensure_ascii=False))
os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/eval_outputs.jsonl', 'w', encoding='utf-8') as f:
    for r, a, b in zip(rows, base_out, our_out):
        f.write(json.dumps({'kind': r['kind'], 'scheme': r['scheme'], 'verdict': r.get('verdict'), 'prompt': r['prompt'],
                            'gold': r['completion'], 'base': a, 'ours': b}, ensure_ascii=False) + '\n')
json.dump({'base': sb, 'ours': so, 'by_kind': by_kind, 'n': len(rows)},
          open('/kaggle/working/eval_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for i in [0, 1, 2]:
    print('\nQ:', rows[i]['prompt'][:200]); print('BASE:', base_out[i][:300]); print('OURS:', our_out[i][:300])
