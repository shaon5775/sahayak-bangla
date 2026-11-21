# Sahayak Bangla — full demo on Kaggle (GPU T4, internet ON).
# Final system = rules router (eligibility yes/no, from official rules + checked fact cards) + our round-2
# AutoScientist model (Qwen3.5-4B + LoRA) for every other question.
# Adapter source, first that exists:
#   1. ADAPTER_DIR env / an attached Kaggle input that holds adapter_config.json (e.g. this notebook's own output),
#   2. download from Adaption with the Kaggle Secret ADAPTION_API_KEY (owner's account).
# The cleaned adapter (key names fixed for the plain text model) is saved to /kaggle/working/sahayak_bangla_lora,
# so anyone can reuse it from this notebook's output.
import os, sys, subprocess, json, re, glob, tarfile, time, io, zipfile, urllib.request
REPO = 'https://github.com/shaon5775/sahayak-bangla/archive/refs/heads/main.zip'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers', 'peft', 'accelerate',
                'adaption', 'zstandard'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'])

# ---------- 1. code + fact cards from GitHub
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(REPO).read())).extractall('/kaggle/repo')
SRC = glob.glob('/kaggle/repo/*/src')[0]
sys.path.insert(0, SRC)
from router import route

# ---------- 2. adapter
cands = [os.environ.get('ADAPTER_DIR', '')] + [os.path.dirname(p) for p in
         glob.glob('/kaggle/input/**/adapter_config.json', recursive=True)]
AD = next((c for c in cands if c and os.path.exists(f'{c}/adapter_config.json')), None)
if AD is None:
    CK = '/kaggle/ours_r2'
    from kaggle_secrets import UserSecretsClient
    os.environ['ADAPTION_API_KEY'] = UserSecretsClient().get_secret('ADAPTION_API_KEY')
    from adaption import Adaption
    import zstandard
    client = Adaption()
    items = list(getattr(client.autoscientist.list(), 'data', None) or client.autoscientist.list())
    ok = [r for r in items if r.status == 'succeeded' and str(r.id).startswith('24d84d4a')] or \
         [r for r in items if r.status == 'succeeded']
    run = sorted(ok, key=lambda r: str(getattr(r, 'created_at', '')))[-1]
    print('Adaption run', run.id)
    with client.autoscientist.with_streaming_response.download(run.id) as resp:
        resp.stream_to_file('/kaggle/ckpt.tar.zst')
    with open('/kaggle/ckpt.tar.zst', 'rb') as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd, \
            tarfile.open(fileobj=rd, mode='r|') as t:
        t.extractall(CK)
    AD = os.path.dirname(glob.glob(f'{CK}/**/adapter_config.json', recursive=True)[0])
print('adapter dir', AD)

# ---------- 3. model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
BASE = 'Qwen/Qwen3.5-4B'
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map='cuda:0').eval()
tok_base = AutoTokenizer.from_pretrained(BASE)
try: tok = AutoTokenizer.from_pretrained(AD)
except Exception: tok = tok_base
for t in (tok, tok_base):
    t.padding_side = 'left'
    if t.pad_token is None: t.pad_token = t.eos_token
model = PeftModel.from_pretrained(model, AD)
want = {n.replace('.default', '') for n, _ in model.named_parameters() if 'lora_' in n}
def remap(k):
    for a, b in (('model.language_model.', 'model.'), ('language_model.model.', 'model.'),
                 ('model.model.language_model.', 'model.model.')):
        if k.replace(a, b) in want: return k.replace(a, b)
    return k
fixed = {remap(k): v for k, v in load_file(glob.glob(f'{AD}/adapter_model*.safetensors')[0]).items()}
assert sum(k in want for k in fixed) == len(fixed), 'adapter keys do not match'
set_peft_model_state_dict(model, fixed)
model = model.to(torch.float16)
model.save_pretrained('/kaggle/working/sahayak_bangla_lora'); tok.save_pretrained('/kaggle/working/sahayak_bangla_lora')

THINK = re.compile(r'<think>.*?</think>', re.S)
@torch.no_grad()
def generate(q, tk, max_new=350):
    msgs = [{'role': 'user', 'content': q}]
    try: s = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: s = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tk([s], return_tensors='pt', add_special_tokens=False).to('cuda:0')
    g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05, pad_token_id=tk.pad_token_id)
    return THINK.sub('', tk.decode(g[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)).strip()

def ask(q, show_base=False):
    """The Sahayak Bangla assistant: rules decide eligibility, the model explains everything else."""
    r = route(q)
    if r['path'] == 'model':
        r['answer'] = generate(q, tok)
    if show_base:
        with model.disable_adapter():
            r['base'] = generate(q, tok_base)
    return r

# ---------- 4. demo conversations (unseen wordings; none of these strings is in the training data)
QUESTIONS = [
    'ami mohila, boyos 30. annapurna yojana pabo?',
    'আমার বয়স ৬২ বছর, অন্য কোনো পেনশন নেই। বার্ধক্য ভাতা পাব কি?',
    'আমি অন্যের জমিতে ভাগে চাষ করি, নিজের জমি নেই। কৃষক বন্ধু প্রকল্পে আমি কি যোগ্য?',
    'আমি কি বিধবা ভাতা পাব?',
    'আমি মেয়ে, ১৫ বছর, সরকারি স্কুলে ৯ ক্লাসে পড়ি, বিয়ে হয়নি। কন্যাশ্রী পাব?',
    'ujjwala gas connection er jonno ki ki kagoj lagbe?',
    'আমার মায়ের অন্নপূর্ণা যোজনার টাকা তিন মাস ধরে আসছে না। কী করব?',
    'একজন লোক ফোন করে বলছে কৃষক বন্ধুর টাকা পেতে OTP বলতে হবে। দেব?',
    'ব্যাংকের পাসবইয়ে আমার নামের বানান আধারের থেকে আলাদা, তাই টাকা আটকে আছে। কী করব?',
    'pm kisan er status kivabe dekhbo?',
    'কন্যাশ্রী আর রূপশ্রী — দুটোর মধ্যে তফাত কী?',
    'দালাল বলছে ৫০০ টাকা দিলে বাংলার আবাসে ঘর পাইয়ে দেবে। কী করব?',
]
lines = ['# Sahayak Bangla — live demo transcript (Kaggle GPU run)', '',
         f'Run: {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} · model: Qwen3.5-4B + Sahayak LoRA (AutoScientist '
         'round 2) · eligibility decided by src/rules.py', '']
for i, q in enumerate(QUESTIONS, 1):
    r = ask(q, show_base=True)
    print(f'\n=== {i}. [{r["path"]}] {q}\n--- Sahayak:\n{r["answer"]}\n--- base Qwen3.5-4B:\n{r["base"][:600]}')
    lines += [f'## {i}. {q}', '', f'**Sahayak Bangla** (`{r["path"]}`' + (f', verdict `{r["verdict"]}`' if r['verdict'] else '') + '):',
              '', '> ' + r['answer'].replace('\n', '\n> '), '', '<details><summary>base Qwen3.5-4B, same question</summary>', '',
              '> ' + r['base'].replace('\n', '\n> '), '', '</details>', '']
open('/kaggle/working/demo_transcript.md', 'w', encoding='utf-8').write('\n'.join(lines))
print('\nsaved /kaggle/working/demo_transcript.md and /kaggle/working/sahayak_bangla_lora')
print("Try your own question:  print(ask('আপনার প্রশ্ন')['answer'])")
