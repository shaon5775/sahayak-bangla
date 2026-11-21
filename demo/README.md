# Demo — try Sahayak Bangla

## 1. Offline, one command (no GPU, no internet, no install)

```bash
git clone https://github.com/shaon5775/sahayak-bangla && cd sahayak-bangla
python demo/ask.py "ami mohila, boyos 30. annapurna yojana pabo?"
python demo/ask.py "আমি কি বিধবা ভাতা পাব?"          # missing fact -> it asks
python demo/ask.py                                     # interactive
```

This runs the **rules path** of the final system: eligibility questions are decided by the official rules and
answered from the checked fact cards. Questions that need the fine-tuned model (documents, problems, scams…) show the
checked fact card for the scheme instead.

## 2. Full system with the fine-tuned model (Kaggle GPU)

Public Kaggle notebook: **KAGGLE_DEMO_URL**

It loads Qwen3.5-4B + our AutoScientist LoRA (round 2), wires it to the rules router and answers 12 new questions,
showing the base model's answer next to ours. Its output contains:
- `demo_transcript.md` — the 12 conversations (also copied here: [`demo_transcript.md`](demo_transcript.md))
- `sahayak_bangla_lora/` — the adapter, ready for `PeftModel.from_pretrained(Qwen3.5-4B, ...)`

Code: [`sahayak_demo_kaggle.py`](sahayak_demo_kaggle.py). In the notebook, call `ask('আপনার প্রশ্ন')`.

## 3. Recorded examples from the test run
[`examples.md`](examples.md) — 11 test questions with base vs round-2 vs final answers.
