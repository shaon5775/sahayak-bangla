# Dataset card — Sahayak Bangla (সহায়ক বাংলা)

**Bengali instruction-tuning data for explaining West Bengal and central government welfare schemes to rural users.**

| | |
|---|---|
| Language | Bengali (bn, Bengali script) answers; questions in Bengali script and in Banglish (Bengali in Roman letters) |
| Domain | Government welfare schemes in West Bengal, India (20 schemes: 12 state, 8 central) |
| Task | Instruction tuning / supervised fine-tuning (prompt → completion) |
| Final size | 5,196 rows (`data/04_final_round2.jsonl`); 290-row held-out test + 32-row free-form test |
| Built with | our own fact cards and rules + **Adaption Adaptive Data** (improve, expand, dedup, quality grade) |
| Licence | CC BY 4.0 (data), MIT (code) |
| Author | Shaon Bandyopadhyay, Kolkata, India — Adaption Dataset Challenge 2026 |
| Facts checked | 16 Nov 2026, from myScheme.gov.in (Government of India) |

## Files

| file | rows | what it is |
|---|---|---|
| `data/01_seed_original.jsonl` | 1,413 | **Original** seed written by us from the fact cards; every eligibility verdict decided by `src/rules.py` |
| `data/02_adaption_improved_raw/part1–7.jsonl.gz` | 19,813 | **Adaption-improved and expanded** dataset exactly as exported (prompt, completion, enhanced_prompt, enhanced_completion, expand_row_source, checklist scores, safety flags; Adaption's long reasoning fields removed for size) |
| `data/03_cleaned_round1.jsonl.gz` | 6,181 | Our audit + clean-up of the Adaption export (`src/clean_export.py`); used for AutoScientist round 1 |
| `data/04_final_round2.jsonl` | 5,196 | **Final dataset** (`src/build_round2.py`); used for AutoScientist round 2 |
| `data/test_heldout.jsonl` | 290 | Held-out test questions with gold answers; wordings never used in training |
| `data/freeform_eligibility_test.jsonl` | 32 | Hand-written eligibility questions in new styles, with gold yes / no / ask |
| `facts/cards_bn.json` | 20 cards | The checked Bengali facts every answer is built from |

## Fields (final dataset)

| field | meaning |
|---|---|
| `prompt` | the user's question (Bengali or Banglish) |
| `completion` | the assistant's answer in simple Bengali |
| `src` | `seed_verified` (built from cards + rules, 4,289), `adaption_scheme` (159), `adaption_general` (668), `adaption_refusal` (80) |
| `kind` | `persona` (personal eligibility story), `problem`, `amount`, `docs`, `where`, `who`, `overview` (+ `_banglish`), `unknown`; empty for Adaption rows |
| `verdict` | `yes` / `no` for persona rows, decided by the rules |

## Composition (final dataset)

- 2,547 personal eligibility stories — 1,461 "yes" / 1,086 "no" — each listing which conditions match and which do not.
- 238 problem-solving rows: OTP not coming, money not credited (DBT failed), name mismatch with Aadhaar, rejected
  application, checking status, fraud and middlemen.
- 1,500 fact questions (amount, documents, where to apply, who can get it, overview), 880 of them in Banglish, with
  several spellings of scheme names.
- 907 Adaption rows kept after filtering (scheme rows whose numbers match the cards, safe general rows, on-topic
  refusals), each ≤ 250 words, HTML removed.
- 4 honest "I don't know — ask at the office" rows.
- 83% of rows are verified against the fact cards; 97% of answers are in Bengali script; median answer 72 words.

## How it was made

1. Official scheme pages (myScheme.gov.in) → 20 Bengali fact cards with sources and dates → eligibility rules in code.
2. Templates × invented personas × cards → 1,413 seed rows (no real people, no personal data).
3. Adaption Adaptive Data: blueprint (simple Bengali, no invented facts, unknown → office, never OTP/money, 1930
   helpline, no politics), recipes (rephrase, dedup, metadata injection, hallucination mitigation, checklist
   verification), 5 safety filters, expansion to 19,813 rows. **Adaption grade 6.0 → 7.4 (C → B).**
4. Our audit found Bangladesh content (606 rows), politics (247), numbers that disagree with official facts (551),
   amounts for schemes without a checked card (871), and rewrites of our verified answers. Clean-up kept our original
   answers and only checked Adaption rows → 6,181 rows. **Adaption grade 7.0 → 8.4, percentile 15.3 → 31.5.**
5. After training and testing round 1, we rebuilt the data to fix the errors we saw (wrong yes/no, wrong amounts,
   Banglish misreads, HTML) → 5,196 rows, 83% verified. **Adaption grade 6.0 → 7.4, percentile 7.2 → 11.8.**

## Quality checks

- 0 training prompts are identical to a test prompt; test-only wordings never appear in training.
- Every ₹ amount and age in seed rows comes from a card; Adaption scheme rows were dropped if any number disagreed.
- Model trained on it (Qwen3.5-4B, AutoScientist): win rate 70.4% vs base; with the rules router, yes/no accuracy 98%,
  key-number recall 97%, invented numbers 2.7% on the 290 held-out questions (see README).
- Hand-check of 50 answers of the final system: 44 correct, 1 partly, 5 wrong.

## Intended use
Training and evaluating small assistants that explain welfare schemes in Bengali, research on low-resource Bengali
and Banglish instruction data, and building fact-grounded (rules + model) helpers.

## Not intended for
Making real eligibility decisions without a human or official check, legal or financial advice, or any scheme not in
the cards. Rules change — the data is a snapshot of 16 Nov 2026.

## Known limitations and biases
- Covers 20 schemes; West Bengal rules changed after the 2026 election and amounts may be out of date.
- Questions are template-based with invented names; real users write more varied, noisier text.
- Personas are spread across West Bengal districts but skew to simple cases (one person, one scheme).
- Adaption general-domain rows were filtered automatically only; some may be generic or long.
- Bengali is standard West Bengal spoken (চলিত) Bengali; Bangladeshi usage and other dialects are not covered.

## Citation
Shaon Bandyopadhyay (2026). *Sahayak Bangla: a Bengali welfare-scheme instruction dataset built with Adaption.*
https://github.com/shaon5775/sahayak-bangla
