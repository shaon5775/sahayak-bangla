# সহায়ক বাংলা · Sahayak Bangla

**A Bengali assistant that explains government welfare schemes to rural West Bengal families — built data-first with Adaption.**

Adaption Dataset Challenge 2026 (Hackyard × Adaption Labs) · Team: **Shaon Bandyopadhyay** (solo, Kolkata, India)

| | |
|---|---|
| Language / users | Bengali (বাংলা script) and Banglish (Bengali typed in Roman letters); elderly people, women, farmers, students and daily workers in West Bengal |
| Task | Instruction tuning: a person asks about a scheme in their own words → a short, correct, safe answer in simple Bengali |
| Datasets | original seed (1,413) → Adaption-improved (19,813) → our cleaned set (6,181) → final round-2 set (5,196) — all in [`data/`](data/) |
| Adaption quality grade | seed **6.0 → 7.4** after Adaption · cleaned set **7.0 → 8.4** · final set **6.0 → 7.4** |
| AutoScientist | 2 runs, LoRA on 4B models, win rate vs base **70%** (Gemma 3 4B) and **70.4%** (Qwen3.5-4B) |
| Our own test (290 held-out questions) | yes/no eligibility **57% → 98%**, key amounts/ages **3% → 97%**, invented numbers **83% → 2.7%**, safety reminder **0.3% → 89%** (base Qwen3.5-4B → final system) |
| Demo | [`demo/`](demo/) — offline in one command (`python demo/ask.py "..."`), full model demo on Kaggle: https://www.kaggle.com/code/shaon5775/sahayak-bangla-demo (12 live conversations in [`demo/demo_transcript.md`](demo/demo_transcript.md)) |

---

## 1. The problem

West Bengal has dozens of welfare schemes — monthly support for women (Annapurna / Lakshmir Bhandar), farmer support
(Krishak Bandhu, PM-KISAN), old-age, widow and tribal pensions, school-girl grants (Kanyashree), bicycles (Sabooj Sathi),
gas connections (Ujjwala), housing and more. The people these schemes are for often miss them, because:

- the rules, documents and steps are written in official English or formal Bengali;
- many people type Bengali in Roman letters ("ami 65 bochor, bardhakya bhata pabo?");
- small problems (OTP not coming, name mismatch with Aadhaar, money not credited) stop the application;
- middlemen charge money for free forms, and OTP fraud is common.

General chat models do badly here. The base Qwen3.5-4B in our test answered a Banglish question about Ujjwala gas by
describing *Bangladeshi* gas cylinders, mistook "Sabooj Sathi" for a passport scheme, invented numbers in 83% of
answers with numbers, and almost never warned about fraud (0.3%).

**Goal:** an assistant that answers in simple spoken Bengali, gets the yes/no and the money right, never invents facts,
says "please check at the BDO office / Duare Sarkar camp" when unsure, and always protects the person from fraud.

## 2. Approach (what we did, in order)

```
official scheme pages (myScheme, Govt. of India)
   └─► 20 checked Bengali fact cards  (facts/cards_bn.json)          src/parse_myscheme.py
         └─► eligibility rules in code (src/rules.py)
               └─► SEED dataset: 1,413 rows, every answer checked by the rules   src/build_seed.py
                     └─► ADAPTION Adaptive Data: improve + expand → 19,813 rows (grade 6.0 → 7.4)
                           └─► OUR AUDIT + CLEAN-UP → 6,181 rows (grade 7.0 → 8.4)     src/clean_export.py
                                 └─► AUTOSCIENTIST round 1 (Gemma 3 4B, win rate 70%)
                                       └─► our 290-question test + hand-check of 50 → found the weak spots
                                             └─► ROUND 2 data fix → 5,196 rows (grade 6.0 → 7.4)  src/build_round2.py
                                                   └─► AUTOSCIENTIST round 2 (Qwen3.5-4B, win rate 70.4%)
                                                         └─► "rules decide, model explains" router   src/router.py
```

### 2.1 Facts first
We collected the official pages of 31 schemes from **myScheme.gov.in** (Government of India), kept 20 that matter most
in West Bengal (12 state, 8 central), and wrote one Bengali **fact card** per scheme: who can get it, how much, documents,
where to apply, common problems, helpline. Each card records the date checked (16 Nov 2026) and a caution where official
texts disagree (for example Krishak Bandhu shows both ₹5,000 and ₹10,000 on the same page). The eligibility conditions
(age, gender, caste, land, class, income tax, government job, other pension…) are written as code in `src/rules.py`.

### 2.2 Seed dataset (original)
`src/build_seed.py` turns cards + rules into **1,413 seed rows**: 763 personal eligibility stories with a yes/no verdict
decided by the rules ("আমার নাম আশা, বয়স ২৫, বাড়িতে গ্যাস নেই…"), 166 problem-solving rows (OTP, DBT failed, name mismatch,
rejected, status, scam), 340 fact questions (amount, documents, where, who — 160 of them in Banglish) and 4 honest
"I don't know" rows. Every answer uses only card facts and ends with a fraud reminder.
A **held-out test set of 290 questions** uses wordings that never appear in training.

### 2.3 Adaption — improve and expand
We uploaded the seed to **Adaption Adaptive Data** (Improve Data Quality + Expand) with a blueprint: Bengali-script answers,
no invented facts, send unknowns to BDO/SDO/Duare Sarkar, never ask for OTP or money, mention 1930 / cybercrime.gov.in,
no politics. Recipes: prompt rephrase, dedup, metadata injection, hallucination mitigation, checklist verification;
all 5 safety filters on. Expansion: +6,196 similar, +6,196 same-domain, +6,196 other-domain rows.

Result: **19,813 rows, Adaption grade 6.0 → 7.4 (C → B), percentile 7.5 → 15.8.**

### 2.4 Our audit of the Adaption output — and the clean-up
A higher grade did not mean every row was right for our users. We read the export and found:

| problem in the expanded data | rows |
|---|---|
| "similar" rows about **Bangladesh** schemes and banks (wrong country) | 606 |
| political content | 247 |
| scheme numbers that do not match the official cards | 551 |
| amounts for schemes we have no checked card for | 871 |
| our checked seed answers rewritten with unverified facts (e.g. "Kanyashree is now called Kanyaratna") | many |
| rows below 80% on Adaption's own checklist | ~57% of general rows |

`src/clean_export.py` keeps our 1,405 original verified answers unchanged, keeps Adaption scheme rows only if they pass
the full checklist, contain no Bangladesh or politics, and every ₹/age number matches a card, and keeps a capped set of
good general rows (checklist ≥ 80%, safe). Full counts: [`data/clean_report.txt`](data/clean_report.txt).

Result: **6,181 rows. Re-graded by Adaption: 7.0 → 8.4 (+20%), percentile 15.3 → 31.5.**

### 2.5 AutoScientist round 1
Instruction tuning on the cleaned set, original prompt/completion columns, **0 augmentation rows** (so no unchecked
generated rows enter training), `google/gemma-3-4b-pt`, LoRA r32/α64, 5 epochs, lr 1e-5 cosine, 1× H100.
**Win rate 70% vs 30% for the base model.** Loss fell smoothly (train ~2.0 → ~1.0, validation ~1.32 → ~1.1),
see [`charts/`](charts/).

### 2.6 Our own test showed what the win rate hid
On our 290 held-out questions (Kaggle, `kaggle/eval_template.py`) the round-1 model was far better than its base
(Bengali 64% → 100%, safety 0.3% → 75%), but a **hand-check of 50 answers** found only 10 fully correct:
wrong yes/no (it told 34 of 61 eligible people "hard to get"), wrong amounts, invented numbers in 62% of answers,
and HTML tags copied from long Adaption rows.

### 2.7 Round 2 — fix the data, not just the model
`src/build_round2.py` → **5,196 rows, 83% verified** (round 1: 23%): 2,547 eligibility stories balanced yes/no
(1,461 / 1,086), many more Bengali and Banglish wordings and spellings of scheme names, 238 problem rows; Adaption rows
cut to ≤ 250 words, HTML removed, foreign-country rows dropped. No training prompt equals a test prompt.
Adaption (dedup only): **5,193 rows, grade 6.0 → 7.4, percentile 7.2 → 11.8.**
(Adaption's grader rewards long answers; ours are short on purpose — median 72 words — so the absolute grade is lower
than round 1 even though the data is more correct.)

AutoScientist round 2: `Qwen/Qwen3.5-4B`, LoRA (r48/α96 as chosen by AutoScientist), 5 epochs, lr 1e-5 cosine,
0 augmentation. **Win rate 70.4%.**

### 2.8 "The rules decide, the model explains"
Round 2 fixed numbers and invented facts (62% → 4%) but **yes/no stayed at ~50%**: a 4B model does not reliably apply
"age between 25 and 60". So the final system routes questions:

1. `src/router.py` finds the scheme (Bengali or Banglish name) and reads the person's facts from the question
   (age, gender, caste, land, class, school type, widow, income tax, govt job, other pension, gas, pregnancy, house type).
2. If it is an eligibility question, **`src/rules.py` decides** and the answer is built from the checked fact card.
3. If a needed fact is missing, it **asks** ("আপনার বয়স কত বছর?") instead of guessing.
4. Every other question (documents, where, problems, scams…) goes to the round-2 model.

## 3. Results

### 3.1 Same 290 held-out questions, same scoring (`results/`)

| metric | base Qwen3.5-4B | round 1 (Gemma) | round 2 model | **final: round 2 + rules** |
|---|---|---|---|---|
| answer in Bengali script | 93% | 100% | 100% | **100%** |
| Bengali answer to a Banglish question | 76% | 100% | 100% | **100%** |
| yes/no eligibility correct (102 questions) | 57% | 50% | 49% | **98%** (100 right, 2 asked a follow-up, 0 wrong) |
| key amount/age from the gold answer present | 3% | 21% | 63% | **97%** |
| answers with invented numbers (lower is better) | 83% | 62% | 4% | **2.7%** |
| fraud/safety reminder (OTP, 1930, no middleman) | 0.3% | 75% | 83% | **89%** |
| chrF vs gold answer | 25.1 | 38.1 | 69.6 | 85.1\* |
| median answer length (words) | 135 | 123 | 55 | 59 |

\* chrF of the final system is flattering: rules answers are built from the same card text as the gold answers.

### 3.2 Hand-check of the same 50 answers (read one by one against the gold answer)

| | correct | partly right | wrong |
|---|---|---|---|
| round 1 model | 10 | 13 | 27 |
| round 2 model | 37 | 1 | 12 |
| **final system** | **44** | 1 | 5 |

The 5 still wrong: wrong document list (2), a documents question answered as "where to apply" (1), wrong office /
instalments for Krishak Bandhu (1), wrong advice on checking status (1) — all from the model path, none yes/no.
Sheets: `results/handcheck_50*.csv`.

### 3.3 New wording check (fairness)
The 102 test eligibility questions come from the same kind of templates as training, which favours the rules reader.
So we wrote **32 new questions** by hand in different styles (Banglish, "আমরা আদিবাসী…", "আয়কর দিই না",
"স্বামী গত বছর মারা গেছেন", 2 with a missing fact): [`data/freeform_eligibility_test.jsonl`](data/freeform_eligibility_test.jsonl).
**First try: 27 / 32 (84%).** We fixed the 5 gaps it showed and it now gets 32 / 32 — but because the fixes were made
on this set, **84% is the honest number** for unseen wording.

### 3.4 Examples (base Qwen3.5-4B vs final system; shortened)

**Q (Banglish):** `ujjwala gas te mase ba bochore koto pai?`
- Base: *"Ujjwal gas cylinders are usually 15 kg… the standard cylinder used in Bangladesh…"* ✗
- Sahayak: *"প্রধানমন্ত্রী উজ্জ্বলা যোজনায়: দরিদ্র পরিবারের মহিলাদের জন্য জমা-টাকা ছাড়া রান্নার গ্যাস সংযোগ; সঙ্গে প্রথম রিফিল ও গ্যাস ওভেন বিনামূল্যে… কেউ টাকা চাইলে দেবেন না।"* ✓

**Q:** `আমি কাকলি, নদিয়া জেলায় থাকি। বয়স ৪১ বছর। আমার কি অন্নপূর্ণা যোজনায় আবেদন করা উচিত?`
- Round 2 model alone: *"এখন পাওয়া কঠিন, কারণ: বয়স 41 — শর্ত ২৫ থেকে ৬০ বছর"* ✗ (41 *is* inside 25–60)
- Final: *"প্রধান শর্তগুলো মিলছে: আপনি মহিলা • বয়স 41 — ২৫ থেকে ৬০-এর মধ্যে… প্রতি মাসে ৩,০০০ টাকা… কাগজপত্র: আধার, ভোটার কার্ড…"* ✓

**Q:** `sabooj sathi cycle korte kon kon document niye jabo?`
- Base: talks about a "Green Card / passport" ✗
- Sahayak: *"সবুজ সাথীতে আবেদন স্কুলের মাধ্যমেই… প্রধান শিক্ষকের সঙ্গে কথা বলুন।"* (right process, but it does not list
  documents — counted as wrong in the hand-check)

**Q:** `আমি কি বিধবা ভাতা পাব?` → Final: asks for the missing facts instead of guessing.

More: [`demo/examples.md`](demo/examples.md).

## 4. Data sources and licence

- **Facts:** public scheme pages on myScheme.gov.in (Government of India), read 16 Nov 2026 (`facts/myscheme_raw.json`).
  We use them as factual reference; we do not claim any official status.
- **Seed rows:** written by us from those facts with our own templates (no third-party text datasets, no personal data;
  all names and people in questions are invented).
- **Expanded rows:** generated by Adaption Adaptive Data from our seed, then filtered by us.
- **Licence:** dataset files in `data/` and `facts/cards_bn.json` — **CC BY 4.0**; code in `src/`, `kaggle/`, `demo/` —
  **MIT**. See [`LICENSE`](LICENSE). Dataset card: [`DATASET_CARD.md`](DATASET_CARD.md).

## 5. Limitations (please read)

- **Not official advice.** Scheme rules change (West Bengal schemes changed after the 2026 election; Annapurna Yojana is
  replacing Lakshmir Bhandar). Facts are dated 16 Nov 2026. Every answer tells the person to confirm at the office.
- **The 4B model alone is not safe for yes/no decisions** (49% on our test). Use it with the rules router.
- The rules reader understands common phrasings only; on new wording it was 84% before fixes. When unsure it should
  ask — but a wrong parse can still give a wrong answer.
- 5 of 50 hand-checked answers still have a wrong detail (documents, office). Document lists need more training rows.
- In the 12-question live demo, 2 model answers were weak (a name-mismatch question answered mostly with fraud advice; a wrong description of Rupashree). Comparison questions between two schemes are not in the training data yet.
- A rules answer lists conditions the person did not mention (for example income) as met, then says the office will check them. It should say "please confirm" instead.
- Only 20 schemes are covered. Questions about other schemes go to the model, which may know less.
- Our test set was made by us, with templates related to training; numbers on real user questions will be lower.
- Adaption's quality grade measures general quality; on our data it preferred long answers, while our users need short
  ones. We used it as one signal, not the target.
- The general-purpose rows from Adaption (other domains) were only filtered automatically.

## 6. Reproduce

| step | command / file |
|---|---|
| build fact cards → seed + test | `python src/build_seed.py` |
| clean the Adaption export | `python src/clean_export.py` (needs the Adaption export; cleaned output is in `data/`) |
| build round-2 data | `python src/build_round2.py` |
| evaluate base vs trained on Kaggle (T4) | `kaggle/eval_r2_template.py` (needs your own Adaption key as a Kaggle secret) |
| score the final system | `python src/eval_hybrid.py` |
| try the assistant | see [`demo/`](demo/) |

AutoScientist details: round 1 run `23e8478a` (dataset `16ed539a`), round 2 run `24d84d4a` (dataset `015e308c`);
both Instruction Tuning (SFT), Tiny (< 10B), original columns, 0 augmentation rows, 5 epochs, lr 1e-5 cosine.

## 7. Repository map

```
data/   01_seed_original.jsonl · 02_adaption_improved_raw/part*.jsonl.gz · 03_cleaned_round1.jsonl.gz
        04_final_round2.jsonl · test_heldout.jsonl · freeform_eligibility_test.jsonl · *_report.txt
facts/  cards_bn.json (20 Bengali fact cards) · schemes_en.json · myscheme_raw.json
src/    rules.py · build_seed.py · clean_export.py · build_round2.py · router.py · eval_hybrid.py · make_release.py
kaggle/ eval_template.py (round 1) · eval_r2_template.py (round 2)
results/ eval summaries, all 290 answers of every model, hand-check sheets
charts/ AutoScientist win rate + loss curves
demo/   ask.py (offline) · examples.md · Kaggle notebook link
```

## Team
**Shaon Bandyopadhyay** — Kolkata, India (solo). Built with Adaption Adaptive Data + AutoScientist, Kaggle for evaluation.
