# Step 08 — Retrieve-then-Verify (LLM verifier) 🔄 IN PROGRESS — paused, not dead-ended

CLAUDE.md Milestone 3b: use e5_k1 similarity only to shortlist candidates, then an
LLM decides same/related/new. Motivated by Step 07's finding that a fixed cosine
threshold cannot separate rare true repeats from the flood of similar-sounding
unrelated topics at gold-segmentation granularity (best-F1 streaming precision ~8%).

## Retrieval side: shortlist recall@K (done, CPU-only) ✅

Built an oracle-growth journal (new entry created on each topic's true first
occurrence, independent of any predicted decision) and, for all 36 gold repeat
events, ranked existing journal entries by e5_k1 cosine similarity.

| K | recall |
|---|---|
| 1 | 0.694 | 2 | 0.778 | 3 | 0.917 | 5 | 0.944 | **10** | **1.000** |

All 36/36 gold repeats retrievable in top 10 — retrieval was never the bottleneck.
Script: `code/shortlist_recall.py` → `outputs/shortlist_recall.json`

## Embedding/lexical-fix side-investigation (done) ✅ — ceiling found, LLM is the right tool

Before building the verifier, dug into *why* recall@1 is only 69% (11/36 events have
a closer wrong candidate). Root cause, verified against real text
(`outputs/miss_inspection.txt`, `code/inspect_misses.py`): this corpus's recurring
"year-end budget surplus reallocation" marathon sessions produce dozens of near-
identical bureaucratic templates across *different* ministries — e.g. event 7:
query and its true match (both Ministry of Education) score cosine sim **0.23**,
while a Ministry of *Health* distractor scores **0.66**, because shared
administrative register (`עודפים`, `בהרשאה להתחייב`, ...) outweighs the sparse
ministry-specific signal when mean-pooling the whole document.

Three fixes tried, in order, each ruled out with evidence rather than assumption:
1. **Regex-strip procedural boilerplate before embedding** (`code/strip_procedural.py`) —
   even in a document known to be built almost entirely from this template, stripping
   only removed 2.5% of words (66/2603) — the rest is generic committee back-and-forth
   that isn't boilerplate but also isn't ministry-identifying. Not pursued further
   (re-embedding never run).
2. **TF-IDF rerank of the top-10 shortlist** (`code/rerank_recall.py`) — best case
   +1/36 (recall@1 69.4%→72.2% at beta=0.5), *worse* than baseline past beta=1.
   Fails because these budget items share so much administrative vocabulary that
   TF-IDF also can't cleanly separate "same specific matter" from "same template,
   different ministry."
3. **Ministry/authority-name dictionary gate** (`code/ministry_gate_recall.py`) —
   same ceiling, +1/36 best case. Per-event diagnosis found: `משרד האוצר` (Finance)
   is a near-universal false-positive confound (mentioned in 153/523 segments
   regardless of topic, since this is the Finance Committee); and one case (event 32)
   is fundamentally unresolvable by ministry alone — true match and top distractor
   are **both exclusively "Ministry of Justice,"** same ministry, different specific
   matter, which needs actual reading comprehension to disambiguate.

**Conclusion**: embedding/lexical tricks have a real, evidenced ceiling around +1/36
(≈72% recall@1) on this corpus's hardest genre. This directly motivated committing
to the LLM verifier rather than continuing to chase embedding fixes.

## LLM verifier: 6 rounds of iteration, precision is now the clear, isolated bottleneck

`code/build_verifier_inputs.py` builds 522 queries (one per streaming decision, K=10
candidates each) with content-only snippets — **deliberately excludes the annotator's
topic label** from the prompt, since showing that string would let the model win by
exact-string-matching the answer rather than judging topical identity from content
(see docstring for the full reasoning). `outputs/verifier_queries.json`.

`code/run_verifier.py` is model-agnostic (any HF chat model);
`code/eval_verifier.py` scores against gold, comparable to the **e5_k1 threshold
baseline: P=0.077 R=0.333 F1=0.125**. Compared 3 models throughout:
`dicta-il/dictalm2.0-instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`.

Six real, distinct bugs found and fixed in sequence (each confirmed via job logs,
not assumed) — history kept here because the debugging path is as informative as
the result:

1. **DictaLM crash**: its Mistral-based chat template rejects a separate `system`
   role (`jinja2.exceptions.TemplateError: Conversation roles must alternate...`).
   Fixed by merging system+user into one message (works across all templates).
2. **Document-continuation**: models "answered" by literally continuing the real
   (likely memorized, since these are public transcripts) Knesset text verbatim past
   the snippet cutoff, under greedy decoding — ~95% JSON-parse failure. Fixed with
   explicit `<segment>` tags + "do not continue this text" instruction, and (at the
   time) `repetition_penalty`/`no_repeat_ngram_size`.
3. **Wrong JSON schema**: once continuation was fixed, DictaLM still invented its own
   schema (`{"decision": {"matchRank": 1}, ...}`) instead of the requested keys.
   Diagnosed as needing a worked example to anchor the exact format.
4. **Few-shot regression**: adding two worked examples *improved* schema-following
   intent but broke output syntax for all 3 models (stray `=`, curly quotes, extra
   braces). Root cause: `no_repeat_ngram_size=4` bans any 4-gram recurring anywhere in
   prompt+generation — since the few-shot examples contain the literal JSON output
   text, the model was banned from producing the very schema tokens it needed to
   legitimately repeat. Fixed by removing `no_repeat_ngram_size`, keeping a lighter
   `repetition_penalty=1.15` (the tag/instruction fix from #2 already handled the
   original continuation problem on its own).
5. **Binary decision was miscalibrated**, differently per model: dictalm2 said "same"
   almost every time (R=86% but FMR=82%), qwen3b never said "same" at all (R=0%),
   qwen7b in between but still poor (F1=0.014). Not a bug — a real calibration finding.
6. **Switched to a numeric confidence score** instead of a fixed decision
   (`{"best_matched_rank", "same_confidence": 0-100}`), letting us sweep our own
   threshold exactly like Step 07's cosine curve instead of trusting one fixed,
   uncalibrated operating point. Current best results:

| model | best F1 | precision | recall | theta |
|---|---|---|---|---|
| dictalm2 | 0.086 | 4.5% | 92.0% | 0 |
| **qwen3b** | **0.111** | 6.0% | 70.4% | 65 |
| qwen7b | 0.077 | 4.0% | 95.0% | 15 |
| **baseline (e5_k1 threshold)** | **0.125** | 7.7% | 33.3% | — |

No model beats the baseline yet, but qwen3b is within 0.014 F1 — closest yet — and
there's a clean, consistent pattern worth carrying forward: **recall is now
excellent across all three models (70-95%, vs. baseline's 33%)** — they genuinely
find almost every true recurrence via the two worked examples anchoring what "same"
should mean. The isolated remaining problem is precision: too many false positives
ride along. For dictalm2/qwen7b the best-F1 threshold sits at the sweep's edge
(theta=0 or 15), meaning their confidence score barely discriminates real matches
from fake ones; qwen3b's interior optimum (theta=65) suggests its score carries
real, if insufficient, signal.

## Ensemble across the 3 models (done, CPU-only, no new LLM calls) ✅ — tried, doesn't beat baseline

`code/ensemble_verifier.py`: majority-vote (>=2/3 agree on `best_matched_rank`) with
confidence = mean of the agreeing models' scores; no-majority queries default to
no-link (consistent with the asymmetric-cost rule — genuine model disagreement is
treated as insufficient evidence). Pure post-hoc combination of the already-computed
predictions, no GPU needed.

Majority reached on 375/522 (71.8%) queries.

| model | best F1 | precision | recall | theta |
|---|---|---|---|---|
| dictalm2 | 0.086 | 4.5% | 92.0% | 0 |
| **ensemble** | **0.109** | 5.9% | 73.3% | 0 |
| qwen3b | 0.111 | 6.0% | 70.4% | 65 |
| qwen7b | 0.077 | 4.0% | 95.0% | 15 |
| baseline (e5_k1 threshold) | 0.125 | 7.7% | 33.3% | — |

Ensemble lands essentially on top of qwen3b alone (0.109 vs 0.111) — majority voting
successfully filters out dictalm2/qwen7b's excess false positives (precision rises to
5.9% vs. their 4.0–4.5%), but it doesn't exceed qwen3b's own ceiling, since it can never
do better than the best individual model once qwen3b is already one of the three votes
and the other two are lower quality. **Still doesn't beat the e5_k1 threshold baseline.**
Script: `code/ensemble_verifier.py` → `outputs/verifier_predictions_ensemble.json`.

## Outputs
- `outputs/shortlist_recall.json`, `outputs/miss_inspection.txt`
- `outputs/verifier_queries.json`
- `outputs/verifier_predictions_{dictalm2,qwen7b,qwen3b,ensemble}.json`
- `outputs/verifier_eval_summary.json`

## Three precision-calibration experiments — implemented, queued, not yet run 🔄

`code/run_verifier.py` now supports all three remaining ideas as opt-in flags
(default behavior unchanged, so the original baseline run stays reproducible):

- `--snippet_words N` — re-derives candidate/query text from full gold span text at N
  words instead of the pre-baked 100, without needing to rebuild `verifier_queries.json`
  (retrieval/gold fields are snippet-length-independent, only the prompt text changes).
- `--fewshot extended` — 4 worked examples instead of 2: keeps the original 2, adds a
  fresh positive (recurs a month later, different phrasing) and a harder negative than
  the original (same *presenter* and same *funding mechanism*, not just same ministry).
- `--two_score` — separates `topical_similarity` (domain/ministry overlap) from
  `specific_match_confidence` (same exact matter); `eval_verifier.py` sweeps whichever
  key is present, so it's a drop-in for the existing eval pipeline.

Three sbatch scripts submit all 3 models per experiment, each writing distinctly-named
predictions (`verifier_predictions_{model}_{experiment}.json`) so they don't clobber the
baseline or each other, then call `eval_verifier.py` (which re-evaluates everything found
in `outputs/` — safe to call after each job, later jobs just add more rows to the summary):

```bash
sbatch sbatch/08b_verifier_longsnippet.sh   # ~9h budget (long prompts, batch_size 1-2)
sbatch sbatch/08c_verifier_fewshot.sh       # ~5h budget
sbatch sbatch/08d_verifier_twoscore.sh      # ~5h budget
```

**Submitted** (2026-07-15): job 31085768 (longsnippet), 31085769 (fewshot),
31085770 (twoscore) — all on the normal partition (gg:g0/a5000), pending scheduling.
Check with `squeue -j 31085768,31085769,31085770`. Once done, `eval_verifier.py`'s
summary will have 9 new model×experiment rows to compare against the existing
baseline table above.

Also still open: whether "related" predictions (currently folded into "not-linked" for
scoring, since gold has no 3-way label) are usable as-is for Task 3 (log writing).
