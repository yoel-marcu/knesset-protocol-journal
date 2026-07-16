# Step 09 — Context-Conditioned Log Writing (Task 3) 🔄 IN PROGRESS

CLAUDE.md Task 3: "Update summarization with retrieval augmentation. Given the
existing journal entry and the new protocol segment, generate only the
incremental contribution (new developments, decisions, status changes)."

## Design decision: built on gold chains, not predicted linking

Uses `gold_chains.json` (Step 07's 22 gold-recurring topics, 36 total
incremental-update instances) rather than method (a)'s *predicted* links.
Method (a) is only 7.7% precision at best-F1 (Step 07), so its predicted chains
are mostly wrong — using them here would conflate "is the summarizer bad" with
"is the upstream linking bad." Same precedent as Step 07 isolating Task 1 from
Task 2 evaluation. (Decision from 2026-07-15 session: keep method (a) as the
linking method for now, Step 08's retrieve-then-verify is paused below baseline.)

## Pipeline
```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp

python steps/09_log_writing/code/build_journal_chains.py   # CPU, gold_chains -> journal_chains.json
# --- GPU, via sbatch/09_log_writing.sh ---
python steps/09_log_writing/code/generate_logs.py --model ... --short_name ...
python steps/09_log_writing/code/eval_faithfulness.py --judge_model ... --targets ...
```

## What each script does

- **`build_journal_chains.py`**: for each of the 22 gold-recurring topics, pulls the
  full `span_text` of every occurrence in chronological order.
- **`generate_logs.py`**: model-agnostic (any HF chat model, mirrors Step 08's
  `run_verifier.py` pattern). For each chain: generates an opening entry from the
  first occurrence's text alone, then for each subsequent occurrence generates an
  incremental update given (a) the running journal built from its *own* prior
  generations (not raw source text of earlier sessions — "faithfulness is
  chain-relative") and (b) the new occurrence's text (capped at 700 words).
- **`eval_faithfulness.py`**: LLM-judge scores every generated entry 0-100 on:
  - `faithfulness` — is every claim grounded in the session text it's based on?
  - `novelty` (updates only, not openings) — does it add real new content beyond
    the existing journal, or just restate prior entries?

  No human-written reference summaries exist for this corpus, so faithfulness/
  novelty are judged against the same two things the generator saw, not an
  external gold summary.

## Model choice

Generate with `dicta-il/dictalm2.0-instruct` (Hebrew-native, good fluency for
free-text log writing — unlike Step 08's rigid JSON-schema task where it
struggled) and `Qwen/Qwen2.5-7B-Instruct` (strongest general-purpose model
available). Judge with `Qwen/Qwen2.5-3B-Instruct` — deliberately *not* one of
the two generators, to avoid self-preference bias in the faithfulness/novelty
scores.

## Outputs
- `outputs/journal_chains.json`
- `outputs/generated_logs_{dictalm2,qwen7b}.json`
- `outputs/faithfulness_scores_{dictalm2,qwen7b}.json`
- `outputs/faithfulness_summary.json`

## Results (2026-07-15, job 31086720, 0/58 parse errors both models)

| model | mean faithfulness | mean novelty (updates only) |
|---|---|---|
| **dictalm2** | **80.7** | 18.1 |
| qwen7b | 63.4 | 10.1 |

**Faithfulness**: dictalm2 (Hebrew-native) is clearly more grounded than qwen7b — the
opposite pattern from Step 08, where dictalm2 struggled with rigid JSON-schema output but
this is free-text generation, playing to its strength.

**Novelty is the real problem, for both models.** Qualitative inspection of low/high-novelty
examples shows a consistent failure mode: rather than emitting a pure incremental diff, the
model re-explains surrounding context alongside whatever new fact it adds ("the housing-grant
matter remains as discussed, though a new detail emerged that grants are limited to elderly
immigrants specifically..."). Even the *highest*-scoring examples only reach ~50/100 for this
reason — genuinely new content is present, but diluted by restated recap. The single lowest
example (tourism ministry budget) is essentially a full re-summary of all figures, new and old
mixed together, with zero isolable new content.

**Interpretation**: this isn't a data or pipeline bug (faithfulness stayed reasonably high,
parse rate was clean) — it's a real prompting/generation limitation. The explicit instruction
"do not repeat information already in the existing journal" isn't enough on its own to prevent
recap-then-append behavior.

## Two few-shot iterations tried — both made novelty worse (2026-07-15)

**v2** (job 31086920): added a worked BAD-vs-GOOD example built from real corpus data (the
housing-crisis chain), with an explicit "no recap" rule. Result: worse on both axes for
dictalm2 (faithfulness 80.7→71.1, novelty 18.1→15.0), flat for qwen7b. Root cause diagnosed by
inspection: **13/36 (36%) of dictalm2's updates were literal verbatim copies of the worked
example's "GOOD UPDATE" text**, regardless of the actual input — the model pattern-matched the
example's exact real-corpus content instead of generalizing its style.

**v3** (job 31087030): rebuilt the worked example with fully fictional content (invented
committee matter, names, numbers) plus an explicit anti-copying instruction, hypothesizing that
using real corpus content in v2 was the specific problem. Result: **worse still** — dictalm2
faithfulness collapsed to 53.6, novelty to 5.0, and the copy rate got *worse*, not better
(33/36 = 92%). Root cause: an authoring bug — the v3 fictional example's session-text slot was
left as a literal placeholder string (`[fictional example text about the fictional
library-funding matter, with a fictional new detail]`) rather than real fictional content,
giving the model nothing to reason from in the example except the fixed answer, making it even
easier to just reproduce as a template.

**The more important finding, independent of that authoring bug**: novelty declined
monotonically with each added round of few-shot scaffolding for *both* models, including
qwen7b, which had zero copying issue at all (10.1 → 8.3 → 5.3). This is evidence the problem
isn't "the model needs more guidance" — few-shot examples for this specific free-text
incremental-diff task cause models to imitate the example's *content*, not just its *style*,
the opposite of Step 08's classification task where few-shot genuinely helped. **Adding more
examples is not the right lever for this task.**

## Final results table

| version | dictalm2 faithfulness | dictalm2 novelty | qwen7b faithfulness | qwen7b novelty |
|---|---|---|---|---|
| **original (no few-shot)** | **80.7** | **18.1** | 63.4 | **10.1** |
| v2 (real-corpus example) | 71.1 | 15.0 | 63.1 | 8.3 |
| v3 (fictional example) | 53.6 | 5.0 | **69.2** | 5.3 |

**Recommendation: use the original (zero-few-shot) prompt.** It's the best or tied-best
configuration on every metric for dictalm2, and few-shot didn't meaningfully help qwen7b either
(only faithfulness ticked up in v3, at the cost of novelty).

## Next (not started — a different lever, not more examples)
- Force a structured "what changed" bullet-list output instead of free prose — harder to blend
  a copied example into a bulleted diff than into a flowing paragraph.
- Try dictalm2 alone at scale with the original prompt (already the stronger generator on both
  metrics) rather than continuing to compare two models.
- If pursuing few-shot again despite the evidence above, the *hard* version to test is whether
  the failure is fixable with 3+ diverse fictional examples (harder to overfit to one specific
  template) rather than exactly one — not attempted here.
