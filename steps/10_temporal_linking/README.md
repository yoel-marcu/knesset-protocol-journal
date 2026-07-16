# Step 10 — Temporal / Geometric Streaming Linking 🔄 IN PROGRESS

Re-frames Task 2 as **online clustering**, not nearest-neighbour classification
against a frozen reference set. Motivated by the observation that Steps 5/7/8
represent each journal entry by its **first span, frozen forever** — throwing away
everything the streaming setting accumulates. Exploits three assets the temporal
setting hands us for free:

- **A. Accumulated per-topic representation** (centroid / robust median / linkage
  variants) instead of a frozen first span.
- **B. Online de-biasing** (mean-centering, ABTT top-k removal, shrinkage whitening)
  to suppress the shared "budget boilerplate" direction that inflates false-merge
  confounds.
- **C. Temporal decision rules** (a distinctiveness *margin* gate; a DP/CRP model
  with size prior + adaptive NEW option) instead of one flat cosine threshold.
- **D. LLM matter-fingerprints** (stable identifiers: laws/programs/entities) for the
  confounds geometry cannot fix — the LLM as extractor, not classifier.
- **E. Verify against the full accumulated timeline** — the method (b) the brief
  actually specifies, which Step 8 never implemented (it showed only the first span).

All of A/B/C are **CPU-only on the frozen e5 embeddings**; D/E use the GPU.
Evaluation mirrors Step 7/8 exactly (oracle-growth journal, P/R/F1 + false-merge
rate), so numbers are directly comparable to the streaming baseline (F1 ≈ 0.125).
This harness reproduces that baseline at **0.103** (first-span + threshold; the small
gap is global-argmax here vs the top-10 retrieval restriction there).

## Headline result (Families A–C, CPU, no LLM)

**The distinctiveness margin gate is the decisive lever, and the centroid
representation is the decisive recall fix — both survive streaming honesty.**

| Setting | Config | Best F1 | FMR≤5% recall |
|---|---|---|---|
| baseline | first-span + threshold | 0.103 | 2.8% (1/36) |
| oracle, batch transform | whiten50 · median · margin | **0.250** | 8.6% |
| **oracle, streaming-honest transform** | **center · centroid · margin** | **0.190** | **8.6% (3/36)** |
| true-streaming feedback (contamination) | winner, pairwise-F1 | 0.164 | — (vs baseline 0.074) |

For reference, **every LLM verifier from Step 8 maxed at F1 = 0.111** — the geometric
method beats all of them, CPU-only.

## What each experiment showed

**Sweep 1 — entry representation** (`streaming_eval.py`): centroid/median lift
recall from **32% → 89%** at similar precision. The frozen first-span was the recall
killer, exactly as hypothesised.

**Sweep 2 — de-biasing**: whitening/ABTT beat raw modestly on best-F1 (0.092 → 0.11).

**Sweep 3 — decision rule**: the **margin gate nearly doubles F1** (0.107 → 0.200) —
switching from "is top-1 similar enough" to "is top-1 *distinctively* closer than
top-2." Boilerplate confounds sit near *many* budget entries (low margin); genuine
repeats sit near *one* (high margin). The DP/CRP rule underperformed (its log-n_k
size prior actively favours the big generic budget cluster, i.e. the confound).

**Sweep 4 — factorial**: best batch/oracle config `whiten50 · median · margin = 0.250`,
2× the baseline.

**Refinement** (`streaming_refine.py`): a 2-D floor+margin gate does not beat pure
margin; DP-without-size-prior is *worse* (0.098), confirming DP is the wrong lever;
whitening sweet spot is dim 50–80.

**Honesty check (a) — streaming-honest transform** (`streaming_honest_compare.py`,
`streaming_robustness.py`): whitening was **leaking future information** — refit only
on spans-seen-so-far it collapses 0.250 → 0.094 (a 50-dim covariance is unestimable
early in the stream). The *cheap* transforms survive: **center · centroid · margin =
0.190 honest** (actually higher than its batch 0.175), abtt1/abtt2 ≈ 0.16. So the
honest flagship drops whitening entirely.

**Honesty check (b) — true-streaming feedback** (contamination, decisions feed back):
induced-clustering pairwise-F1 **0.164 (winner) vs 0.074 (baseline)** — 2.2×. The
winner produces 478 clusters (gold: 487) — it slightly *over-splits*, the SAFE
direction under the false-merge≫false-split cost; the baseline over-merges (367
clusters) and pays for it.

## Families D & E (GPU, job 31088669, done) — the LLM helps as an *extractor*, not a *classifier*

**D — matter-fingerprints (LLM extracts stable identifiers, then fused with geometry):**
The LLM extracts `{ministry, laws_programs, entities, request_numbers, matter}` per
span; entries accumulate the union of stable IDs; linking fuses geometric margin with
identifier overlap. On the honest `center` base:

| method | F1 | FMR≤5% recall |
|---|---|---|
| geometry only (center · centroid · margin) | 0.190 | 8.6% |
| **+ dictalm2 fingerprints, additive fusion** | **0.211** | 8.6% |
| **+ dictalm2 fingerprints, hard identifier gate** | 0.184 | **11.4% (4/36)** |

Fingerprint fusion lifts F1 to **0.211** and pushes the anti-duplication (FMR≤5%)
recall to **11.4% — 4× the baseline's 2.8%**, the best precision-priority operating
point of anything tried. Notes: dictalm2's Hebrew extraction is the useful one
(qwen7b fingerprints added nothing, additive stayed at 0.201); and dictalm2 had
129/523 (25%) JSON parse errors, so the identifier signal has headroom with cleaner
extraction. Stable IDs (laws/programs/entities) are used; transient פנייה/בקשה
numbers are deliberately ignored (they change every session and would split true
budget-item recurrences).

**E — verify against the full accumulated timeline** (the method (b) the brief
specifies, which Step 8 never built): **F1 = 0.065 (qwen7b) / 0.068 (dictalm2) —
worse than Step 8's single-snippet verifier (0.111) and far below the geometry.**
Giving the LLM verifier *more* context (the whole timeline) hurt, consistent with
Step 8's finding that longer snippets hurt. The verifier-as-classifier is a dead end;
the LLM's value is purely as a per-span extractor (Family D).

## Final honest leaderboard (all directly comparable to the baseline 0.103/0.125)

| method | best F1 | FMR≤5% recall |
|---|---|---|
| baseline: first-span + threshold | 0.103 | 2.8% |
| Step 8 best LLM verifier (single snippet) | 0.111 | ~0% |
| Family E: LLM verifier, full timeline | 0.068 | 0% |
| **geometry: center · centroid · margin** | **0.190** | 8.6% |
| **geometry + dictalm2 fingerprints (additive)** | **0.211** | 8.6% |
| geometry + dictalm2 fingerprints (hard gate) | 0.184 | **11.4%** |

## Takeaways

1. **Represent topics, not spans** — the single change (centroid vs frozen first
   span) lifts recall 32%→89% and triples the anti-duplication recall. This was the
   biggest, cheapest fix and it survives every honesty check.
2. **Distinctiveness beats magnitude** — the margin gate is the decisive innovation;
   cheap, interpretable, and it survives both honesty checks.
3. **The LLM helps as an extractor, not a classifier.** Fingerprint extraction fused
   with geometry gives the best numbers (F1 0.211, FMR≤5% recall 11.4% = 4× baseline);
   every LLM *verifier/classifier* variant (Step 8 single-snippet, Family E
   full-timeline) stays ≤ 0.11.
4. **Whitening leaks** — batch-fit de-anisotropization looks best (0.250) but uses
   future spans; refit honestly it collapses to 0.094, so the honest flagship uses
   only running-mean centering.
5. **The DP/CRP temporal model does not help here** — its size prior reinforces the
   generic-confound cluster; the margin gate is the better use of the temporal structure.
6. The whole gain came from **reframing the geometry (online clustering), not a bigger
   model** — the strongest method is CPU-first, with the LLM contributing a modest,
   honest extraction signal on top.

## Files
- `code/streaming_eval.py` — Families A/B/C full battery → `outputs/streaming_eval_results.json`
- `code/streaming_refine.py` — 2-D gate, DP-no-size, whiten-dim sweep → `outputs/streaming_refine_results.json`
- `code/streaming_robustness.py` — honesty checks (a)+(b) → `outputs/streaming_robustness_results.json`
- `code/streaming_honest_compare.py` — honest-transform comparison → `outputs/streaming_honest_compare.json`
- `code/extract_fingerprints.py`, `code/fingerprint_linking.py` — Family D (GPU + CPU)
- `code/timeline_verify.py` — Family E (GPU)
