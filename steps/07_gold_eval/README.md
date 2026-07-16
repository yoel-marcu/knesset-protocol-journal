# Step 07 — Gold Evaluation Data & Re-Validation ✅ DONE

Converts the manual segmentation annotations (Step 06, 98% complete) into real gold
data for both coupled tasks, then re-runs Steps 02/03/04/05's evaluations against it
to check whether their marker-derived-gold findings hold up on the real thing.
Replaces the plan in `steps/06_annotation/README.md` (sampled 100-pair linking set),
superseded by full-protocol manual segmentation.

## Input
`steps/06_annotation/outputs/segmentation.json` — 211/213 protocols annotated
(2 missing entirely; 6 have at least one unconfirmed `auto=true` segment saved
without edits). No label canonicalization applied: near-duplicate topic strings
across protocols are not merged, so some recurring topics surface as extra
singleton chains. Accepted per the project's own asymmetric-cost rule
(false-split << false-merge) rather than doing manual/LLM dedup review.

## Pipeline
```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp

python steps/07_gold_eval/code/build_gold.py           # segmentation.json -> gold_segments/gold_chains
# --- GPU, via sbatch/07_gold_reeval.sh ---
python steps/07_gold_eval/code/embed_gold.py           # e5 + alephbert embeddings of gold spans
python steps/07_gold_eval/code/gold_reeval.py           # Task 1 clustering + Task 2 pairwise/streaming
# --- CPU-only follow-up, run locally ---
python steps/07_gold_eval/code/gold_abtt_and_hybrid.py  # ABTT k-sweep + hybrid clustering on gold
```

## Outputs
- `outputs/gold_segments.json` — Task 1 gold, 523 segments, same shape as `topic_spans.json`.
- `outputs/gold_chains.json` — Task 2 gold, 487 unique labels, 22 recurring (36 implied SAME pairs).
- `outputs/embeddings/{e5,alephbert}.npy` + `ids.json`
- `outputs/gold_clustering.json`, `outputs/gold_reeval_results.json`,
  `outputs/gold_pr_curves_{pairwise,streaming}.png`
- `outputs/gold_abtt_sweep.json`, `outputs/gold_hybrid_clustering.json`

## Results

**Task 1 — clustering** (58 recurring gold segments, 22 gold topics):

| repr | ARI (agglomerative) | ARI (kmeans) | AUC |
|---|---|---|---|
| tfidf | 0.382 | 0.407 | 0.874 |
| **e5_k1** | **0.657** | **0.515** | **0.954** |
| alephbert_k2 | 0.265 | 0.274 | 0.854 |
| hybrid_α0.6 | 0.265 (spectral 0.411) | — | 0.884 |

**Task 2 — pairwise (oracle) linking:** e5_k1 best F1=0.62@θ=0.26, AUC=0.954 — all confirmed better than the Step 03/05 marker-based estimates.

**Task 2 — streaming (realistic) linking — the headline result:**

| repr | best F1 | precision @ that point | recall |
|---|---|---|---|
| e5_k1 | 0.125 | 7.7% | 33% |
| alephbert_k2 | 0.075 | 4.2% | 33% |
| hybrid_α0.6 | 0.090 | 5.2% | 33% |

Streaming F1 collapsed from Step 05's 0.41 (marker-based) to 0.13 (gold). At the best
threshold, ~92% of "LINK" predictions are false merges.

## Key finding

Not a representation problem — e5_k1 is still the best of everything tried, and by more
than before. It's a task-formulation problem: gold segmentation is far more fine-grained
(523 segments, only 22 ever recur) than the old per-protocol canonical topics (30/151
recurring), so true "same subject" pairs are rare, and any similarity threshold loose
enough to catch them is loose enough to flood in unrelated same-sounding topics. No
fixed cosine cutoff can be both loose enough for recall and tight enough for precision
at this granularity.

## Follow-up checks (confirmed e5_k1, ruled out alternatives)

- **Hybrid does not replicate**: Step 04's "hybrid α=0.6 beats everything" finding does
  not hold on gold data — hybrid is worse than plain e5_k1 on every clustering metric.
- **ABTT k mostly holds**: e5 k=1 is still ~optimal (highest AUC of any k in the sweep).
  alephbert's fixed k=2 was suboptimal on gold (k=5 does better), but doesn't matter —
  alephbert never gets close to e5_k1 at any k.
- **Conclusion: no further representation tuning needed.** e5_k1 (unchanged) is the
  representation to carry forward.

## Next (not started — scoping next session)

Retrieve-then-verify (CLAUDE.md Milestone 3b): use e5_k1 similarity only as a
high-recall shortlist (top-K candidates), then an LLM verifier reads each candidate
against the new segment and decides same/related/new. Open questions for next
session: pick K, design the verifier prompt/label schema, and check shortlist
recall@K against `gold_chains.json`'s 22 recurring chains before trusting the
pipeline (if the true match isn't even in the top-K, the LLM never gets a chance
to catch it).
