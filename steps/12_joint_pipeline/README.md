# Step 12 — Joint Pipeline (segments → journal) + Colab canonicalization bridge

Unifies the two halves of the project into one end-to-end run, and wires in Tomer's
NLP_ADV canonicalization so we can measure its effect on the FULL pipeline (not just
an intrinsic metric as in Step 11).

## The joint pipeline (`run_journal.py`, one GPU job)

```
segments {seg_key, date, text}
   │  1. EMBED   multilingual-e5-large
   ▼
   │  2. LINK    Step-10 online method: ABTT-k1 + accumulated centroid + margin gate
   ▼            (each segment joins the best journal entry if its distinctiveness
   │             margin clears theta, else opens a new entry)
   │  3. LOG-WRITE  Step-9 method (dictalm2): opening summary + incremental update
   ▼               per later segment, conditioned on the running journal
finished per-topic journal.json
```

`--segments` selects the input, so the SAME command runs on raw or canonical text and
the raw-vs-canonical effect is measured end-to-end:

```bash
python code/run_journal.py --segments outputs/segments_raw.json       --tag raw
python code/run_journal.py --segments outputs/segments_canonical.json --tag canonical
```
(`sbatch/12_joint_pipeline.sh` runs raw always, and canonical automatically once the
Colab output is in place.)

## Canonicalizing OUR segments on Colab (not the cluster)

Gemma-4-31B is memory-heavy and the user prefers Colab. **Colab cannot mount the
cluster filesystem** (separate networks), but there are two clean bridges:

1. **Google Drive (zero setup, default in the notebook)** — matches Tomer's existing
   workflow. Flow:
   - Cluster: `python code/export_for_canon.py` → `outputs/canon_input.json` (523 gold
     segments, raw text capped at 3500 words; the gold topic is *not* included so it
     can't leak into the canonical).
   - Upload `canon_input.json` to your Drive at
     `MyDrive/NLP ADVANCED/FinalProject/canon_input.json`.
   - Open `notebooks/canonicalize_gold.ipynb` in Colab (File → Open → GitHub, or from
     Drive), set an `HF_TOKEN` secret, run. It loads Gemma-4-31B-4bit and applies
     **Tomer's exact canonicalization prompt**. **Resumable** — re-run after any Colab
     disconnect and it skips finished segments; output `canonical_gold.json` is
     rewritten to Drive after every segment. (Long job: ~hours over 523 coarse segments.)
   - Download `canonical_gold.json` → place at `outputs/canonical_gold.json` on the cluster.
2. **Direct scp from Colab (most connected)** — the notebook's last cell scps the result
   straight to the cluster via the CS SSH gateway (needs an SSH key added in Colab). No
   Drive round-trip.

Then on the cluster: `python code/build_segments.py` (builds `segments_canonical.json`,
falling back to raw for any segment without a canonical) and re-run the joint pipeline /
`sbatch/12_joint_pipeline.sh` — it now also produces `journal_canonical.json`, and Step
11's comparison can be re-run on the real gold benchmark.

Why canonicalizing our (coarse) segments is worthwhile despite Gemma compressing them:
e5 only embeds the first ~512 tokens of raw text anyway, so a clean ~600-word canonical
distillation of the whole segment can represent it *better* than raw-truncated — exactly
Tomer's hypothesis, now testable on our full 36-repeat linking benchmark.

## Files
- `code/export_for_canon.py` → `outputs/canon_input.json` (Colab input)
- `notebooks/canonicalize_gold.ipynb` — Colab canonicalizer (Tomer's prompt, resumable)
- `code/build_segments.py` → `outputs/segments_{raw,canonical}.json`
- `code/run_journal.py` → `outputs/journal_{raw,canonical}.json`
- `sbatch/12_joint_pipeline.sh`, `code/make_colab_notebook.py`
