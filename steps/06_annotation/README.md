# Step 06 — Annotation (Milestone 1)

Creates the gold evaluation set for Task 2 (streaming subject linking).

## What to annotate

100 pairs of protocol spans, stratified:
- **50 positive** — same gold topic (ground truth SAME)
- **30 hard negatives** — different topic, highest e5-k1 cosine similarity (most confusable)
- **20 easy negatives** — different topic, random low similarity

For each pair, decide:
- **same** — same legislative matter; should be ONE journal entry
- **related** — related but distinct; separate entries, may cross-reference
- **new** — clearly different subject

**Asymmetric cost reminder:** false-merge (labeling different matters as "same") is worse
than false-split. When in doubt, prefer "related" or "new" over "same".

## Workflow

### Step 1 — Each annotator annotates independently
```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# Replace YOURNAME with: yoel / tomer / or
python steps/06_annotation/code/annotate.py --annotator YOURNAME
```
- Progress saved after every pair. Ctrl-C to pause, re-run to resume.
- Each annotator produces `outputs/annotations_YOURNAME.json`.
- Aim: ~30 min per annotator for all 100 pairs.

### Step 2 — Check agreement + adjudicate
```bash
python steps/06_annotation/code/adjudicate.py
# Resolve conflicts interactively:
python steps/06_annotation/code/adjudicate.py --resolve
```
Produces `outputs/gold_labels.json` and `outputs/agreement_stats.json`.

### Step 3 — Regenerate batch (if needed)
```bash
python steps/06_annotation/code/sample_pairs.py
```
Only needed if you want to change sampling parameters (MAX_POS_PER_TOPIC, N_HARD_NEG, etc.).

## Outputs

| File | Description |
|---|---|
| `outputs/annotation_batch.json` | Canonical 100-pair list (generated, don't edit) |
| `outputs/annotations_YOURNAME.json` | Per-annotator labels |
| `outputs/gold_labels.json` | Adjudicated gold labels (produced by adjudicate.py) |
| `outputs/agreement_stats.json` | Cohen's κ, pairwise agreement, label distribution |

## Target

- At least 2 annotators on all 100 pairs
- Cohen's κ ≥ 0.6 (substantial agreement) on the same/new binary
- "related" is inherently noisier — κ on 3-class will be lower
