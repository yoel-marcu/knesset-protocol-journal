# Step 02 — Topic Clustering

Feasibility study: using only discussion **content** (no header/agenda/attendees/speaker
names), do same-topic protocols cluster together? Compares 4 representations.

## Run (in order)
```bash
python steps/02_topic_clustering/code/extract_spans.py   # Stage 1: clean per-topic spans
python steps/02_topic_clustering/code/embed.py           # Stage 2: dense embeddings (GPU)
python steps/02_topic_clustering/code/cluster_eval.py    # Stage 3: TF-IDF + cluster + metrics
```

## Layout
- `code/extract_spans.py` — body isolation + segmentation → `outputs/topic_spans.json`
- `code/embed.py` — e5 / mpnet / alephbert dense vectors → `outputs/embeddings/*.npy`
- `code/cluster_eval.py` — TF-IDF + clustering + metrics → `clustering_results.json`, `clustering_umap.png`
- `report/clustering_report.pdf` — full write-up + future directions

## Result
On the 105 recurring-topic spans (28 topics): **TF-IDF wins** (ARI 0.38, NMI 0.80) >
e5 > mpnet ≈ alephbert. Dense embeddings are severely anisotropic (cross-topic cosine ≈ 0.95);
the discriminative signal is **lexical** (bill numbers, entities), not semantic — as the
project brief predicted.

## Depends on
Step 01 `outputs/topics_canonical.json` (gold labels).
