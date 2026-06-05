# Step 01 — Topic Preprocessing

Extract the gold topic label(s) for each protocol from the `<< נושא >>` markers,
canonicalize near-duplicates (typos / formatting), and split false-positive merges.

## Run
```bash
python steps/01_topic_preprocessing/code/preprocess_topics.py
```

## Layout
- `code/preprocess_topics.py` — extraction + fuzzy canonicalization
- `inputs/topics_corrections.json` — **manual** overrides for false-positive merges
- `outputs/topics_raw.json` — per-file raw extracted topics
- `outputs/topics_canonical.json` — per-file canonical topics ← **consumed by Step 02**
- `outputs/topics_clusters.json` — the typo/variant cluster map
- `report/data_summary_report.pdf` — dataset + preprocessing summary

## Result
213 protocols → **151 canonical topics**. All files assigned a topic
(205 via `<< נושא >>`, 4 via `<< הצח >>`, 4 via agenda fallback).
80% of topics occur in a single protocol; 30 recur across 2+.
