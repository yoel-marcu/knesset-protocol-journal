# Step 03 — Dense Embedding De-Anisotropization (ABTT)

## Motivation

Step 02 showed dense models (e5, mpnet, alephbert) have severe anisotropy:
cross-topic cosine ≈ 0.95 for e5/alephbert (gap between same/diff topics only 0.007–0.066).
This collapses clustering performance vs. TF-IDF (e5 ARI_km=0.32 vs TF-IDF ARI_km=0.38).

The fix: **All-But-The-Top (ABTT)** — center the embedding matrix, project out the
top-k principal directions (which capture language/register rather than topic), then
re-normalize. (Mu & Viswanath 2018, ICLR.)

## Method

1. Subtract mean from all 198 embeddings
2. SVD → keep top-k right singular vectors V (k × d)
3. `X_proj = X_centered - (X_centered @ Vᵀ) @ V`
4. L2-normalize rows

Sweep k ∈ {0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50}.

## How to Run

```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT
python steps/03_dense_deanisotrize/code/deanisotrize_eval.py
```

No GPU required — pure CPU/NumPy + sklearn.

## Outputs

| File | Description |
|------|-------------|
| `outputs/abtt_results.json` | All (model, k, metric) triples + TF-IDF baseline |
| `outputs/abtt_curves.png` | ARI + AUC vs k per model, TF-IDF dashed line |
| `outputs/best_embeddings/{model}_abtt_k{k}.npy` | Best-k de-anisotropized embeddings |

## Key Result

**AlephBERT k=2 beats TF-IDF on kmeans ARI** (0.391 > 0.377). E5 k=1 dominates
agglomerative (ARI=0.540 vs TF-IDF 0.249). The anisotropy was concentrated in just 1–2 PCs.

| Model | Best k | ARI_km | ARI_agg | AUC | sim_gap |
|-------|--------|--------|---------|-----|---------|
| e5 | 3 | 0.375 | 0.440 | 0.861 | 0.259 |
| e5 (k=1) | 1 | 0.363 | **0.540** | **0.899** | 0.343 |
| mpnet | 10 | 0.314 | 0.309 | 0.707 | 0.173 |
| alephbert | **2** | **0.391** | 0.513 | 0.887 | 0.344 |
| TF-IDF (baseline) | — | 0.377 | 0.249 | 0.741 | 0.069 |

Sweet spot: **k=1–3**. Parliamentary Hebrew register signal is in the top 1–2 PCs.
Over-removal (k>10) degrades all models as topic-discriminative directions are also lost.

Best embeddings saved in `outputs/best_embeddings/` for downstream steps.
