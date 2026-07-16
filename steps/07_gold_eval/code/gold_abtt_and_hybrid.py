"""
Step 07d — Follow-up analysis on gold data:
  1. ABTT k-sweep (was fixed at e5 k=1 / alephbert k=2 from the marker-based
     Step 03 sweep in gold_reeval.py) -- check those are still the best k on
     the real, finer-grained gold segments rather than assuming it transfers.
  2. hybrid_α0.6 clustering score (gold_reeval.py only scored it for linking,
     not Task 1 clustering) -- fused similarity has no raw feature matrix, so
     it needs Step 04's precomputed-similarity clustering (agglomerative on
     1-S distance + spectral), not KMeans.

CPU-only: reuses embeddings already written by embed_gold.py, no GPU needed.

Usage:
    python steps/07_gold_eval/code/gold_abtt_and_hybrid.py
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
EMB_DIR = OUT / "embeddings"
SPANS_FILE = OUT / "gold_segments.json"

K_SWEEP = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def abtt(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


def recurring_mask(topics):
    counts = Counter(topics.tolist())
    return np.array([counts[t] >= 2 for t in topics])


def encode_labels(topics):
    uniq = {t: i for i, t in enumerate(sorted(set(topics.tolist())))}
    return np.array([uniq[t] for t in topics])


def eval_feat(X, y_rec, k_topics) -> dict:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    Xd = np.asarray(X, dtype=np.float64)
    ari_agg = adjusted_rand_score(
        y_rec, AgglomerativeClustering(n_clusters=k_topics, metric="cosine",
                                       linkage="average").fit_predict(Xd))
    ari_km = adjusted_rand_score(
        y_rec, KMeans(n_clusters=k_topics, n_init=10, random_state=0).fit_predict(Xd))
    S = Xd @ Xd.T
    n = len(y_rec)
    iu = np.triu_indices(n, k=1)
    same = (y_rec[iu[0]] == y_rec[iu[1]]).astype(int)
    auc = roc_auc_score(same, S[iu]) if 0 < same.sum() < len(same) else float("nan")
    return {"ARI_agg": round(ari_agg, 4), "ARI_km": round(ari_km, 4), "AUC": round(auc, 4)}


def eval_sim(S: np.ndarray, y_rec, k_topics) -> dict:
    """Precomputed-similarity clustering (for hybrid, which has no raw features)."""
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    D = np.clip(1.0 - S, 0, None)
    A = np.maximum(S, 0)
    ari_agg = adjusted_rand_score(
        y_rec, AgglomerativeClustering(n_clusters=k_topics, metric="precomputed",
                                       linkage="average").fit_predict(D))
    ari_spec = adjusted_rand_score(
        y_rec, SpectralClustering(n_clusters=k_topics, affinity="precomputed",
                                  n_init=10, random_state=0).fit_predict(A))
    n = len(y_rec)
    iu = np.triu_indices(n, k=1)
    same = (y_rec[iu[0]] == y_rec[iu[1]]).astype(int)
    auc = roc_auc_score(same, S[iu]) if 0 < same.sum() < len(same) else float("nan")
    return {"ARI_agg": round(ari_agg, 4), "ARI_spectral": round(ari_spec, 4), "AUC": round(auc, 4)}


def build_tfidf(ids):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    spans_lookup = {
        (s["doc_id"], s["seg_idx"]): s["span_text"]
        for s in json.load(open(SPANS_FILE, encoding="utf-8"))
        if s["clean_for_eval"]
    }
    texts = [spans_lookup[(r["doc_id"], r["seg_idx"])] for r in ids]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.85,
                          sublinear_tf=True, max_features=50000)
    return normalize(vec.fit_transform(texts)).astype(np.float32)


def main():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    rec = recurring_mask(topics)
    y_rec = encode_labels(topics[rec])
    k_topics = int(y_rec.max()) + 1
    log.info("Recurring gold spans: %d, k_topics: %d", rec.sum(), k_topics)

    # --- 1. ABTT k-sweep, re-checked on gold ---
    sweep_results = {}
    for name in ["e5", "alephbert"]:
        X = np.load(EMB_DIR / f"{name}.npy")
        rows = []
        for k in K_SWEEP:
            X_abtt = abtt(X, k)
            m = eval_feat(X_abtt[rec], y_rec, k_topics)
            rows.append({"k": k, **m})
        sweep_results[name] = rows
        best = max(rows, key=lambda r: r["ARI_km"])
        log.info("%-10s best k=%2d  ARI_km=%.4f  ARI_agg=%.4f  AUC=%.4f  (fixed choice was k=%d)",
                 name, best["k"], best["ARI_km"], best["ARI_agg"], best["AUC"],
                 1 if name == "e5" else 2)
        for r in rows:
            log.info("  %-10s k=%2d  ARI_km=%.4f  ARI_agg=%.4f  AUC=%.4f",
                     name, r["k"], r["ARI_km"], r["ARI_agg"], r["AUC"])
    json.dump(sweep_results, open(OUT / "gold_abtt_sweep.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # --- 2. hybrid_α0.6 clustering (precomputed-similarity, fixed k from Step 03) ---
    e5_abtt = abtt(np.load(EMB_DIR / "e5.npy"), 1)
    aleph_abtt = abtt(np.load(EMB_DIR / "alephbert.npy"), 2)
    tfidf = build_tfidf(ids)

    S_aleph_rec = (aleph_abtt[rec] @ aleph_abtt[rec].T).astype(np.float64)
    tfidf_rec = tfidf[np.where(rec)[0]]
    S_tfidf_rec = (tfidf_rec @ tfidf_rec.T).toarray().astype(np.float64)
    S_hybrid_rec = 0.6 * S_tfidf_rec + 0.4 * S_aleph_rec

    hybrid_clustering = eval_sim(S_hybrid_rec, y_rec, k_topics)
    log.info("hybrid_α0.6 clustering: ARI_agg=%.4f  ARI_spectral=%.4f  AUC=%.4f",
             hybrid_clustering["ARI_agg"], hybrid_clustering["ARI_spectral"],
             hybrid_clustering["AUC"])
    json.dump(hybrid_clustering, open(OUT / "gold_hybrid_clustering.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
