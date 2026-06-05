"""
Stage 3 of the topic-clustering pipeline: cluster + evaluate all representations.

Compares 4 representations of the clean per-topic spans:
  e5, mpnet, alephbert (dense, from embed.py) + tfidf (built here, CPU).

For each representation:
  * Clustering: Agglomerative (cosine, average linkage, K=#gold topics) and KMeans(K).
  * Supervised metrics vs gold canonical topic: ARI, NMI, V-measure, homogeneity, completeness.
    Computed on the RECURRING subset (topics with >=2 spans) where clustering is non-trivial.
  * Separability: intra-topic vs inter-topic cosine similarity, summarized as ROC-AUC for the
    "same topic?" decision -- a direct preview of the Task-2 similarity-threshold baseline.

Outputs:
  outputs/clustering_results.json   -- all metrics
  outputs/clustering_umap.png       -- 2D UMAP per representation, colored by recurring topic
  console summary table

Usage:
    python src/clustering/cluster_eval.py
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]      # steps/02_topic_clustering
OUT = STEP_DIR / "outputs"
EMB_DIR = OUT / "embeddings"
SPANS_FILE = OUT / "topic_spans.json"

DENSE_MODELS = ["e5", "mpnet", "alephbert"]


def load_ids_and_labels():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = [r["topic"] for r in ids]
    # integer-encode gold labels
    uniq = {t: i for i, t in enumerate(sorted(set(topics)))}
    y = np.array([uniq[t] for t in topics])
    return ids, topics, y


def build_tfidf(ids):
    """TF-IDF over the same spans, in ids row order."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    spans = {(_s["doc_id"], _s["topic"]): _s["span_text"]
             for _s in json.load(open(SPANS_FILE, encoding="utf-8")) if _s["clean_for_eval"]}
    texts = [spans[(r["doc_id"], r["topic"])] for r in ids]
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.85,
        sublinear_tf=True, max_features=50000,
    )
    X = vec.fit_transform(texts)
    # L2-normalize rows -> cosine == dot
    from sklearn.preprocessing import normalize
    return normalize(X).astype(np.float32)


def recurring_mask(topics):
    counts = Counter(topics)
    return np.array([counts[t] >= 2 for t in topics])


def supervised_metrics(X, y, k):
    """Agglomerative + KMeans clustering metrics vs gold labels y (already subset)."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure)
    import scipy.sparse as sp

    Xd = X.toarray() if sp.issparse(X) else X
    out = {}
    # Agglomerative with cosine distance
    agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    la = agg.fit_predict(Xd)
    # KMeans (euclidean on normalized == cosine-ish)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    lk = km.fit_predict(Xd)

    for name, lab in [("agglomerative", la), ("kmeans", lk)]:
        h, c, v = homogeneity_completeness_v_measure(y, lab)
        out[name] = {
            "ARI": round(adjusted_rand_score(y, lab), 4),
            "NMI": round(normalized_mutual_info_score(y, lab), 4),
            "V": round(v, 4),
            "homogeneity": round(h, 4),
            "completeness": round(c, 4),
        }
    return out


def separability_auc(X, y):
    """ROC-AUC of 'same gold topic' predicted by pairwise cosine similarity."""
    import scipy.sparse as sp
    from sklearn.metrics import roc_auc_score
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    S = Xd @ Xd.T
    n = len(y)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same = (y[iu[0]] == y[iu[1]]).astype(int)
    if same.sum() == 0 or same.sum() == len(same):
        return None, None, None
    auc = roc_auc_score(same, sims)
    return round(auc, 4), round(float(sims[same == 1].mean()), 4), round(float(sims[same == 0].mean()), 4)


def main():
    ids, topics, y_full = load_ids_and_labels()
    topics = np.array(topics)
    rec = recurring_mask(topics)
    n_rec_topics = len(set(topics[rec]))
    log.info("Total spans: %d | recurring spans: %d | recurring topics: %d",
             len(ids), rec.sum(), n_rec_topics)

    # Load all representations
    reps = {}
    for m in DENSE_MODELS:
        p = EMB_DIR / f"{m}.npy"
        if p.exists():
            reps[m] = np.load(p)
        else:
            log.warning("Missing %s -- run embed.py first", p)
    reps["tfidf"] = build_tfidf(ids)

    # Re-encode gold labels within the recurring subset
    rec_topics = topics[rec]
    uniq = {t: i for i, t in enumerate(sorted(set(rec_topics)))}
    y_rec = np.array([uniq[t] for t in rec_topics])

    results = {}
    log.info("\n%-10s | %-13s | %5s %5s %5s | %5s | sameSim infSim",
             "rep", "cluster", "ARI", "NMI", "V", "AUC")
    log.info("-" * 78)
    for name, X in reps.items():
        import scipy.sparse as sp
        Xrec = X[rec] if not sp.issparse(X) else X[np.where(rec)[0]]
        metrics = supervised_metrics(Xrec, y_rec, k=n_rec_topics)
        auc, same_s, inf_s = separability_auc(Xrec, y_rec)
        results[name] = {"clustering": metrics, "auc_same_topic": auc,
                         "mean_sim_same": same_s, "mean_sim_diff": inf_s}
        for cl in ("agglomerative", "kmeans"):
            mm = metrics[cl]
            log.info("%-10s | %-13s | %.3f %.3f %.3f | %s | %s %s",
                     name, cl, mm["ARI"], mm["NMI"], mm["V"],
                     f"{auc:.3f}" if auc else "  -  ",
                     same_s, inf_s)

    json.dump(results, open(OUT / "clustering_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("\nWrote %s", OUT / "clustering_results.json")

    _umap_plot(reps, topics, rec)


def _umap_plot(reps, topics, rec):
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("umap-learn / matplotlib not available; skipping UMAP plot")
        return
    import scipy.sparse as sp

    rec_topics = topics[rec]
    uniq = sorted(set(rec_topics))
    color = {t: i for i, t in enumerate(uniq)}
    c = [color[t] for t in rec_topics]

    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, X) in zip(axes, reps.items()):
        Xrec = (X[np.where(rec)[0]].toarray() if sp.issparse(X) else X[rec])
        emb = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="cosine",
                        random_state=0).fit_transform(Xrec)
        ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap="tab20", s=18)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("UMAP of recurring-topic spans (color = gold topic)")
    fig.tight_layout()
    fig.savefig(OUT / "clustering_umap.png", dpi=130)
    log.info("Wrote %s", OUT / "clustering_umap.png")


if __name__ == "__main__":
    main()
