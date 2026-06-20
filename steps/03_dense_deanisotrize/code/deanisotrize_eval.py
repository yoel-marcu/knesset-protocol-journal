"""
Step 03 — De-anisotropize dense embeddings (All-But-The-Top / ABTT).

The anisotropy problem (Step 02 finding): cross-topic cosine ≈ 0.95 for e5/alephbert,
meaning the embedding space is dominated by a small number of "universal" directions
(language, register, formality) that swamp the topic signal.

Fix: remove the top-k principal components from the full embedding matrix, then
re-normalize. This is the ABTT procedure (Mu & Viswanath 2018, originally for word
vectors; applies equally to sentence embeddings).

Sweeps k ∈ K_SWEEP for each dense model (e5, mpnet, alephbert).
Evaluates clustering on the recurring subset (same protocol as Step 02).
Saves the best-k embedding per model for use in downstream steps.

Input:  steps/02_topic_clustering/outputs/embeddings/{model}.npy + ids.json
        steps/02_topic_clustering/outputs/topic_spans.json
Output: outputs/abtt_results.json        — all (model, k, metric) triples
        outputs/abtt_curves.png          — ARI + AUC vs k, TF-IDF baseline shown
        outputs/best_embeddings/{model}_abtt_k{k}.npy  — best-k embeddings per model
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
STEP02_OUT = STEP_DIR.parent / "02_topic_clustering" / "outputs"
EMB_DIR = STEP02_OUT / "embeddings"
SPANS_FILE = STEP02_OUT / "topic_spans.json"

DENSE_MODELS = ["e5", "mpnet", "alephbert"]
K_SWEEP = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def abtt(X: np.ndarray, k: int) -> np.ndarray:
    """
    Remove the top-k principal directions from X (fit on all rows), then L2-normalize.

    k=0: just L2-normalize (baseline, no removal).
    k>0: center → SVD → project out top-k right singular vectors → L2-normalize.

    Centering is done before SVD so that the SVD finds directions of maximum variance
    rather than directions pulled toward the mean. The mean is also removed from the
    final output, which is correct for cosine-similarity downstream use.
    """
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    # full_matrices=False: Vt is (min(n,d), d) — much cheaper than (d, d)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]                      # (k, d) top-k PC directions
    X_proj = X_c - (X_c @ V.T) @ V  # project out
    return normalize(X_proj).astype(np.float32)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ids_and_labels():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    return ids, topics


def recurring_mask(topics: np.ndarray) -> np.ndarray:
    counts = Counter(topics.tolist())
    return np.array([counts[t] >= 2 for t in topics])


def encode_labels(topics: np.ndarray) -> np.ndarray:
    uniq = {t: i for i, t in enumerate(sorted(set(topics.tolist())))}
    return np.array([uniq[t] for t in topics])


def build_tfidf(ids):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    spans = {
        (_s["doc_id"], _s["topic"]): _s["span_text"]
        for _s in json.load(open(SPANS_FILE, encoding="utf-8"))
        if _s["clean_for_eval"]
    }
    texts = [spans[(r["doc_id"], r["topic"])] for r in ids]
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.85,
        sublinear_tf=True, max_features=50000,
    )
    return normalize(vec.fit_transform(texts)).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_rep(X, y_rec: np.ndarray, k_topics: int) -> dict:
    """
    Cluster X (already subset to recurring spans) and evaluate against gold y_rec.
    Returns a dict with agglomerative + kmeans ARI/NMI/V and separability AUC.
    """
    import scipy.sparse as sp
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure, roc_auc_score)

    Xd = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)

    out = {}
    for name, labels in [
        ("agglomerative",
         AgglomerativeClustering(n_clusters=k_topics, metric="cosine",
                                 linkage="average").fit_predict(Xd)),
        ("kmeans",
         KMeans(n_clusters=k_topics, n_init=10, random_state=0).fit_predict(Xd)),
    ]:
        h, c, v = homogeneity_completeness_v_measure(y_rec, labels)
        out[name] = {
            "ARI": round(adjusted_rand_score(y_rec, labels), 4),
            "NMI": round(normalized_mutual_info_score(y_rec, labels), 4),
            "V": round(v, 4),
            "homogeneity": round(h, 4),
            "completeness": round(c, 4),
        }

    # separability: ROC-AUC for "same topic?" from cosine similarity
    S = Xd @ Xd.T
    n = len(y_rec)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same = (y_rec[iu[0]] == y_rec[iu[1]]).astype(int)
    if 0 < same.sum() < len(same):
        out["auc"] = round(roc_auc_score(same, sims), 4)
        out["mean_sim_same"] = round(float(sims[same == 1].mean()), 4)
        out["mean_sim_diff"] = round(float(sims[same == 0].mean()), 4)
        out["sim_gap"] = round(out["mean_sim_same"] - out["mean_sim_diff"], 4)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ids, topics = load_ids_and_labels()
    rec = recurring_mask(topics)
    y_rec = encode_labels(topics[rec])
    k_topics = int(y_rec.max()) + 1
    log.info("Recurring spans: %d | gold topics: %d", rec.sum(), k_topics)

    # TF-IDF baseline
    import scipy.sparse as sp
    tfidf = build_tfidf(ids)
    tfidf_rec = tfidf[np.where(rec)[0]]
    tfidf_m = eval_rep(tfidf_rec, y_rec, k_topics)
    tfidf_ari_km = tfidf_m["kmeans"]["ARI"]
    log.info("TF-IDF baseline  ARI_km=%.4f  ARI_agg=%.4f  AUC=%.4f  gap=%.4f",
             tfidf_ari_km, tfidf_m["agglomerative"]["ARI"],
             tfidf_m.get("auc", float("nan")), tfidf_m.get("sim_gap", float("nan")))

    results = {"tfidf_baseline": tfidf_m, "dense": {}}
    best_k: dict[str, int] = {}

    for model in DENSE_MODELS:
        X = np.load(EMB_DIR / f"{model}.npy")
        rows = []
        best_ari, best_k_val, best_X_abtt = -1.0, 0, None

        for k in K_SWEEP:
            X_abtt = abtt(X, k)
            X_rec = X_abtt[rec]
            m = eval_rep(X_rec, y_rec, k_topics)
            rows.append({"k": k, **m})
            ari_km = m["kmeans"]["ARI"]
            log.info("%-10s k=%2d  ARI_km=%.4f  ARI_agg=%.4f  AUC=%s  gap=%s",
                     model, k, ari_km, m["agglomerative"]["ARI"],
                     f"{m.get('auc', float('nan')):.4f}",
                     f"{m.get('sim_gap', float('nan')):.4f}")
            if ari_km > best_ari:
                best_ari, best_k_val, best_X_abtt = ari_km, k, X_abtt

        results["dense"][model] = rows
        best_k[model] = best_k_val
        save_path = OUT / "best_embeddings" / f"{model}_abtt_k{best_k_val}.npy"
        np.save(save_path, best_X_abtt)
        log.info("%s  best k=%d  ARI_km=%.4f  (saved %s)", model, best_k_val, best_ari,
                 save_path.name)

    json.dump(results, open(OUT / "abtt_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "abtt_results.json")
    _plot(results, best_k, tfidf_ari_km)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results: dict, best_k: dict, tfidf_ari_km: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot")
        return

    colors = {"e5": "#1f77b4", "mpnet": "#ff7f0e", "alephbert": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = [
        ("kmeans ARI",       lambda r: r["kmeans"]["ARI"]),
        ("agglomerative ARI", lambda r: r["agglomerative"]["ARI"]),
        ("AUC (same-topic?)", lambda r: r.get("auc", float("nan"))),
    ]

    for ax, (title, getter) in zip(axes, metrics_to_plot):
        if "ARI" in title:
            tfidf_val = (results["tfidf_baseline"]["kmeans"]["ARI"] if "kmeans" in title
                         else results["tfidf_baseline"]["agglomerative"]["ARI"])
            ax.axhline(tfidf_val, color="red", linestyle="--", linewidth=1.5,
                       label=f"TF-IDF ({tfidf_val:.3f})")
        else:
            tfidf_val = results["tfidf_baseline"].get("auc", float("nan"))
            ax.axhline(tfidf_val, color="red", linestyle="--", linewidth=1.5,
                       label=f"TF-IDF ({tfidf_val:.3f})")

        for model, rows in results["dense"].items():
            ks = [r["k"] for r in rows]
            vals = [getter(r) for r in rows]
            ax.plot(ks, vals, marker="o", color=colors[model], label=model)
            bk = best_k.get(model, 0)
            if bk in ks:
                bi = ks.index(bk)
                ax.scatter([bk], [vals[bi]], color=colors[model], s=100, zorder=5,
                           edgecolors="black", linewidths=0.8)

        ax.set_xlabel("k (PCs removed)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("All-But-The-Top (ABTT) de-anisotropization — Step 03")
    fig.tight_layout()
    out_path = OUT / "abtt_curves.png"
    fig.savefig(out_path, dpi=130)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
