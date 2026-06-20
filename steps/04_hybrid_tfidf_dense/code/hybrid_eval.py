"""
Step 04 — Hybrid TF-IDF + ABTT Dense Score Fusion.

Motivation: TF-IDF captures lexical signal (bill IDs, law names, specific Hebrew terms)
while ABTT-corrected dense embeddings capture distributional/semantic similarity.
They should be complementary — TF-IDF wins on exact-match topics, dense on paraphrase.

Method: score fusion on precomputed cosine similarity matrices.
    S_hybrid = α · S_tfidf + (1-α) · S_dense
Sweep α ∈ {0.0, 0.1, ..., 1.0} for each dense model.

Clustering uses S_hybrid directly:
  - Agglomerative: precomputed distance = 1 − S_hybrid (average linkage)
  - Spectral:      precomputed affinity = max(S_hybrid, 0)

Evaluation protocol: recurring subset (105 spans, 28 gold topics), same as Steps 02–03.

Input:  steps/02_topic_clustering/outputs/     -- ids.json, topic_spans.json
        steps/03_dense_deanisotrize/outputs/best_embeddings/  -- saved ABTT embeddings
        steps/02_topic_clustering/outputs/embeddings/e5.npy   -- for e5_k1 recompute
Output: outputs/hybrid_results.json
        outputs/hybrid_curves.png
        outputs/best_sim_matrix.npy   -- best hybrid similarity matrix (Task 2 baseline)
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
STEP03_OUT = STEP_DIR.parent / "03_dense_deanisotrize" / "outputs"
EMB_DIR = STEP02_OUT / "embeddings"
SPANS_FILE = STEP02_OUT / "topic_spans.json"

ALPHA_SWEEP = [round(a * 0.1, 1) for a in range(11)]   # 0.0 to 1.0


# ---------------------------------------------------------------------------
# ABTT (repeated from Step 03 to keep this step self-contained)
# ---------------------------------------------------------------------------

def abtt(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    X_proj = X_c - (X_c @ V.T) @ V
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
# ABTT embeddings to evaluate: saved best-k + e5 k=1 (best agglomerative)
# ---------------------------------------------------------------------------

def load_dense_variants(rec: np.ndarray) -> dict:
    """Returns {name: X_rec (n_rec, d)} for each ABTT variant."""
    variants = {}

    for fname in sorted((STEP03_OUT / "best_embeddings").glob("*.npy")):
        X = np.load(fname)
        variants[fname.stem] = X[rec]

    # e5 k=1 was best for agglomerative but wasn't saved (step 03 optimized for kmeans ARI)
    e5_raw = np.load(EMB_DIR / "e5.npy")
    variants["e5_abtt_k1"] = abtt(e5_raw, 1)[rec]

    return variants


# ---------------------------------------------------------------------------
# Similarity matrices (precomputed, 105×105)
# ---------------------------------------------------------------------------

def sim_matrix(X) -> np.ndarray:
    """Cosine similarity matrix. Works for dense numpy or sparse scipy."""
    import scipy.sparse as sp
    if sp.issparse(X):
        S = (X @ X.T).toarray()
    else:
        S = X @ X.T
    return S.astype(np.float64)


# ---------------------------------------------------------------------------
# Clustering + evaluation on precomputed similarity matrix
# ---------------------------------------------------------------------------

def eval_sim(S: np.ndarray, y_rec: np.ndarray, k_topics: int) -> dict:
    """
    Cluster using S (cosine similarity matrix) and evaluate vs gold y_rec.

    Agglomerative: distance = 1 - S (average linkage, metric='precomputed').
    Spectral: affinity = max(S, 0), eigen-decomposition + kmeans inside.
    """
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure, roc_auc_score)

    D = np.clip(1.0 - S, 0, None)  # distance matrix; clip to avoid tiny negatives
    A = np.maximum(S, 0)           # affinity matrix; negatives → 0

    agg = AgglomerativeClustering(n_clusters=k_topics, metric="precomputed",
                                  linkage="average")
    labels_agg = agg.fit_predict(D)

    spec = SpectralClustering(n_clusters=k_topics, affinity="precomputed",
                              n_init=10, random_state=0)
    labels_spec = spec.fit_predict(A)

    out = {}
    for name, labels in [("agglomerative", labels_agg), ("spectral", labels_spec)]:
        h, c, v = homogeneity_completeness_v_measure(y_rec, labels)
        out[name] = {
            "ARI": round(adjusted_rand_score(y_rec, labels), 4),
            "NMI": round(normalized_mutual_info_score(y_rec, labels), 4),
            "V": round(v, 4),
            "homogeneity": round(h, 4),
            "completeness": round(c, 4),
        }

    # separability AUC — uses the raw similarity scores, not cluster labels
    n = len(y_rec)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same = (y_rec[iu[0]] == y_rec[iu[1]]).astype(int)
    if 0 < same.sum() < len(same):
        out["auc"] = round(roc_auc_score(same, sims), 4)
        out["sim_gap"] = round(float(sims[same == 1].mean() - sims[same == 0].mean()), 4)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)

    ids, topics = load_ids_and_labels()
    rec = recurring_mask(topics)
    y_rec = encode_labels(topics[rec])
    k_topics = int(y_rec.max()) + 1
    log.info("Recurring spans: %d | gold topics: %d", rec.sum(), k_topics)

    # Precompute TF-IDF similarity matrix on recurring subset
    tfidf = build_tfidf(ids)
    import scipy.sparse as sp
    tfidf_rec = tfidf[np.where(rec)[0]]
    S_tfidf = sim_matrix(tfidf_rec)

    # TF-IDF-only baseline (α=1.0 equivalent)
    tfidf_m = eval_sim(S_tfidf, y_rec, k_topics)
    log.info("TF-IDF only  ARI_agg=%.4f  ARI_spec=%.4f  AUC=%.4f  gap=%.4f",
             tfidf_m["agglomerative"]["ARI"], tfidf_m["spectral"]["ARI"],
             tfidf_m.get("auc", float("nan")), tfidf_m.get("sim_gap", float("nan")))

    # Load all dense variants
    dense_variants = load_dense_variants(rec)
    log.info("Dense variants: %s", list(dense_variants.keys()))

    results = {"tfidf_baseline": tfidf_m, "models": {}}
    global_best = {"ari_agg": -1.0, "ari_spec": -1.0, "auc": -1.0}
    best_S_agg = None

    for variant_name, X_abtt_rec in dense_variants.items():
        S_dense = sim_matrix(X_abtt_rec)
        rows = []
        best_ari_agg, best_alpha_agg = -1.0, 0.5

        for alpha in ALPHA_SWEEP:
            S_h = alpha * S_tfidf + (1 - alpha) * S_dense
            m = eval_sim(S_h, y_rec, k_topics)
            rows.append({"alpha": alpha, **m})
            log.info("%-22s α=%.1f  ARI_agg=%.4f  ARI_spec=%.4f  AUC=%s  gap=%s",
                     variant_name, alpha,
                     m["agglomerative"]["ARI"], m["spectral"]["ARI"],
                     f"{m.get('auc', float('nan')):.4f}",
                     f"{m.get('sim_gap', float('nan')):.4f}")

            if m["agglomerative"]["ARI"] > best_ari_agg:
                best_ari_agg = m["agglomerative"]["ARI"]
                best_alpha_agg = alpha

            if m["agglomerative"]["ARI"] > global_best["ari_agg"]:
                global_best["ari_agg"] = m["agglomerative"]["ARI"]
                global_best["model"] = variant_name
                global_best["alpha"] = alpha
                best_S_agg = S_h.copy()

        results["models"][variant_name] = rows
        log.info("%s  best α=%.1f  ARI_agg=%.4f", variant_name, best_alpha_agg, best_ari_agg)

    log.info("\n=== GLOBAL BEST ===")
    log.info("Model=%s  α=%.1f  ARI_agg=%.4f",
             global_best.get("model"), global_best.get("alpha"), global_best["ari_agg"])

    json.dump(results, open(OUT / "hybrid_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "hybrid_results.json")

    if best_S_agg is not None:
        np.save(OUT / "best_sim_matrix.npy", best_S_agg.astype(np.float32))
        log.info("Saved best hybrid similarity matrix → best_sim_matrix.npy")

    _plot(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot")
        return

    tfidf_ari_agg = results["tfidf_baseline"]["agglomerative"]["ARI"]
    tfidf_ari_spec = results["tfidf_baseline"]["spectral"]["ARI"]
    tfidf_auc = results["tfidf_baseline"].get("auc", float("nan"))

    colors = {
        "e5_abtt_k3": "#1f77b4",
        "e5_abtt_k1": "#aec7e8",
        "mpnet_abtt_k10": "#ff7f0e",
        "alephbert_abtt_k2": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_cfg = [
        ("ARI (agglomerative)", lambda r: r["agglomerative"]["ARI"], tfidf_ari_agg),
        ("ARI (spectral)",      lambda r: r["spectral"]["ARI"],      tfidf_ari_spec),
        ("AUC (same-topic?)",   lambda r: r.get("auc", float("nan")), tfidf_auc),
    ]

    for ax, (title, getter, tfidf_val) in zip(axes, metric_cfg):
        ax.axhline(tfidf_val, color="red", linestyle="--", linewidth=1.5,
                   label=f"TF-IDF only ({tfidf_val:.3f})")
        for name, rows in results["models"].items():
            alphas = [r["alpha"] for r in rows]
            vals = [getter(r) for r in rows]
            color = colors.get(name, "gray")
            ax.plot(alphas, vals, marker="o", color=color, label=name, linewidth=1.5)
            # Mark best point
            best_v = max(vals)
            best_a = alphas[vals.index(best_v)]
            ax.scatter([best_a], [best_v], color=color, s=100, zorder=5,
                       edgecolors="black", linewidths=0.8)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel("α (weight on TF-IDF)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        # α=0 label = pure dense, α=1 = pure TF-IDF
        ax.set_xticks(ALPHA_SWEEP)
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_SWEEP], rotation=45, fontsize=7)

    fig.suptitle("Hybrid TF-IDF + ABTT Dense Score Fusion — Step 04")
    fig.tight_layout()
    out_path = OUT / "hybrid_curves.png"
    fig.savefig(out_path, dpi=130)
    log.info("Wrote %s", out_path)


ALPHA_SWEEP = [round(a * 0.1, 1) for a in range(11)]


if __name__ == "__main__":
    main()
