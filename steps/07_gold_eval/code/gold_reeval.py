"""
Step 07c — Re-evaluate Step 02 (clustering) and Step 05 (linking) against
real gold data (gold_segments.json / gold_chains.json) instead of the
<< נושא >>-marker-derived spans they were validated on originally.

ABTT k is NOT re-swept here: k=1 (e5) and k=2 (alephbert) are fixed at the
values Step 03 found best on the marker-derived spans. This is a confirmatory
check ("does the winning representation still win on real gold segments?"),
not a new hyperparameter search.

Representations compared (same three as Step 05):
  e5_k1         ABTT k=1 on e5-large
  alephbert_k2  ABTT k=2 on alephbert
  hybrid_α0.6   0.6*TF-IDF + 0.4*alephbert_k2

Task 1 (clustering): recurring gold segments only (chain len >= 2), evaluated
  against the gold_chains grouping (agglomerative + kmeans ARI/NMI).
Task 2 (linking): pairwise (oracle) + streaming simulation, evaluated against
  gold_chains SAME/NEW ground truth -- reuses Step 05's exact eval functions.

Input:  outputs/gold_segments.json, outputs/gold_chains.json
        outputs/embeddings/{e5,alephbert}.npy + ids.json  (from embed_gold.py)
Output: outputs/gold_reeval_results.json
        outputs/gold_pr_curves_{pairwise,streaming}.png
        outputs/gold_clustering.json

Usage:
    python steps/07_gold_eval/code/gold_reeval.py
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]      # steps/07_gold_eval
OUT = STEP_DIR / "outputs"
EMB_DIR = OUT / "embeddings"
SPANS_FILE = OUT / "gold_segments.json"

E5_K = 1
ALEPHBERT_K = 2
THETA_SWEEP = np.linspace(0.0, 1.0, 201)


# ---------------------------------------------------------------------------
# ABTT (self-contained copy; same as Steps 03-05)
# ---------------------------------------------------------------------------

def abtt(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ids_topics_dates():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    dates = np.array([r["date"] for r in ids])
    return ids, topics, dates


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


# ---------------------------------------------------------------------------
# Task 1: clustering eval (mirrors Step 03's eval_rep)
# ---------------------------------------------------------------------------

def recurring_mask(topics: np.ndarray) -> np.ndarray:
    counts = Counter(topics.tolist())
    return np.array([counts[t] >= 2 for t in topics])


def encode_labels(topics: np.ndarray) -> np.ndarray:
    uniq = {t: i for i, t in enumerate(sorted(set(topics.tolist())))}
    return np.array([uniq[t] for t in topics])


def eval_clustering(X, y_rec: np.ndarray, k_topics: int) -> dict:
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
        }

    S = Xd @ Xd.T
    n = len(y_rec)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same = (y_rec[iu[0]] == y_rec[iu[1]]).astype(int)
    if 0 < same.sum() < len(same):
        out["auc"] = round(roc_auc_score(same, sims), 4)
        out["sim_gap"] = round(
            float(sims[same == 1].mean()) - float(sims[same == 0].mean()), 4)
    return out


# ---------------------------------------------------------------------------
# Task 2: linking eval (verbatim logic from Step 05)
# ---------------------------------------------------------------------------

def pairwise_eval(S: np.ndarray, topics: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    counts = Counter(topics.tolist())
    rec_idx = np.array([i for i, t in enumerate(topics) if counts[t] >= 2])
    y = topics[rec_idx]
    Srec = S[np.ix_(rec_idx, rec_idx)]

    n = len(rec_idx)
    iu = np.triu_indices(n, k=1)
    sims = Srec[iu]
    gold = (y[iu[0]] == y[iu[1]]).astype(int)

    auc = roc_auc_score(gold, sims)
    n_neg = len(gold) - gold.sum()

    rows = []
    for theta in THETA_SWEEP:
        pred = (sims >= theta).astype(int)
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fmr = fp / n_neg if n_neg > 0 else 0.0
        rows.append({"theta": round(float(theta), 3), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "false_merge_rate": round(fmr, 4)})

    best_f1 = max(rows, key=lambda r: r["f1"])
    prec_priority = max(
        (r for r in rows if r["false_merge_rate"] <= 0.05),
        key=lambda r: r["recall"], default=rows[0])
    log.info("  [pairwise] AUC=%.4f  best_F1=%.4f@θ=%.2f  prec_priority F1=%.4f@θ=%.2f",
             auc, best_f1["f1"], best_f1["theta"], prec_priority["f1"], prec_priority["theta"])
    return {"auc": round(auc, 4), "curve": rows, "best_f1": best_f1,
            "prec_priority_θ": prec_priority}


def streaming_eval(S: np.ndarray, topics: np.ndarray, dates: np.ndarray, order_key) -> dict:
    chron_order = np.array(sorted(range(len(topics)), key=lambda i: order_key[i]))

    first_seen = {}
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    n_pos_total = sum(1 for idx in chron_order
                      if first_seen[topics[idx]] != idx)

    rows = []
    for theta in THETA_SWEEP:
        journal_reps, journal_topics = [], []
        tp = fp = fn = tn = 0

        for idx in chron_order:
            t = topics[idx]
            gold_link = (first_seen[t] != idx)

            if not journal_reps:
                journal_reps.append(idx)
                journal_topics.append(t)
                tn += 1
                continue

            sims_j = [float(S[idx, rep]) for rep in journal_reps]
            best_sim = max(sims_j)
            best_j = int(np.argmax(sims_j))

            if best_sim >= theta:
                if gold_link:
                    if journal_topics[best_j] == t:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                else:
                    fp += 1
            else:
                if gold_link:
                    fn += 1
                else:
                    tn += 1
                journal_reps.append(idx)
                journal_topics.append(t)

        n_neg_total = len(chron_order) - n_pos_total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fmr = fp / n_neg_total if n_neg_total > 0 else 0.0
        rows.append({"theta": round(float(theta), 3), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "false_merge_rate": round(fmr, 4)})

    best_f1 = max(rows, key=lambda r: r["f1"])
    prec_priority = max(
        (r for r in rows if r["false_merge_rate"] <= 0.05),
        key=lambda r: r["recall"], default=rows[0])
    log.info("  [streaming] best_F1=%.4f@θ=%.2f  prec_priority F1=%.4f@θ=%.2f",
             best_f1["f1"], best_f1["theta"], prec_priority["f1"], prec_priority["theta"])
    return {"curve": rows, "best_f1": best_f1, "prec_priority_θ": prec_priority}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_pr(results: dict, mode: str, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot")
        return

    colors = {"e5_k1": "#1f77b4", "alephbert_k2": "#2ca02c", "hybrid_α0.6": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, data in results.items():
        curve = data[mode]["curve"]
        prec = [r["precision"] for r in curve]
        rec = [r["recall"] for r in curve]
        fmr = [r["false_merge_rate"] for r in curve]
        thetas = [r["theta"] for r in curve]
        c = colors.get(name, "gray")
        bf = data[mode]["best_f1"]
        auc_label = f" (AUC={data['pairwise']['auc']:.3f})" if mode == "pairwise" else ""
        axes[0].plot(rec, prec, color=c, label=f"{name}{auc_label}", linewidth=1.5)
        axes[0].scatter([bf["recall"]], [bf["precision"]], color=c, s=80, zorder=5,
                        edgecolors="black", linewidths=0.7)
        axes[1].plot(thetas, fmr, color=c, label=name, linewidth=1.5)

    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall ({mode}) -- GOLD")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1, label="5% FMR limit")
    axes[1].set_xlabel("θ"); axes[1].set_ylabel("False-Merge Rate")
    axes[1].set_title(f"False-Merge Rate vs θ ({mode}) -- GOLD")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Subject Linking on GOLD segmentation -- {mode}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ids, topics, dates = load_ids_topics_dates()
    order_key = [(ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]) for i in range(len(ids))]
    n_recurring_topics = sum(1 for c in Counter(topics.tolist()).values() if c >= 2)
    log.info("Gold spans: %d | unique topics: %d | recurring topics: %d",
             len(ids), len(set(topics.tolist())), n_recurring_topics)

    tfidf = build_tfidf(ids)
    e5_raw = np.load(EMB_DIR / "e5.npy")
    aleph_raw = np.load(EMB_DIR / "alephbert.npy")
    X_e5k1 = abtt(e5_raw, E5_K)
    X_abk2 = abtt(aleph_raw, ALEPHBERT_K)

    S_e5k1 = (X_e5k1 @ X_e5k1.T).astype(np.float64)
    S_abk2 = (X_abk2 @ X_abk2.T).astype(np.float64)
    S_tfidf = (tfidf @ tfidf.T).toarray().astype(np.float64)
    S_hybrid = 0.6 * S_tfidf + 0.4 * S_abk2

    sim_matrices = {"e5_k1": S_e5k1, "alephbert_k2": S_abk2, "hybrid_α0.6": S_hybrid}

    # --- Task 1: clustering on recurring subset ---
    rec = recurring_mask(topics)
    y_rec = encode_labels(topics[rec])
    k_topics = int(y_rec.max()) + 1
    clustering_results = {"tfidf": eval_clustering(tfidf[np.where(rec)[0]], y_rec, k_topics)}
    for name, X in [("e5_k1", X_e5k1), ("alephbert_k2", X_abk2)]:
        clustering_results[name] = eval_clustering(X[rec], y_rec, k_topics)
    log.info("=== Task 1: clustering (n_recurring=%d, k_topics=%d) ===", rec.sum(), k_topics)
    for name, m in clustering_results.items():
        log.info("  %-14s kmeans ARI=%.4f  agg ARI=%.4f  AUC=%s",
                 name, m["kmeans"]["ARI"], m["agglomerative"]["ARI"], m.get("auc"))
    json.dump(clustering_results, open(OUT / "gold_clustering.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # --- Task 2: linking ---
    linking_results = {}
    for name, S in sim_matrices.items():
        log.info("--- %s ---", name)
        pw = pairwise_eval(S, topics)
        st = streaming_eval(S, topics, dates, order_key)
        linking_results[name] = {"pairwise": pw, "streaming": st}
    json.dump(linking_results, open(OUT / "gold_reeval_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    _plot_pr(linking_results, "pairwise", OUT / "gold_pr_curves_pairwise.png")
    _plot_pr(linking_results, "streaming", OUT / "gold_pr_curves_streaming.png")

    log.info("\n=== SUMMARY (gold) ===")
    for name, d in linking_results.items():
        pw_b, st_b = d["pairwise"]["best_f1"], d["streaming"]["best_f1"]
        log.info("%-14s pairwise F1=%.3f@θ=%.2f | streaming F1=%.3f@θ=%.2f",
                 name, pw_b["f1"], pw_b["theta"], st_b["f1"], st_b["theta"])


if __name__ == "__main__":
    main()
