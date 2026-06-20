"""
Step 05 — Streaming Subject Linking: Similarity Threshold Baseline (Task 2a).

Evaluates whether cosine similarity alone can decide: "does this new protocol span
belong to an existing journal entry, or is it a new subject?"

Two evaluation modes:

  Pairwise (upper bound):
    For each ordered pair (i, j) of recurring spans (same topic = LINK gold),
    predict LINK if sim(i,j) > θ. Threshold sweep → P/R/F1/AUC.
    This assumes a perfect journal with all past spans accessible — an oracle view.

  Streaming simulation (realistic):
    Process all 198 spans in chronological order. Maintain a journal whose entries
    are represented by the first span seen for each cluster. For each new span:
      - Compute sim to all current journal representatives.
      - If max_sim > θ: LINK to best match.
      - Else: NEW (create fresh journal entry).
    Evaluate: gold = same topic seen before → should LINK; first occurrence → should NEW.
    Reports TP, FP (false merge), FN (false split), TN, with separate F1s.

Representations compared:
  e5_k1       ABTT k=1 on e5-large-1024d  (best AUC in Step 03)
  alephbert_k2  ABTT k=2 on alephbert-768d  (best clustering ARI in Step 04)
  hybrid      0.6·TF-IDF + 0.4·alephbert_k2 (best ARI_agg=0.569 in Step 04)

Asymmetric cost: false-merge (FP) >> false-split (FN). Report precision-priority
threshold (FP-rate ≤ 5%) alongside F1-optimal.

Input:  steps/02/outputs/embeddings/ids.json + topic_spans.json
        steps/03/outputs/best_embeddings/alephbert_abtt_k2.npy
        steps/02/outputs/embeddings/e5.npy  (for e5_k1 recompute)
        PROTOCOLS/*.json  (for chronological ordering)
Output: outputs/linking_results.json
        outputs/pr_curves_pairwise.png
        outputs/pr_curves_streaming.png
"""

import json
import logging
from collections import Counter, defaultdict
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
PROTOCOLS_DIR = STEP_DIR.parents[1] / "PROTOCOLS"

THETA_SWEEP = np.linspace(0.0, 1.0, 201)


# ---------------------------------------------------------------------------
# ABTT (self-contained copy; same as Steps 03–04)
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

def load_data():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])

    # Dates from protocol files
    dates = []
    for r in ids:
        p = PROTOCOLS_DIR / f"{r['doc_id']}.json"
        dates.append(json.load(open(p, encoding="utf-8"))["date"] if p.exists() else "")
    dates = np.array(dates)

    return ids, topics, dates


def build_tfidf_all(ids):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    spans_lookup = {
        (_s["doc_id"], _s["topic"]): _s["span_text"]
        for _s in json.load(open(SPANS_FILE, encoding="utf-8"))
        if _s["clean_for_eval"]
    }
    texts = [spans_lookup[(r["doc_id"], r["topic"])] for r in ids]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.85,
                          sublinear_tf=True, max_features=50000)
    return normalize(vec.fit_transform(texts)).astype(np.float32)


# ---------------------------------------------------------------------------
# Build 198×198 similarity matrices for each representation
# ---------------------------------------------------------------------------

def build_sim_matrices(ids, topics):
    import scipy.sparse as sp

    log.info("Building e5_k1 similarity matrix ...")
    e5_raw = np.load(EMB_DIR / "e5.npy")
    X_e5k1 = abtt(e5_raw, 1)
    S_e5k1 = (X_e5k1 @ X_e5k1.T).astype(np.float64)

    log.info("Building alephbert_k2 similarity matrix ...")
    X_abk2 = np.load(STEP03_OUT / "best_embeddings" / "alephbert_abtt_k2.npy").astype(np.float64)
    S_abk2 = X_abk2 @ X_abk2.T

    log.info("Building hybrid (alephbert_k2 α=0.6) similarity matrix ...")
    tfidf = build_tfidf_all(ids)
    S_tfidf = (tfidf @ tfidf.T).toarray().astype(np.float64)
    S_hybrid = 0.6 * S_tfidf + 0.4 * S_abk2

    return {
        "e5_k1": S_e5k1,
        "alephbert_k2": S_abk2,
        "hybrid_α0.6": S_hybrid,
    }


# ---------------------------------------------------------------------------
# Pairwise evaluation (recurring subset only)
# ---------------------------------------------------------------------------

def pairwise_eval(S: np.ndarray, topics: np.ndarray) -> dict:
    """
    Threshold sweep on all ordered pairs (i < j) within the recurring subset.
    Gold: same_topic(i,j) → LINK; different_topic → NOT-LINK.
    Returns list of {theta, precision, recall, f1, false_merge_rate}.
    """
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
    n_pos = gold.sum()
    n_neg = len(gold) - n_pos

    rows = []
    for theta in THETA_SWEEP:
        pred = (sims >= theta).astype(int)
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        tn = int(((pred == 0) & (gold == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fmr = fp / n_neg if n_neg > 0 else 0.0   # false-merge rate
        rows.append({"theta": round(float(theta), 3), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "false_merge_rate": round(fmr, 4),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    best_f1 = max(rows, key=lambda r: r["f1"])
    # Precision-priority threshold: highest recall s.t. false_merge_rate <= 5%
    prec_priority = max(
        (r for r in rows if r["false_merge_rate"] <= 0.05),
        key=lambda r: r["recall"], default=rows[0]
    )
    log.info("  [pairwise] AUC=%.4f  best_F1=%.4f@θ=%.2f  "
             "prec_priority: F1=%.4f@θ=%.2f (FMR≤5%%)",
             auc, best_f1["f1"], best_f1["theta"],
             prec_priority["f1"], prec_priority["theta"])

    return {"auc": round(auc, 4), "curve": rows,
            "best_f1": best_f1, "prec_priority_θ": prec_priority}


# ---------------------------------------------------------------------------
# Streaming simulation
# ---------------------------------------------------------------------------

def streaming_eval(S: np.ndarray, topics: np.ndarray, dates: np.ndarray) -> dict:
    """
    Simulate online subject linking in chronological order.

    Journal: dict {representative_idx → [list of linked span indices]}.
    Decision: LINK (sim > θ) to best matching representative, or NEW.
    Gold: LINK if same topic has been seen before; NEW otherwise.

    Returns threshold curve + best-F1 / precision-priority points.
    """
    # Chronological order (stable sort on date string — ISO format sorts lexically)
    chron_order = np.argsort(dates, kind="stable")

    # Pre-compute: for each span, is this its first occurrence in chron order?
    first_seen = {}   # topic → chronological index of first occurrence
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    # For each threshold, simulate the journal
    n_pos_total = sum(1 for idx in chron_order
                      if topics[idx] in first_seen and
                      first_seen[topics[idx]] != idx)   # not-first occurrences

    rows = []
    for theta in THETA_SWEEP:
        journal_reps = []    # representative span indices (one per journal entry)
        journal_topics = []  # gold topic label per journal entry (for evaluation only)
        tp = fp = fn = tn = 0

        for idx in chron_order:
            t = topics[idx]
            gold_link = (first_seen[t] != idx)  # True = this topic seen before

            if not journal_reps:
                journal_reps.append(idx)
                journal_topics.append(t)
                # First span ever → always NEW, gold=NEW → TN
                tn += 1
                continue

            sims_j = [float(S[idx, rep]) for rep in journal_reps]
            best_sim = max(sims_j)
            best_j = int(np.argmax(sims_j))

            if best_sim >= theta:
                # Predict LINK
                if gold_link:
                    # Did we link to the right entry?
                    # For binary eval we count TP regardless of which entry (any correct link)
                    if journal_topics[best_j] == t:
                        tp += 1
                    else:
                        # Linked to wrong entry — counts as FP (false merge) AND FN (missed right link)
                        fp += 1
                        fn += 1
                else:
                    fp += 1  # false merge: linked a new topic to an existing one
                # Regardless of correctness, update representative to first-seen (stable repr)
                # (we do NOT update — representative stays as the entry's first span)
            else:
                # Predict NEW → create new journal entry
                if gold_link:
                    fn += 1   # false split: missed a recurring topic
                else:
                    tn += 1   # correct: this was genuinely a new topic

                journal_reps.append(idx)
                journal_topics.append(t)

        n_neg_total = len(chron_order) - n_pos_total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fmr = fp / n_neg_total if n_neg_total > 0 else 0.0
        rows.append({"theta": round(float(theta), 3), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "false_merge_rate": round(fmr, 4),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    best_f1 = max(rows, key=lambda r: r["f1"])
    prec_priority = max(
        (r for r in rows if r["false_merge_rate"] <= 0.05),
        key=lambda r: r["recall"], default=rows[0]
    )
    log.info("  [streaming] best_F1=%.4f@θ=%.2f (P=%.3f R=%.3f)  "
             "prec_priority: F1=%.4f@θ=%.2f (FMR≤5%%)",
             best_f1["f1"], best_f1["theta"], best_f1["precision"], best_f1["recall"],
             prec_priority["f1"], prec_priority["theta"])

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
        log.warning("matplotlib not available")
        return

    colors = {"e5_k1": "#1f77b4", "alephbert_k2": "#2ca02c", "hybrid_α0.6": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, data in results.items():
        curve = data[mode]["curve"]
        prec = [r["precision"] for r in curve]
        rec  = [r["recall"]    for r in curve]
        fmr  = [r["false_merge_rate"] for r in curve]
        thetas = [r["theta"] for r in curve]
        c = colors.get(name, "gray")
        bf = data[mode]["best_f1"]
        auc_label = f" (AUC={data['pairwise']['auc']:.3f})" if mode == "pairwise" else ""
        axes[0].plot(rec, prec, color=c, label=f"{name}{auc_label}", linewidth=1.5)
        axes[0].scatter([bf["recall"]], [bf["precision"]], color=c, s=80, zorder=5,
                        edgecolors="black", linewidths=0.7)
        axes[1].plot(thetas, fmr, color=c, label=name, linewidth=1.5)

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall ({mode})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1, label="5% FMR limit")
    axes[1].set_xlabel("θ (similarity threshold)")
    axes[1].set_ylabel("False-Merge Rate")
    axes[1].set_title(f"False-Merge Rate vs θ ({mode})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Subject Linking Baseline — {mode} evaluation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)
    ids, topics, dates = load_data()
    log.info("Loaded %d spans, %d unique topics, %d recurring topics",
             len(ids), len(set(topics.tolist())),
             sum(1 for c in Counter(topics.tolist()).values() if c >= 2))

    sim_matrices = build_sim_matrices(ids, topics)

    all_results = {}
    for name, S in sim_matrices.items():
        log.info("--- %s ---", name)
        pw = pairwise_eval(S, topics)
        st = streaming_eval(S, topics, dates)
        all_results[name] = {"pairwise": pw, "streaming": st}

    json.dump(all_results, open(OUT / "linking_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote linking_results.json")

    _plot_pr(all_results, "pairwise",  OUT / "pr_curves_pairwise.png")
    _plot_pr(all_results, "streaming", OUT / "pr_curves_streaming.png")

    # Print summary table
    log.info("\n=== SUMMARY ===")
    log.info("%-18s | %-30s | %-30s", "Representation",
             "Pairwise (F1-opt / FMR≤5%)", "Streaming (F1-opt / FMR≤5%)")
    log.info("-" * 85)
    for name, d in all_results.items():
        pw_b = d["pairwise"]["best_f1"]
        pw_p = d["pairwise"]["prec_priority_θ"]
        st_b = d["streaming"]["best_f1"]
        st_p = d["streaming"]["prec_priority_θ"]
        log.info("%-18s | F1=%.3f@θ=%.2f / F1=%.3f@θ=%.2f | F1=%.3f@θ=%.2f / F1=%.3f@θ=%.2f",
                 name,
                 pw_b["f1"], pw_b["theta"], pw_p["f1"], pw_p["theta"],
                 st_b["f1"], st_b["theta"], st_p["f1"], st_p["theta"])


if __name__ == "__main__":
    main()
