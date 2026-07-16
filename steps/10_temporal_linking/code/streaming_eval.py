"""
Step 10 — Temporal / geometric streaming linking (Families A, B, C).

Re-frames Task 2 as ONLINE CLUSTERING rather than nearest-neighbour classification
against a frozen reference set. The current pipeline (Steps 5/7/8) represents each
journal entry by its FIRST span forever; this step exploits the three assets the
streaming setting hands us for free:

  A. Entry representation from ACCUMULATED members (centroid / robust median /
     linkage variants) instead of a frozen first span.
  B. Online DE-BIASING (mean-centering, ABTT top-k removal, shrinkage whitening)
     to suppress the shared "budget boilerplate" direction that inflates the
     false-merge confounds.
  C. Temporal DECISION RULES (DP/CRP with a size prior n_k and an adaptive NEW
     option; a distinctiveness/margin gate) instead of one flat cosine threshold.

All CPU-only on the frozen e5 embeddings from Step 07. The oracle-growth journal
(new entry on each topic's TRUE first occurrence) isolates representation quality
from decision-threshold error propagation, exactly matching build_verifier_inputs.py
so results are directly comparable to the Step 7 baseline (streaming best-F1 0.125).

STREAMING HONESTY: entry members only ever include occurrences STRICTLY BEFORE the
span being scored (the span is added to its gold entry only AFTER its decision is
recorded). De-biasing transforms are batch-fit on all spans by default (matching the
existing baseline, which batch-fits ABTT-k1); the winning config is separately
re-checked with a streaming-honest transform + true-streaming feedback in
streaming_robustness.py.

Usage (CPU):
    python steps/10_temporal_linking/code/streaming_eval.py
"""

import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"

THETA_STEPS = 101  # resolution of each sweep


# ────────────────────────────────────────────────────────────────────────────
# De-biasing transforms (batch-fit on all spans; matches the baseline's batch ABTT).
# Each returns L2-normalized transformed vectors.
# ────────────────────────────────────────────────────────────────────────────
def transform(X: np.ndarray, kind: str) -> np.ndarray:
    X = X.astype(np.float64)
    if kind == "raw":
        return normalize(X)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    if kind == "center":
        return normalize(Xc)
    if kind.startswith("abtt"):
        k = int(kind[4:])
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:k]
        return normalize(Xc - (Xc @ V.T) @ V)
    if kind.startswith("whiten"):
        # reduce to top-d PCs, shrinkage-whiten, then normalize
        d = int(kind[6:]) if len(kind) > 6 else 50
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # keep only components with meaningful variance (cold-start / rank safety)
        eps = 1e-9 * (S[0] if len(S) else 1.0)
        d = min(d, int((S > eps).sum()))
        if d < 1:
            return normalize(Xc)
        Z = Xc @ Vt[:d].T                     # project to top-d PCs
        var = (S[:d] ** 2) / len(X)
        lam = 0.1                              # shrinkage toward isotropic
        denom = (1 - lam) * var + lam * var.mean()
        w = 1.0 / np.sqrt(np.maximum(denom, 1e-12))
        Zw = Z * w
        # rows that are all-zero (degenerate) fall back to a safe unit vector
        norms = np.linalg.norm(Zw, axis=1, keepdims=True)
        Zw = np.where(norms > 1e-12, Zw, 1.0)
        return normalize(Zw)
    raise ValueError(kind)


# ────────────────────────────────────────────────────────────────────────────
# Streaming pass: records, per decision point, everything all three rules need.
# ────────────────────────────────────────────────────────────────────────────
def stream_records(Xt: np.ndarray, ids: list, entry_repr: str):
    """
    Returns a list of per-decision records. Xt is already transformed+normalized.
    entry_repr in {first, centroid, median, members_max, members_min, members_mean}.

    Each record: dict with
      gold_is_repeat : bool
      cos            : np.ndarray[num_entries]  cosine affinity to each entry
      sizes          : np.ndarray[num_entries]  n_k
      gold_entry     : int  index of the correct entry (or -1 if not a repeat)
      d0_sq          : float  squared dist of the span to the running global mean (on sphere)
    """
    topics = [r["topic"] for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)

    entry_topic = []                 # entry idx -> topic
    entry_members = []               # entry idx -> list of member row-indices (into Xt)
    topic_to_entry = {}

    global_sum = np.zeros(Xt.shape[1])
    n_seen = 0

    def entry_vec(members):
        M = Xt[members]
        if entry_repr in ("first",):
            return M[0]
        if entry_repr == "centroid":
            v = M.mean(0); return v / (np.linalg.norm(v) + 1e-12)
        if entry_repr == "median":
            # Weiszfeld geometric median
            v = M.mean(0)
            for _ in range(16):
                d = np.linalg.norm(M - v, axis=1) + 1e-9
                v_new = (M / d[:, None]).sum(0) / (1 / d).sum()
                if np.linalg.norm(v_new - v) < 1e-7:
                    v = v_new; break
                v = v_new
            return v / (np.linalg.norm(v) + 1e-12)
        return None  # members_* handled via aggregation below

    records = []
    for i in order:
        x = Xt[i]
        global_sum += x; n_seen += 1
        gmean = global_sum / n_seen
        gmean_n = gmean / (np.linalg.norm(gmean) + 1e-12)
        d0_sq = float(2 - 2 * (x @ gmean_n))

        if entry_topic:  # something to compare against
            if entry_repr.startswith("members_"):
                agg = entry_repr.split("_")[1]
                cos = np.empty(len(entry_members))
                for k, mem in enumerate(entry_members):
                    sims = Xt[mem] @ x
                    cos[k] = sims.max() if agg == "max" else sims.min() if agg == "min" else sims.mean()
            else:
                E = np.stack([entry_vec(mem) for mem in entry_members])
                cos = E @ x
            sizes = np.array([len(m) for m in entry_members], dtype=float)
            t = topics[i]
            is_repeat = first_seen[t] != i
            gold_entry = topic_to_entry.get(t, -1) if is_repeat else -1
            records.append({
                "gold_is_repeat": bool(is_repeat),
                "cos": cos.astype(np.float64),
                "sizes": sizes,
                "gold_entry": int(gold_entry),
                "d0_sq": d0_sq,
            })

        # update journal AFTER recording (streaming honesty)
        t = topics[i]
        if first_seen[t] == i:  # first occurrence -> new entry
            topic_to_entry[t] = len(entry_topic)
            entry_topic.append(t)
            entry_members.append([i])
        else:                   # repeat -> add to its (gold) entry
            entry_members[topic_to_entry[t]].append(i)

    return records


# ────────────────────────────────────────────────────────────────────────────
# Decision rules: each maps records + a swept scalar -> confusion counts.
# ────────────────────────────────────────────────────────────────────────────
def confusion(tp, fp, fn, tn):
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fmr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "false_merge_rate": round(fmr, 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def sweep_threshold(records):
    """LINK to best-cosine entry if best cosine >= theta."""
    rows = []
    for theta in np.linspace(-0.2, 1.0, THETA_STEPS):
        tp = fp = fn = tn = 0
        for r in records:
            k = int(np.argmax(r["cos"]))
            link = r["cos"][k] >= theta
            if link:
                if r["gold_is_repeat"] and k == r["gold_entry"]:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
        rows.append({"theta": round(float(theta), 4), **confusion(tp, fp, fn, tn)})
    return rows


def sweep_margin(records):
    """LINK to best-cosine entry if (best - second) >= theta_margin."""
    rows = []
    for theta in np.linspace(0.0, 0.6, THETA_STEPS):
        tp = fp = fn = tn = 0
        for r in records:
            c = r["cos"]
            k = int(np.argmax(c))
            if len(c) >= 2:
                srt = np.partition(c, -2)
                margin = srt[-1] - srt[-2]
            else:
                margin = c[k]  # only one entry: margin undefined, use magnitude
            link = margin >= theta
            if link:
                if r["gold_is_repeat"] and k == r["gold_entry"]:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
        rows.append({"theta_margin": round(float(theta), 4), **confusion(tp, fp, fn, tn)})
    return rows


def sweep_dp(records, sigma2, sigma0_2):
    """
    DP/CRP: link_score_k = log n_k - d_k^2/(2 sigma2);
            new_score    = new_bias - d0^2/(2 sigma0_2).
    LINK to argmax link_score if it exceeds new_score, else NEW. Sweep new_bias.
    (d^2 = 2 - 2 cos on the unit sphere.)
    """
    rows = []
    for new_bias in np.linspace(-6.0, 6.0, THETA_STEPS):
        tp = fp = fn = tn = 0
        for r in records:
            dk_sq = 2 - 2 * r["cos"]
            link_score = np.log(r["sizes"]) - dk_sq / (2 * sigma2)
            k = int(np.argmax(link_score))
            new_score = new_bias - r["d0_sq"] / (2 * sigma0_2)
            link = link_score[k] > new_score
            if link:
                if r["gold_is_repeat"] and k == r["gold_entry"]:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
        rows.append({"new_bias": round(float(new_bias), 3), **confusion(tp, fp, fn, tn)})
    return rows


def summarize(rows):
    best = max(rows, key=lambda r: r["f1"])
    ok = [r for r in rows if r["false_merge_rate"] <= 0.05]
    fmr5 = max(ok, key=lambda r: r["recall"]) if ok else None
    return {"best_f1": best, "best_f1_fmr<=5%": fmr5}


def estimate_sigmas(Xt, ids):
    """Pooled within-entry and global squared-distance scales (batch, for DP)."""
    from collections import defaultdict
    topics = [r["topic"] for r in ids]
    by_topic = defaultdict(list)
    for i, t in enumerate(topics):
        by_topic[t].append(i)
    within = []
    for t, idxs in by_topic.items():
        if len(idxs) >= 2:
            M = Xt[idxs]; c = M.mean(0); c /= np.linalg.norm(c) + 1e-12
            within += list(2 - 2 * (M @ c))
    gm = Xt.mean(0); gm /= np.linalg.norm(gm) + 1e-12
    glob = 2 - 2 * (Xt @ gm)
    sigma2 = float(np.mean(within)) if within else float(np.mean(glob))
    return max(sigma2, 1e-3), float(np.mean(glob))


# ────────────────────────────────────────────────────────────────────────────
def main():
    e5 = np.load(STEP07_OUT / "embeddings" / "e5.npy")
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))

    transforms = ["raw", "center", "abtt1", "abtt2", "abtt3", "whiten50"]
    reprs = ["first", "centroid", "median", "members_max", "members_min", "members_mean"]

    results = {}

    # cache transformed spaces
    Xt_cache = {tf: transform(e5, tf) for tf in transforms}

    # ---- Sweep 1: entry representation (transform fixed = abtt1, matches baseline) ----
    log.info("=== Sweep 1: entry representation (transform=abtt1, rule=threshold) ===")
    for rep in reprs:
        recs = stream_records(Xt_cache["abtt1"], ids, rep)
        s = summarize(sweep_threshold(recs))
        results[f"S1|abtt1|{rep}|threshold"] = s
        log.info("  %-14s best_f1=%.3f (P=%.3f R=%.3f)  fmr5_recall=%s",
                 rep, s["best_f1"]["f1"], s["best_f1"]["precision"], s["best_f1"]["recall"],
                 f'{s["best_f1_fmr<=5%"]["recall"]:.3f}' if s["best_f1_fmr<=5%"] else "n/a")

    # ---- Sweep 2: de-biasing transform (repr fixed = centroid) ----
    log.info("=== Sweep 2: de-biasing transform (repr=centroid, rule=threshold) ===")
    for tf in transforms:
        recs = stream_records(Xt_cache[tf], ids, "centroid")
        s = summarize(sweep_threshold(recs))
        results[f"S2|{tf}|centroid|threshold"] = s
        log.info("  %-10s best_f1=%.3f (P=%.3f R=%.3f)", tf, s["best_f1"]["f1"],
                 s["best_f1"]["precision"], s["best_f1"]["recall"])

    # ---- Sweep 3: decision rule (repr=centroid, transform=abtt1) ----
    log.info("=== Sweep 3: decision rule (transform=abtt1, repr=centroid) ===")
    recs = stream_records(Xt_cache["abtt1"], ids, "centroid")
    sig2, sig0 = estimate_sigmas(Xt_cache["abtt1"], ids)
    for rule_name, rows in [("threshold", sweep_threshold(recs)),
                            ("margin", sweep_margin(recs)),
                            ("dp", sweep_dp(recs, sig2, sig0))]:
        s = summarize(rows)
        results[f"S3|abtt1|centroid|{rule_name}"] = s
        log.info("  %-10s best_f1=%.3f (P=%.3f R=%.3f)  fmr5_recall=%s", rule_name,
                 s["best_f1"]["f1"], s["best_f1"]["precision"], s["best_f1"]["recall"],
                 f'{s["best_f1_fmr<=5%"]["recall"]:.3f}' if s["best_f1_fmr<=5%"] else "n/a")

    # ---- Sweep 4: full factorial of the promising contenders ----
    log.info("=== Sweep 4: factorial (transforms x reprs x rules) ===")
    grid_tf = ["center", "abtt1", "abtt2", "whiten50"]
    grid_rep = ["centroid", "median", "members_min", "members_mean"]
    for tf, rep in product(grid_tf, grid_rep):
        Xt = Xt_cache[tf]
        recs = stream_records(Xt, ids, rep)
        sig2, sig0 = estimate_sigmas(Xt, ids)
        for rule_name, rows in [("threshold", sweep_threshold(recs)),
                                ("margin", sweep_margin(recs)),
                                ("dp", sweep_dp(recs, sig2, sig0))]:
            results[f"S4|{tf}|{rep}|{rule_name}"] = summarize(rows)

    # ---- Baseline reproduction ----
    recs0 = stream_records(Xt_cache["abtt1"], ids, "first")
    results["BASELINE|abtt1|first|threshold"] = summarize(sweep_threshold(recs0))

    json.dump(results, open(OUT / "streaming_eval_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ranked leaderboard
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["best_f1"]["f1"])
    log.info("\n=== LEADERBOARD (top 20 by best-F1) ===")
    log.info("%-40s %6s %7s %7s %8s", "config", "F1", "P", "R", "fmr5_R")
    for name, s in ranked[:20]:
        b = s["best_f1"]; f = s["best_f1_fmr<=5%"]
        log.info("%-40s %6.3f %6.1f%% %6.1f%% %8s", name, b["f1"], b["precision"]*100,
                 b["recall"]*100, f'{f["recall"]*100:.1f}%' if f else "n/a")
    log.info("\nWrote %s", OUT / "streaming_eval_results.json")


if __name__ == "__main__":
    main()
