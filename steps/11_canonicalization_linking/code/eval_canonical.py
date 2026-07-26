"""
Step 11 — does canonicalization improve the linking geometry? Compares raw vs
canonical embeddings on two measures, using Tomer's `subject` labels as pseudo-gold.

  1. Same/different-subject separation AUC (the Step 3/4 intrinsic metric): for all
     segment pairs, is same-subject cosine separable from different-subject? Higher
     AUC = the embedding space groups a topic's segments together and apart from
     others -- exactly what linking needs. Reported on ALL pairs and on SUBSTANTIVE-
     only pairs (procedural subjects excluded).
  2. Streaming linking F1 (the Step 10 method: ABTT-k1 + centroid + margin gate),
     with recurring subjects as the gold chains -- run identically on raw vs canonical.

CAVEAT (stated honestly): `subject` is itself an output of the same LLM pass that
produced the canonical text, so canonical embeddings have a mild built-in advantage
on any subject-derived metric. The raw-vs-canonical *gap* is therefore an optimistic
estimate; a fully clean test needs subject labels from an independent source.

Usage (CPU): python steps/11_canonicalization_linking/code/eval_canonical.py
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"


def abtt(X, k=1):
    mu = X.mean(0, keepdims=True); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return normalize(Xc - (Xc @ Vt[:k].T) @ Vt[:k])


def pair_auc(X, labels, mask=None):
    """AUC of same-subject vs different-subject cosine over all pairs (optionally masked)."""
    idx = np.arange(len(X)) if mask is None else np.where(mask)[0]
    Xn = normalize(X[idx]); lab = np.array(labels)[idx]
    S = Xn @ Xn.T
    same, diff = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            (same if lab[a] == lab[b] else diff).append(S[a, b])
    same, diff = np.array(same), np.array(diff)
    # AUC = P(sim(same) > sim(diff))
    allv = np.concatenate([same, diff])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(allv))
    r_same = ranks[:len(same)].sum()
    auc = (r_same - len(same) * (len(same) - 1) / 2) / (len(same) * len(diff))
    return float(auc), len(same), len(diff)


def streaming_link_f1(X, rows):
    """Step-10 method (ABTT1 + centroid + margin), gold chains = recurring subjects."""
    Xt = abtt(X, 1)
    subj = [r["subject"] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["date"], rows[i]["seg_id"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(subj[i], i)
    entry_members, entry_subj, subj_to_entry = [], [], {}
    records = []
    for i in order:
        x = Xt[i]
        if entry_members:
            cos = np.array([(lambda M: (M.mean(0) / (np.linalg.norm(M.mean(0)) + 1e-12)) @ x)(Xt[m])
                            for m in entry_members])
            is_rep = first_seen[subj[i]] != i
            records.append((is_rep, cos, subj_to_entry.get(subj[i], -1) if is_rep else -1))
        if first_seen[subj[i]] == i:
            subj_to_entry[subj[i]] = len(entry_members); entry_members.append([i]); entry_subj.append(subj[i])
        else:
            entry_members[subj_to_entry[subj[i]]].append(i)
    # sweep margin
    best = {"f1": -1}
    for th in np.linspace(0, 0.6, 61):
        tp = fp = fn = tn = 0
        for is_rep, cos, gold in records:
            k = int(np.argmax(cos))
            srt = np.partition(cos, -2) if len(cos) >= 2 else np.array([cos[k], -1])
            link = (srt[-1] - srt[-2]) >= th
            if link:
                ok = is_rep and k == gold; tp += ok; fp += (not ok)
            else:
                fn += is_rep; tn += (not is_rep)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"theta": round(float(th), 3), "precision": round(prec, 3),
                    "recall": round(rec, 3), "f1": round(f1, 3), "TP": tp, "FP": fp, "FN": fn}
    n_rep = sum(r[0] for r in records)
    return best, len(records), n_rep


def main():
    rows = json.load(open(OUT / "canon_dataset.json", encoding="utf-8"))
    subj = [r["subject"] for r in rows]
    subst = np.array([not r["is_procedural"] for r in rows])
    Xr = np.load(OUT / "emb_raw.npy").astype(np.float64)
    Xc = np.load(OUT / "emb_canonical.npy").astype(np.float64)

    res = {}
    log.info("=== same/different-subject separation AUC (higher = better) ===")
    for name, X in [("raw", Xr), ("canonical", Xc)]:
        auc_all, ns, nd = pair_auc(X, subj)
        auc_sub, nss, nds = pair_auc(X, subj, mask=subst)
        res[f"auc_{name}"] = {"all_pairs": round(auc_all, 4), "substantive_only": round(auc_sub, 4)}
        log.info("  %-9s AUC all=%.4f (same=%d/diff=%d) | substantive=%.4f", name, auc_all, ns, nd, auc_sub)

    log.info("=== streaming linking F1 (Step-10 method; gold = recurring subjects) ===")
    for name, X in [("raw", Xr), ("canonical", Xc)]:
        best, ndec, nrep = streaming_link_f1(X, rows)
        res[f"link_{name}"] = best | {"n_decisions": ndec, "n_repeats": nrep}
        log.info("  %-9s best_f1=%.3f (P=%.3f R=%.3f) over %d decisions, %d true repeats",
                 name, best["f1"], best["precision"], best["recall"], ndec, nrep)

    json.dump(res, open(OUT / "canonical_eval_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "canonical_eval_results.json")


if __name__ == "__main__":
    main()
