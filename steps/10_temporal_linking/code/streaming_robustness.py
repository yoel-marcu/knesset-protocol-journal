"""
Step 10 (robustness) — does the winning geometric config survive the two
optimistic assumptions baked into streaming_eval.py?

  (a) STREAMING-HONEST TRANSFORM: streaming_eval batch-fits the de-biasing
      transform on all 523 spans (matching the existing baseline, which also
      batch-fits ABTT-k1). Here we refit the transform at every step on only the
      spans seen so far, so no future information leaks into the geometry.

  (b) TRUE-STREAMING FEEDBACK: streaming_eval uses ORACLE growth (each span is
      added to its GOLD entry). Here the journal grows by the MODEL's own
      decisions -- a wrong LINK contaminates an entry and can propagate. Measured
      by pairwise-F1 of the induced clustering vs gold (a false-merge creates many
      wrong same-pairs, so pairwise-F1 reflects the anti-duplication cost), for the
      baseline vs the winner at each one's best operating point.

CPU-only. Usage: python steps/10_temporal_linking/code/streaming_robustness.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

import streaming_eval as se

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"


# ── (a) streaming-honest transform: refit each step on spans seen so far ──────
def stream_records_honest(e5, ids, kind, entry_repr):
    topics = [r["topic"] for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)

    entry_topic, entry_members, topic_to_entry = [], [], {}
    records = []
    for p, i in enumerate(order):
        seen = order[: p + 1]
        Xt = se.transform(e5[seen], kind)              # honest: only spans <= t
        x = Xt[p]
        gmean = normalize(Xt.mean(0, keepdims=True))[0]
        d0_sq = float(2 - 2 * (x @ gmean))
        if entry_topic:
            cos = np.empty(len(entry_members))
            for k, mem in enumerate(entry_members):
                M = Xt[mem]
                if entry_repr == "median":
                    v = M.mean(0)
                    for _ in range(16):
                        d = np.linalg.norm(M - v, axis=1) + 1e-9
                        v = (M / d[:, None]).sum(0) / (1 / d).sum()
                    v = v / (np.linalg.norm(v) + 1e-12)
                    cos[k] = v @ x
                else:  # centroid
                    v = M.mean(0); v /= np.linalg.norm(v) + 1e-12
                    cos[k] = v @ x
            t = topics[i]; is_rep = first_seen[t] != i
            records.append({"gold_is_repeat": bool(is_rep), "cos": cos,
                            "gold_entry": topic_to_entry.get(t, -1) if is_rep else -1,
                            "d0_sq": d0_sq})
        t = topics[i]
        if first_seen[t] == i:
            topic_to_entry[t] = len(entry_topic); entry_topic.append(t); entry_members.append([p])
        else:
            entry_members[topic_to_entry[t]].append(p)
    return records


# ── (b) true-streaming feedback: journal grows by the model's own decisions ───
def pairwise_f1(labels_pred, labels_gold):
    from collections import defaultdict
    def pairs(labels):
        groups = defaultdict(list)
        for i, l in enumerate(labels):
            groups[l].append(i)
        s = set()
        for g in groups.values():
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    s.add((g[a], g[b]))
        return s
    P, G = pairs(labels_pred), pairs(labels_gold)
    if not P:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    tp = len(P & G)
    prec = tp / len(P); rec = tp / len(G) if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def run_feedback(Xt, ids, rule, theta, entry_repr):
    """Grow the journal by the model's decisions; return induced cluster labels."""
    topics = [r["topic"] for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    entry_members = []                 # list of member row-indices
    labels = [None] * len(ids)

    def entry_cos(x):
        cos = np.empty(len(entry_members))
        for k, mem in enumerate(entry_members):
            M = Xt[mem]
            if entry_repr == "first":
                cos[k] = M[0] @ x
            elif entry_repr == "median":
                v = M.mean(0)
                for _ in range(12):
                    d = np.linalg.norm(M - v, axis=1) + 1e-9
                    v = (M / d[:, None]).sum(0) / (1 / d).sum()
                v /= np.linalg.norm(v) + 1e-12; cos[k] = v @ x
            else:
                v = M.mean(0); v /= np.linalg.norm(v) + 1e-12; cos[k] = v @ x
        return cos

    for i in order:
        x = Xt[i]
        if not entry_members:
            entry_members.append([i]); labels[i] = 0; continue
        cos = entry_cos(x); k = int(np.argmax(cos))
        if rule == "threshold":
            link = cos[k] >= theta
        else:  # margin
            srt = np.partition(cos, -2) if len(cos) >= 2 else np.array([cos[k], -1])
            link = (srt[-1] - srt[-2]) >= theta
        if link:
            entry_members[k].append(i); labels[i] = k
        else:
            entry_members.append([i]); labels[i] = len(entry_members) - 1
    return labels


def best_feedback_f1(Xt, ids, rule, thetas, entry_repr, gold):
    best = {"f1": -1}
    for th in thetas:
        labels = run_feedback(Xt, ids, rule, th, entry_repr)
        m = pairwise_f1(labels, gold)
        m["theta"] = round(float(th), 3); m["n_clusters"] = len(set(labels))
        if m["f1"] > best["f1"]:
            best = m
    return best


def main():
    e5 = np.load(STEP07_OUT / "embeddings" / "e5.npy")
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    gold = [r["topic"] for r in ids]
    results = {}

    # (a) streaming-honest transform vs batch, on the winner
    log.info("=== (a) streaming-honest transform (winner: whiten50 + median + margin) ===")
    recs_honest = stream_records_honest(e5, ids, "whiten50", "median")
    s_honest = se.summarize(se.sweep_margin(recs_honest))
    recs_batch = se.stream_records(se.transform(e5, "whiten50"), ids, "median")
    s_batch = se.summarize(se.sweep_margin(recs_batch))
    results["honest_transform"] = {"streaming_honest": s_honest["best_f1"],
                                   "batch_transform": s_batch["best_f1"]}
    log.info("  batch-fit transform : best_f1=%.3f (P=%.3f R=%.3f)",
             s_batch["best_f1"]["f1"], s_batch["best_f1"]["precision"], s_batch["best_f1"]["recall"])
    log.info("  streaming-honest    : best_f1=%.3f (P=%.3f R=%.3f)",
             s_honest["best_f1"]["f1"], s_honest["best_f1"]["precision"], s_honest["best_f1"]["recall"])

    # (b) true-streaming feedback: pairwise-F1 of induced clustering vs gold
    log.info("=== (b) true-streaming feedback (pairwise-F1 of induced clusters) ===")
    Xt_base = se.transform(e5, "abtt1")
    Xt_win = se.transform(e5, "whiten50")
    base = best_feedback_f1(Xt_base, ids, "threshold", np.linspace(0.2, 0.95, 40), "first", gold)
    win = best_feedback_f1(Xt_win, ids, "margin", np.linspace(0.02, 0.5, 40), "median", gold)
    results["feedback_pairwise_f1"] = {"baseline_first_threshold": base,
                                       "winner_whiten_median_margin": win}
    log.info("  baseline (first+threshold): pairwise F1=%.3f (P=%.3f R=%.3f, %d clusters @theta=%.2f)",
             base["f1"], base["precision"], base["recall"], base["n_clusters"], base["theta"])
    log.info("  winner (whiten+median+margin): pairwise F1=%.3f (P=%.3f R=%.3f, %d clusters @theta=%.2f)",
             win["f1"], win["precision"], win["recall"], win["n_clusters"], win["theta"])
    log.info("  (gold has %d clusters)", len(set(gold)))

    json.dump(results, open(OUT / "streaming_robustness_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "streaming_robustness_results.json")


if __name__ == "__main__":
    main()
