"""
Step 08a — Shortlist recall@K check (retrieve-then-verify, stage 1 sanity check).

Before building an LLM verifier on top of e5_k1 retrieval, check whether the
correct prior journal entry is even *retrievable* in the top-K candidates.
If recall@K is low, the LLM verifier can never recover those misses -- garbage in,
garbage out -- and K (or the representation) needs to change before verifier design
is worth doing.

Journal construction here is "oracle-growth": a new journal entry is created on a
topic's true first occurrence (from gold_chains.json), independent of any predicted
decision -- this isolates retrieval quality from decision-threshold effects (unlike
Step 07's streaming_eval, which conflates the two).

For every one of the 36 gold repeat occurrences (a segment whose topic already has
a journal entry), rank all currently-existing journal entries by e5_k1 cosine
similarity to the new segment and record the rank of the true match.

Input:  steps/07_gold_eval/outputs/embeddings/{e5}.npy + ids.json
Output: outputs/shortlist_recall.json  -- recall@K table + per-event ranks
Usage (CPU-only):
    python steps/08_retrieve_verify/code/shortlist_recall.py
"""

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]      # steps/08_retrieve_verify
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"
EMB_DIR = STEP07_OUT / "embeddings"

E5_K = 1
K_SWEEP = [1, 2, 3, 5, 10, 20, 50]


def abtt(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


def main():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    order_key = [(r["date"], r["doc_id"], r["seg_idx"]) for r in ids]
    chron_order = np.array(sorted(range(len(ids)), key=lambda i: order_key[i]))

    e5_raw = np.load(EMB_DIR / "e5.npy")
    X = abtt(e5_raw, E5_K)
    S = (X @ X.T).astype(np.float64)

    first_seen = {}
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    journal_reps: list[int] = []       # journal entry -> representative segment idx
    journal_topic: list[str] = []      # journal entry -> topic label
    events = []  # one per gold repeat: {idx, doc_id, topic, n_candidates, rank}

    for idx in chron_order:
        t = topics[idx]
        is_repeat = (first_seen[t] != idx)

        if journal_reps:
            sims = np.array([S[idx, rep] for rep in journal_reps])
            order = np.argsort(-sims)  # descending similarity -> candidate ranking
            if is_repeat:
                correct_entry = journal_topic.index(t)  # the one true match
                rank = int(np.where(order == correct_entry)[0][0]) + 1  # 1-indexed
                events.append({
                    "doc_id": ids[idx]["doc_id"], "seg_idx": ids[idx]["seg_idx"],
                    "topic": t, "n_candidates": len(journal_reps), "rank": rank,
                })

        if not is_repeat:
            journal_reps.append(idx)
            journal_topic.append(t)

    ranks = np.array([e["rank"] for e in events])
    recall_at_k = {k: round(float((ranks <= k).mean()), 4) for k in K_SWEEP}

    log.info("Gold repeat events: %d", len(events))
    log.info("Candidate pool size at event time: min=%d median=%d max=%d",
             min(e["n_candidates"] for e in events),
             int(np.median([e["n_candidates"] for e in events])),
             max(e["n_candidates"] for e in events))
    for k in K_SWEEP:
        log.info("  recall@%-3d = %.4f", k, recall_at_k[k])

    misses = [e for e in events if e["rank"] > 20]
    if misses:
        log.info("Events with rank > 20 (would miss even a K=20 shortlist):")
        for e in misses:
            log.info("  %s seg=%d topic=%r rank=%d/%d",
                     e["doc_id"], e["seg_idx"], e["topic"][:50], e["rank"], e["n_candidates"])

    json.dump({"recall_at_k": recall_at_k, "events": events},
              open(OUT / "shortlist_recall.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "shortlist_recall.json")


if __name__ == "__main__":
    main()
