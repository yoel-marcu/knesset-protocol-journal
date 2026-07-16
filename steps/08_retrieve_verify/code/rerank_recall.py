"""
Step 08f — Rerank the already-retrieved top-10 (e5_k1) candidates using a
lexical/entity-overlap signal (TF-IDF full-text cosine), and re-check recall@K
on the 36 gold repeat events.

Motivation (see miss_inspection.txt): all 11 rank>1 misses are same-committee
distractors from one recurring genre (year-end budget "עודפים" reallocation
requests) that are template-near-duplicates of each other -- the only thing
that differentiates them is a short ministry/entity name diluted by mean-
pooling over a long, mostly-generic discussion. TF-IDF is exactly the lexical/
entity-name signal that should break these ties.

Deliberately narrower than Step 04's global hybrid blend: this only reranks
*within* the already-retrieved top-10 (which has 100% recall@10), so it can't
hurt cases outside that shortlist, and can't hurt already-correct rank-1 cases
unless the lexical signal actively disagrees with the dense signal there.

Sweeps a combination weight beta: score = dense_sim + beta * tfidf_sim
(both already in cosine space, so directly additive; tfidf_sim is 0-1, dense
cosine is roughly 0.2-0.8 in this corpus per miss_inspection.txt).

Usage (CPU-only):
    python steps/08_retrieve_verify/code/rerank_recall.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parents[1]
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"
EMB_DIR = STEP07_OUT / "embeddings"
SPANS_FILE = STEP07_OUT / "gold_segments.json"

E5_K = 1
TOP_K = 10
BETA_SWEEP = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def abtt(X, k):
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


def build_tfidf(ids):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    span_lookup = {(s["doc_id"], s["seg_idx"]): s["span_text"]
                   for s in json.load(open(SPANS_FILE, encoding="utf-8")) if s["clean_for_eval"]}
    texts = [span_lookup[(r["doc_id"], r["seg_idx"])] for r in ids]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.85,
                          sublinear_tf=True, max_features=50000)
    return normalize(vec.fit_transform(texts)).astype(np.float32)


def main():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    order_key = [(r["date"], r["doc_id"], r["seg_idx"]) for r in ids]
    chron_order = np.array(sorted(range(len(ids)), key=lambda i: order_key[i]))

    X = abtt(np.load(EMB_DIR / "e5.npy"), E5_K)
    S_dense = (X @ X.T).astype(np.float64)

    tfidf = build_tfidf(ids)
    S_tfidf = (tfidf @ tfidf.T).toarray().astype(np.float64)

    first_seen = {}
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    journal_reps, journal_topic = [], []
    # per event: candidate journal indices (top-10 by dense), dense sims, tfidf sims, correct pos
    events = []

    for idx in chron_order:
        t = topics[idx]
        is_repeat = (first_seen[t] != idx)
        if journal_reps and is_repeat:
            dense_sims = np.array([S_dense[idx, rep] for rep in journal_reps])
            top10_local = np.argsort(-dense_sims)[:TOP_K]  # indices into journal_reps
            correct_entry = journal_topic.index(t)
            pos_in_top10 = np.where(top10_local == correct_entry)[0]
            if len(pos_in_top10):  # true match is within the dense top-10 (always true here)
                tfidf_sims = np.array([S_tfidf[idx, journal_reps[ci]] for ci in top10_local])
                events.append({
                    "dense_top10": top10_local,
                    "dense_sims": dense_sims[top10_local],
                    "tfidf_sims": tfidf_sims,
                    "correct_local_rank": int(pos_in_top10[0]) + 1,  # baseline rank (1-10)
                })
        if not is_repeat:
            journal_reps.append(idx)
            journal_topic.append(t)

    baseline_ranks = [e["correct_local_rank"] for e in events]
    print(f"Baseline (dense-only) rank distribution: {dict(sorted(Counter(baseline_ranks).items()))}")
    print(f"Baseline recall@1: {np.mean([r == 1 for r in baseline_ranks]):.3f}\n")

    for beta in BETA_SWEEP:
        ranks = []
        for e in events:
            combined = e["dense_sims"] + beta * e["tfidf_sims"]
            order = np.argsort(-combined)
            # correct entry is always at local position (correct_local_rank - 1) within dense_top10
            correct_pos_in_dense_top10 = e["correct_local_rank"] - 1
            new_rank = int(np.where(order == correct_pos_in_dense_top10)[0][0]) + 1
            ranks.append(new_rank)
        recall1 = np.mean([r == 1 for r in ranks])
        recall3 = np.mean([r <= 3 for r in ranks])
        print(f"beta={beta:<4} recall@1={recall1:.3f}  recall@3={recall3:.3f}  "
              f"rank_dist={dict(sorted(Counter(ranks).items()))}")


if __name__ == "__main__":
    main()
