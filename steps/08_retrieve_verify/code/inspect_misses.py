"""
Step 08e — For every one of the 36 gold repeat events, print the full ranked
candidate list (not just top-10) so a human/LLM analyst can inspect *why*
e5_k1 ranks something above the true match when rank > 1 (recall@1 = 69%,
i.e. 11/36 events have a closer distractor than the real match).

Not a pipeline step with saved outputs -- a one-off diagnostic to look for a
systematic embedding-space pattern (fixable with e.g. more ABTT, per-committee
centering, etc.) rather than a modeling/prompting problem.

Usage (CPU-only):
    python steps/08_retrieve_verify/code/inspect_misses.py > outputs/miss_inspection.txt
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


def abtt(X, k):
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


def snip(text, n=40):
    words = text.split()
    return " ".join(words[:n])


def main():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    order_key = [(r["date"], r["doc_id"], r["seg_idx"]) for r in ids]
    chron_order = np.array(sorted(range(len(ids)), key=lambda i: order_key[i]))

    span_lookup = {(s["doc_id"], s["seg_idx"]): s
                   for s in json.load(open(SPANS_FILE, encoding="utf-8")) if s["clean_for_eval"]}

    X = abtt(np.load(EMB_DIR / "e5.npy"), E5_K)
    S = (X @ X.T).astype(np.float64)

    first_seen = {}
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    journal_reps, journal_topic = [], []
    ranks = []
    ev = 0

    for idx in chron_order:
        t = topics[idx]
        is_repeat = (first_seen[t] != idx)

        if journal_reps and is_repeat:
            ev += 1
            rec = ids[idx]
            span = span_lookup[(rec["doc_id"], rec["seg_idx"])]

            sims = np.array([S[idx, rep] for rep in journal_reps])
            order_full = np.argsort(-sims)
            correct_entry = journal_topic.index(t)
            rank = int(np.where(order_full == correct_entry)[0][0]) + 1
            ranks.append(rank)

            true_rep = ids[journal_reps[correct_entry]]
            true_span = span_lookup[(true_rep["doc_id"], true_rep["seg_idx"])]

            print(f"=== event {ev}/36 | query {rec['doc_id']}#{rec['seg_idx']} "
                  f"({rec['date'][:10]}) | true match rank={rank}/{len(journal_reps)} "
                  f"| topic={t[:70]}")
            print(f"  QUERY snippet: {snip(span['span_text'])}")
            print(f"  TRUE MATCH (sim={sims[correct_entry]:.4f}): "
                  f"{true_rep['doc_id']}#{true_rep['seg_idx']} ({true_rep['date'][:10]}, "
                  f"committee={true_span.get('committee', '')})")
            print(f"    snippet: {snip(true_span['span_text'])}")

            if rank > 1:
                print("  DISTRACTORS ranked ABOVE the true match:")
                for r in range(rank - 1):
                    ci = order_full[r]
                    d_rep = ids[journal_reps[ci]]
                    d_span = span_lookup[(d_rep["doc_id"], d_rep["seg_idx"])]
                    same_committee = d_span.get("committee", "") == span.get("committee", "")
                    print(f"    [{r + 1}] sim={sims[ci]:.4f} same_committee={same_committee} "
                          f"topic={journal_topic[ci][:60]}")
                    print(f"        snippet: {snip(d_span['span_text'])}")
            print()

        if not is_repeat:
            journal_reps.append(idx)
            journal_topic.append(t)

    print(f"\n# rank distribution over 36 events: {dict(sorted(Counter(ranks).items()))}")


if __name__ == "__main__":
    main()
