"""
Step 08b — Build verifier queries: one record per streaming decision, with its
top-K (=10) e5_k1-retrieved candidates, ready to hand to an LLM.

Deliberately does NOT expose the annotator's gold topic label to the model, only
raw span-text snippets (+ date/committee metadata). The gold label was picked by
annotators reusing exact prior label strings -- showing that text as a
"candidate title" would let the model win by string-matching the answer instead
of judging topical identity from content, which is not available in a real
pipeline (an upstream extractor would title each segment independently, not
copy a prior title). Content-only is the harder, realistic test.

Journal growth is "oracle-growth" (matches shortlist_recall.py): a new journal
entry is created on a topic's true first occurrence, isolating the verifier's
judgment quality from upstream decision-threshold effects.

Input:  steps/07_gold_eval/outputs/embeddings/{e5}.npy + ids.json
        steps/07_gold_eval/outputs/gold_segments.json  (for span_text snippets)
Output: outputs/verifier_queries.json
Usage (CPU-only):
    python steps/08_retrieve_verify/code/build_verifier_inputs.py
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
SPANS_FILE = STEP07_OUT / "gold_segments.json"

E5_K = 1
TOP_K = 10
SNIPPET_WORDS = 100


def abtt(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


def snippet(text: str, n: int = SNIPPET_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])


def main():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    order_key = [(r["date"], r["doc_id"], r["seg_idx"]) for r in ids]
    chron_order = np.array(sorted(range(len(ids)), key=lambda i: order_key[i]))

    span_lookup = {
        (s["doc_id"], s["seg_idx"]): s
        for s in json.load(open(SPANS_FILE, encoding="utf-8"))
        if s["clean_for_eval"]
    }

    e5_raw = np.load(EMB_DIR / "e5.npy")
    X = abtt(e5_raw, E5_K)
    S = (X @ X.T).astype(np.float64)

    first_seen = {}
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    journal_reps: list[int] = []
    journal_topic: list[str] = []
    queries = []

    for idx in chron_order:
        t = topics[idx]
        is_repeat = (first_seen[t] != idx)
        rec = ids[idx]
        span = span_lookup[(rec["doc_id"], rec["seg_idx"])]

        if journal_reps:
            sims = np.array([S[idx, rep] for rep in journal_reps])
            order = np.argsort(-sims)[:TOP_K]  # top-K candidate journal-entry indices

            correct_candidate_rank = None
            if is_repeat:
                correct_entry = journal_topic.index(t)
                pos = np.where(order == correct_entry)[0]
                correct_candidate_rank = int(pos[0]) + 1 if len(pos) else None  # None if outside top-K

            candidates = []
            for rank, entry_i in enumerate(order, start=1):
                rep_idx = journal_reps[entry_i]
                rep_rec = ids[rep_idx]
                rep_span = span_lookup[(rep_rec["doc_id"], rep_rec["seg_idx"])]
                candidates.append({
                    "rank": rank,
                    "doc_id": rep_rec["doc_id"], "seg_idx": rep_rec["seg_idx"],
                    "date": rep_rec["date"],
                    "committee": rep_span.get("committee", ""),
                    "snippet": snippet(rep_span["span_text"]),
                    "similarity": round(float(sims[entry_i]), 4),
                    "_gold_topic": journal_topic[entry_i],  # eval-only, never shown to model
                })

            queries.append({
                "query_doc_id": rec["doc_id"], "query_seg_idx": rec["seg_idx"],
                "query_date": rec["date"], "query_committee": span.get("committee", ""),
                "query_snippet": snippet(span["span_text"]),
                "candidates": candidates,
                "gold_is_repeat": bool(is_repeat),
                "gold_correct_candidate_rank": correct_candidate_rank,
                "_gold_topic": t,  # eval-only
            })

        if not is_repeat:
            journal_reps.append(idx)
            journal_topic.append(t)

    n_repeat = sum(q["gold_is_repeat"] for q in queries)
    n_recoverable = sum(1 for q in queries if q["gold_correct_candidate_rank"] is not None)
    log.info("Built %d verifier queries (%d gold repeats, %d with true match in top-%d)",
             len(queries), n_repeat, n_recoverable, TOP_K)

    json.dump(queries, open(OUT / "verifier_queries.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "verifier_queries.json")


if __name__ == "__main__":
    main()
