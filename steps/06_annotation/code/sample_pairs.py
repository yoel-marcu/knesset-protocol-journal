"""
Step 06 — Generate the annotation batch for human evaluation.

Samples a stratified set of span pairs across three strata:

  positive   — same gold topic (should be SAME in the journal).
               Up to MAX_POS_PER_TOPIC pairs sampled per topic; large topics
               (15x, 12x, ...) are capped so no single topic dominates.

  hard_neg   — different gold topics, highest e5-k1 cosine similarity.
               These are the pairs most likely to fool a threshold baseline
               and most important to cover in evaluation.

  easy_neg   — different gold topics, random low-similarity pairs.
               Provide a realistic prior for the NEW class.

Output: outputs/annotation_batch.json   — canonical pair list for all annotators.
"""

import json
import random
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP02_OUT = STEP_DIR.parent / "02_topic_clustering" / "outputs"
STEP03_OUT = STEP_DIR.parent / "03_dense_deanisotrize" / "outputs"
EMB_DIR = STEP02_OUT / "embeddings"
SPANS_FILE = STEP02_OUT / "topic_spans.json"
PROTOCOLS_DIR = STEP_DIR.parents[1] / "PROTOCOLS"

MAX_POS_PER_TOPIC = 3   # max positive pairs sampled from any single topic
N_HARD_NEG       = 30   # hard negatives (high sim, different topic)
N_EASY_NEG       = 20   # easy negatives (random low sim, different topic)
RANDOM_SEED      = 42


def load_data():
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    spans_lookup = {
        (_s["doc_id"], _s["topic"]): _s["span_text"]
        for _s in json.load(open(SPANS_FILE, encoding="utf-8"))
        if _s["clean_for_eval"]
    }
    dates = {}
    for r in ids:
        p = PROTOCOLS_DIR / f"{r['doc_id']}.json"
        if p.exists():
            dates[r["doc_id"]] = json.load(open(p, encoding="utf-8"))["date"][:10]
    return ids, spans_lookup, dates


def make_span_record(r, spans_lookup, dates, preview_chars=300):
    text = spans_lookup.get((r["doc_id"], r["topic"]), "")
    return {
        "doc_id":  r["doc_id"],
        "topic":   r["topic"],
        "date":    dates.get(r["doc_id"], ""),
        "preview": text[:preview_chars].strip(),
    }


def abtt(X, k):
    from sklearn.preprocessing import normalize
    if k == 0:
        return normalize(X.copy()).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V = Vt[:k]
    return normalize(X_c - (X_c @ V.T) @ V).astype(np.float32)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUT.mkdir(exist_ok=True)

    ids, spans_lookup, dates = load_data()
    topics = [r["topic"] for r in ids]
    counts = Counter(topics)

    # Build e5-k1 similarity matrix for hard-negative selection
    e5_raw = np.load(EMB_DIR / "e5.npy")
    X_e5 = abtt(e5_raw, 1)
    S = X_e5 @ X_e5.T  # (198, 198)

    # Group span indices by topic
    by_topic = defaultdict(list)
    for i, r in enumerate(ids):
        by_topic[r["topic"]].append(i)

    recurring_topics = [t for t, c in counts.items() if c >= 2]
    pairs = []
    pair_set = set()   # (min_i, max_i) dedup

    def add_pair(i, j, stratum):
        key = (min(i, j), max(i, j))
        if key in pair_set:
            return
        pair_set.add(key)
        pairs.append({
            "id":      f"pair_{len(pairs)+1:04d}",
            "stratum": stratum,
            "a":       make_span_record(ids[i], spans_lookup, dates),
            "b":       make_span_record(ids[j], spans_lookup, dates),
        })

    # ---- POSITIVES ------------------------------------------------
    for topic in recurring_topics:
        idxs = by_topic[topic]
        # enumerate all ordered pairs, shuffle, cap
        topic_pairs = [(idxs[a], idxs[b])
                       for a in range(len(idxs))
                       for b in range(a + 1, len(idxs))]
        random.shuffle(topic_pairs)
        for i, j in topic_pairs[:MAX_POS_PER_TOPIC]:
            add_pair(i, j, "positive")

    # ---- HARD NEGATIVES  (different topic, high cosine) ------------
    # All cross-topic pairs sorted by descending similarity
    n = len(ids)
    iu = np.triu_indices(n, k=1)
    sims_all = S[iu]
    cross_mask = np.array([topics[iu[0][k]] != topics[iu[1][k]] for k in range(len(sims_all))])
    cross_order = np.argsort(sims_all * cross_mask)[::-1]   # highest sim first, cross-topic only

    hard_added = 0
    for k in cross_order:
        if not cross_mask[k]:
            continue
        i, j = int(iu[0][k]), int(iu[1][k])
        add_pair(i, j, "hard_neg")
        hard_added += 1
        if hard_added >= N_HARD_NEG:
            break

    # ---- EASY NEGATIVES  (different topic, random low sim) ---------
    low_sim_threshold = float(np.percentile(sims_all[cross_mask], 20))
    low_candidates = [
        (int(iu[0][k]), int(iu[1][k]))
        for k in range(len(sims_all))
        if cross_mask[k] and sims_all[k] <= low_sim_threshold
    ]
    random.shuffle(low_candidates)
    for i, j in low_candidates[:N_EASY_NEG * 3]:   # try 3× to account for dedup
        add_pair(i, j, "easy_neg")
        if sum(1 for p in pairs if p["stratum"] == "easy_neg") >= N_EASY_NEG:
            break

    # Shuffle final order so annotators don't see all positives first
    random.shuffle(pairs)

    batch = {
        "description": "Annotation batch for Step 06 — cross-meeting subject linking.",
        "labels":      ["same", "related", "new"],
        "label_guide": {
            "same":    "Same legislative matter — should be a single journal entry.",
            "related": "Related but distinct matter — separate entries, may be cross-referenced.",
            "new":     "Clearly different subject — no meaningful connection."
        },
        "stats": {
            "total":    len(pairs),
            "positive": sum(1 for p in pairs if p["stratum"] == "positive"),
            "hard_neg": sum(1 for p in pairs if p["stratum"] == "hard_neg"),
            "easy_neg": sum(1 for p in pairs if p["stratum"] == "easy_neg"),
        },
        "pairs": pairs,
    }

    out_path = OUT / "annotation_batch.json"
    json.dump(batch, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")
    print(f"  Total pairs : {batch['stats']['total']}")
    print(f"  Positive    : {batch['stats']['positive']}")
    print(f"  Hard neg    : {batch['stats']['hard_neg']}")
    print(f"  Easy neg    : {batch['stats']['easy_neg']}")


if __name__ == "__main__":
    main()
