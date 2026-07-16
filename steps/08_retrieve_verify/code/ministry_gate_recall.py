"""
Step 08h — Test exact-ministry-match as a rerank gate on top of the e5_k1
top-10 shortlist, instead of TF-IDF (which failed -- see rerank_recall.py --
because shared administrative *register* vocabulary outweighs the sparse
true-topic overlap).

Israeli government ministries/authorities are a small, closed, known set --
this is dictionary lookup, not open-ended NER. List curated from a frequency
scan of gold_segments.json (see conversation); canonicalizes spelling variants
(e.g. רשות המיסים / רשות המסים).

score = dense_sim + GATE * ministry_overlap(query, candidate)
Sweeps GATE and reports recall@K on the 36 gold repeat events, same protocol
as rerank_recall.py, for direct comparison.

Usage (CPU-only):
    python steps/08_retrieve_verify/code/ministry_gate_recall.py
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
GATE_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]

# Canonical ministry/authority names (spelling variants merged onto one key)
MINISTRIES = [
    "משרד האוצר", "רשות המיסים", "רשות המסים", "משרד המשפטים", "משרד החינוך",
    "משרד הפנים", "משרד הבריאות", "משרד הכלכלה", "משרד הביטחון", "משרד הרווחה",
    "משרד השיכון", "משרד הבינוי והשיכון", "משרד החקלאות", "משרד התחבורה",
    "רשות התחרות", "המשרד להגנת הסביבה", "המשרד לביטחון לאומי",
    "רשות החברות הממשלתיות", "רשות החברות", "משרד התיירות", "משרד האנרגיה",
    "המשרד לביטחון פנים", "משרד הקליטה", "משרד העלייה והקליטה",
    "רשות האוכלוסין וההגירה", "רשות האוכלוסין", "משרד התרבות והספורט",
    "רשות החדשנות", "רשות הדואר", "המשרד לשירותי דת", "הרשות לניירות ערך",
    "המשרד לשוויון חברתי", "רשות החשמל", "משרד העבודה",
    "הרשות לזכויות ניצולי שואה", "הרשות לניצולי שואה", "משרד התקשורת",
    "משרד החוץ", "רשות מקרקעי ישראל", 'רמ"י', "בנק ישראל",
]
# canonicalize spelling variants to one bucket
CANON = {
    "רשות המסים": "רשות המיסים",
    "משרד הבינוי והשיכון": "משרד השיכון",
    "משרד העלייה והקליטה": "משרד הקליטה",
    "רשות האוכלוסין": "רשות האוכלוסין וההגירה",
    "רשות החברות": "רשות החברות הממשלתיות",
    "הרשות לניצולי שואה": "הרשות לזכויות ניצולי שואה",
    'רמ"י': "רשות מקרקעי ישראל",
}


def extract_ministries(text: str) -> set:
    found = set()
    for m in MINISTRIES:
        if m in text:
            found.add(CANON.get(m, m))
    return found


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
    ids = json.load(open(EMB_DIR / "ids.json", encoding="utf-8"))
    topics = np.array([r["topic"] for r in ids])
    order_key = [(r["date"], r["doc_id"], r["seg_idx"]) for r in ids]
    chron_order = np.array(sorted(range(len(ids)), key=lambda i: order_key[i]))

    span_lookup = {(s["doc_id"], s["seg_idx"]): s["span_text"]
                   for s in json.load(open(SPANS_FILE, encoding="utf-8")) if s["clean_for_eval"]}
    ministries_by_idx = [extract_ministries(span_lookup[(r["doc_id"], r["seg_idx"])]) for r in ids]

    n_with_ministry = sum(1 for m in ministries_by_idx if m)
    print(f"Segments with >=1 detected ministry: {n_with_ministry}/{len(ids)} "
          f"({100 * n_with_ministry / len(ids):.1f}%)\n")

    X = abtt(np.load(EMB_DIR / "e5.npy"), E5_K)
    S_dense = (X @ X.T).astype(np.float64)

    first_seen = {}
    for idx in chron_order:
        t = topics[idx]
        if t not in first_seen:
            first_seen[t] = idx

    journal_reps, journal_topic = [], []
    events = []

    for idx in chron_order:
        t = topics[idx]
        is_repeat = (first_seen[t] != idx)
        if journal_reps and is_repeat:
            dense_sims = np.array([S_dense[idx, rep] for rep in journal_reps])
            top10_local = np.argsort(-dense_sims)[:TOP_K]
            correct_entry = journal_topic.index(t)
            pos_in_top10 = np.where(top10_local == correct_entry)[0]
            if len(pos_in_top10):
                query_min = ministries_by_idx[idx]
                overlap = np.array([
                    1.0 if (query_min & ministries_by_idx[journal_reps[ci]]) else 0.0
                    for ci in top10_local
                ])
                events.append({
                    "dense_sims": dense_sims[top10_local],
                    "overlap": overlap,
                    "correct_local_rank": int(pos_in_top10[0]) + 1,
                    "query_has_ministry": bool(query_min),
                })
        if not is_repeat:
            journal_reps.append(idx)
            journal_topic.append(t)

    baseline_ranks = [e["correct_local_rank"] for e in events]
    n_query_no_ministry = sum(1 for e in events if not e["query_has_ministry"])
    print(f"Baseline (dense-only) rank distribution: {dict(sorted(Counter(baseline_ranks).items()))}")
    print(f"Baseline recall@1: {np.mean([r == 1 for r in baseline_ranks]):.3f}")
    print(f"Events where query has NO detected ministry (gate can't help): {n_query_no_ministry}/36\n")

    for gate in GATE_SWEEP:
        ranks = []
        for e in events:
            combined = e["dense_sims"] + gate * e["overlap"]
            order = np.argsort(-combined)
            correct_pos = e["correct_local_rank"] - 1
            new_rank = int(np.where(order == correct_pos)[0][0]) + 1
            ranks.append(new_rank)
        recall1 = np.mean([r == 1 for r in ranks])
        recall3 = np.mean([r <= 3 for r in ranks])
        print(f"gate={gate:<5} recall@1={recall1:.3f}  recall@3={recall3:.3f}  "
              f"rank_dist={dict(sorted(Counter(ranks).items()))}")


if __name__ == "__main__":
    main()
