"""
Step 06 — Merge annotator files and compute inter-annotator agreement.

Loads all outputs/annotations_*.json files and produces:
  outputs/gold_labels.json   — adjudicated gold label per pair
  outputs/agreement_stats.json — Cohen's κ, pairwise agreement, confusion matrix

Adjudication rule:
  - Majority vote (≥2/3 annotators agree) → gold label.
  - All disagree (no majority) → flagged as "conflict"; left for manual resolution.

Usage:
    python adjudicate.py
    python adjudicate.py --resolve   # interactive resolution of conflicts
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

try:
    from bidi.algorithm import get_display as rtl
except ImportError:
    def rtl(text): return text

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
BATCH_FILE = OUT / "annotation_batch.json"

LABELS = ["same", "related", "new", "skip"]


def load_all_annotations() -> dict[str, dict]:
    """Returns {annotator_name: {pair_id: label}}."""
    result = {}
    for f in sorted(OUT.glob("annotations_*.json")):
        data = json.load(open(f, encoding="utf-8"))
        name = data["annotator"]
        result[name] = {pid: v["label"] for pid, v in data["pairs"].items()}
    return result


def cohen_kappa(a_labels: list, b_labels: list) -> float:
    """Cohen's κ between two equal-length label lists."""
    label_set = sorted(set(a_labels) | set(b_labels))
    n = len(a_labels)
    if n == 0:
        return float("nan")
    p_o = sum(a == b for a, b in zip(a_labels, b_labels)) / n
    count_a = Counter(a_labels)
    count_b = Counter(b_labels)
    p_e = sum((count_a[l] / n) * (count_b[l] / n) for l in label_set)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def main(resolve: bool = False):
    batch = json.load(open(BATCH_FILE, encoding="utf-8"))
    pair_ids = [p["id"] for p in batch["pairs"]]

    annotations = load_all_annotations()
    if not annotations:
        print("No annotation files found in outputs/. Run annotate.py first.")
        return

    annotators = list(annotations.keys())
    print(f"Annotators: {annotators}")
    for name, ann in annotations.items():
        print(f"  {name}: {len(ann)} pairs annotated")

    # --- Pairwise κ ---
    print("\n── Pairwise Cohen's κ ─────────────────────────────────────")
    for a, b in combinations(annotators, 2):
        shared = [pid for pid in pair_ids if pid in annotations[a] and pid in annotations[b]]
        if not shared:
            print(f"  {a} vs {b}: no shared pairs yet")
            continue
        la = [annotations[a][pid] for pid in shared]
        lb = [annotations[b][pid] for pid in shared]
        kappa = cohen_kappa(la, lb)
        agree  = sum(x == y for x, y in zip(la, lb)) / len(shared)
        print(f"  {a} vs {b}: κ={kappa:.3f}  raw_agree={agree:.1%}  (n={len(shared)})")

    # --- Adjudication (majority vote) ---
    gold = {}
    conflicts = []
    for pid in pair_ids:
        votes = {name: annotations[name][pid]
                 for name in annotators if pid in annotations[name]}
        if not votes:
            continue
        count = Counter(votes.values())
        top_label, top_count = count.most_common(1)[0]
        if top_count >= 2 or len(votes) == 1:
            gold[pid] = {"gold": top_label, "votes": votes, "status": "agreed"}
        else:
            gold[pid] = {"gold": None, "votes": votes, "status": "conflict"}
            conflicts.append(pid)

    agreed   = sum(1 for v in gold.values() if v["status"] == "agreed")
    conflict = len(conflicts)
    print(f"\n── Adjudication ───────────────────────────────────────────")
    print(f"  Agreed   : {agreed}")
    print(f"  Conflict : {conflict}")

    label_dist = Counter(v["gold"] for v in gold.values() if v["gold"])
    print(f"  Gold distribution: {dict(label_dist)}")

    # --- Resolve conflicts interactively ---
    if resolve and conflicts:
        print(f"\n── Resolving {len(conflicts)} conflicts ────────────────────────")
        pair_lookup = {p["id"]: p for p in batch["pairs"]}
        for pid in conflicts:
            pair = pair_lookup[pid]
            votes = gold[pid]["votes"]
            print(f"\n  {pid}  [{pair['stratum']}]")
            print(f"  A: {rtl(pair['a']['topic'][:70])}  ({pair['a']['date']})")
            print(f"  B: {rtl(pair['b']['topic'][:70])}  ({pair['b']['date']})")
            print(f"  Votes: {votes}")
            while True:
                raw = input("  Resolved label [s/r/n/skip]: ").strip().lower()
                label_map = {"s": "same", "r": "related", "n": "new", "skip": "skip"}
                if raw in label_map:
                    gold[pid]["gold"] = label_map[raw]
                    gold[pid]["status"] = "resolved"
                    break
                print("  Use s/r/n/skip")

    # --- Save outputs ---
    out_gold = OUT / "gold_labels.json"
    json.dump({"annotators": annotators, "labels": gold},
              open(out_gold, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nWrote {out_gold}  ({len(gold)} pairs)")

    stats = {
        "annotators": annotators,
        "total_adjudicated": len(gold),
        "agreed": agreed,
        "conflicts": conflict,
        "gold_distribution": dict(label_dist),
    }
    json.dump(stats, open(OUT / "agreement_stats.json", "w", encoding="utf-8"), indent=2)
    print(f"Wrote {OUT / 'agreement_stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolve", action="store_true",
                        help="Interactively resolve annotation conflicts")
    args = parser.parse_args()
    main(args.resolve)
