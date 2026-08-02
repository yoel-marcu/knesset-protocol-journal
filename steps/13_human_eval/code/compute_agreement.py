"""Once all 3 raters have filled in outputs/annotation_packet.json, compute per-rater
means and inter-annotator agreement (Cohen's kappa, pairwise, and mean pairwise) for
both longitudinal_coherence and faithfulness."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
RATERS = ["tomer", "or", "yoel"]
DIMS = ["longitudinal_coherence", "faithfulness"]


def cohens_kappa(a, b, labels):
    a, b = np.array(a), np.array(b)
    n = len(a)
    po = np.mean(a == b)
    pe = sum(np.mean(a == lbl) * np.mean(b == lbl) for lbl in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    packet = json.load(open(OUT / "annotation_packet.json", encoding="utf-8"))

    for dim in DIMS:
        print(f"\n=== {dim} ===")
        scores = {r: [] for r in RATERS}
        missing = False
        for c in packet:
            for r in RATERS:
                v = c["scores"][r][dim]
                if v is None:
                    missing = True
                scores[r].append(v)

        if missing:
            print("  (not fully scored yet -- fill in outputs/annotation_packet.json first)")
            continue

        labels = sorted(set(v for r in RATERS for v in scores[r]))
        for r in RATERS:
            print(f"  {r}: mean={np.mean(scores[r]):.2f}  scores={scores[r]}")

        kappas = []
        for r1, r2 in combinations(RATERS, 2):
            k = cohens_kappa(scores[r1], scores[r2], labels)
            kappas.append(k)
            print(f"  kappa({r1}, {r2}) = {k:.3f}")
        print(f"  mean pairwise kappa = {np.mean(kappas):.3f}")


if __name__ == "__main__":
    main()
