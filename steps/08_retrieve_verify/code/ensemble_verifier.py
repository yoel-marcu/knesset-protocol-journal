"""
Step 08e — Ensemble the 3 existing verifier predictions (majority-vote on
best_matched_rank, averaged confidence among the agreeing models).

Motivated by Step 08's finding that recall is already strong per-model (70-95%)
but precision is the isolated bottleneck (too many false "same" calls riding
along) -- majority voting should filter out the false positives that only one
of the three models produces, at some recall cost. Pure post-hoc combination
of already-computed predictions; no new model calls, no GPU needed.

Rule per query: if >=2 of the 3 models agree on the same best_matched_rank
(and it's non-null), the ensemble picks that rank with confidence = mean of
the agreeing models' same_confidence. Otherwise (no majority), ensemble
predicts no-link (rank=null, confidence=0) -- consistent with the project's
asymmetric-cost rule (false-merge >> false-split): genuine disagreement between
models is treated as insufficient evidence to link.

Input:  outputs/verifier_predictions_{dictalm2,qwen3b,qwen7b}.json
Output: outputs/verifier_predictions_ensemble.json (same schema, so
        eval_verifier.py's glob picks it up automatically)
Usage (CPU-only):
    python steps/08_retrieve_verify/code/ensemble_verifier.py
"""

import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
MODELS = ["dictalm2", "qwen3b", "qwen7b"]


def main():
    preds = {m: json.load(open(OUT / f"verifier_predictions_{m}.json", encoding="utf-8"))
              for m in MODELS}
    n = len(preds[MODELS[0]])
    for m in MODELS:
        assert len(preds[m]) == n, f"{m} has {len(preds[m])} predictions, expected {n}"

    ensemble = []
    n_majority = 0
    for i in range(n):
        triple = [preds[m][i] for m in MODELS]
        ranks = [p.get("best_matched_rank") for p in triple]
        counts = Counter(r for r in ranks if r is not None)
        top_rank, top_count = counts.most_common(1)[0] if counts else (None, 0)

        if top_count >= 2:
            agreeing_conf = [p["same_confidence"] for p, r in zip(triple, ranks) if r == top_rank]
            ensemble.append({
                "best_matched_rank": top_rank,
                "same_confidence": round(sum(agreeing_conf) / len(agreeing_conf)),
                "_n_agree": top_count,
            })
            n_majority += 1
        else:
            ensemble.append({"best_matched_rank": None, "same_confidence": 0, "_n_agree": top_count})

    log.info("Majority (>=2/3) reached on %d/%d queries (%.1f%%)", n_majority, n, 100 * n_majority / n)

    out_path = OUT / "verifier_predictions_ensemble.json"
    json.dump(ensemble, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
