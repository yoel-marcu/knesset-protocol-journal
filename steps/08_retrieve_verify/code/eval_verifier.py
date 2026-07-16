"""
Step 08d — Evaluate verifier predictions against gold, sweeping the model's
same_confidence score as a threshold (theta), exactly like Step 07's cosine-
threshold streaming_eval, so results are directly comparable to the e5_k1
baseline curve (best F1=0.125 @ P=0.077 R=0.333) instead of scored at one
fixed, uncalibrated operating point.

For each theta in THETA_SWEEP: predict LINK to best_matched_rank if
same_confidence >= theta (and best_matched_rank is not null), else NEW.
  TP: predicted LINK AND best_matched_rank == the correct candidate's rank
  FP: predicted LINK AND (wrong candidate OR gold_is_repeat is False)  -- false merge
  FN: predicted NEW/no-link AND gold_is_repeat is True                -- false split
  TN: predicted NEW/no-link AND gold_is_repeat is False

Usage:
    python steps/08_retrieve_verify/code/eval_verifier.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"

BASELINE = {"precision": 0.077, "recall": 0.333, "f1": 0.125}  # Step 07 e5_k1 best-F1 point
THETA_SWEEP = list(range(0, 101, 5))


def eval_sweep(queries: list[dict], preds: list[dict]) -> dict:
    assert len(queries) == len(preds)
    rows = []
    for theta in THETA_SWEEP:
        tp = fp = fn = tn = 0
        for q, p in zip(queries, preds):
            gold_link = q["gold_is_repeat"]
            rank = p.get("best_matched_rank")
            # two_score predictions use specific_match_confidence instead of same_confidence
            conf = p.get("same_confidence", p.get("specific_match_confidence", 0))
            predict_link = (rank is not None) and (conf >= theta)

            if predict_link:
                if gold_link and rank == q["gold_correct_candidate_rank"]:
                    tp += 1
                else:
                    fp += 1
            else:
                if gold_link:
                    fn += 1
                else:
                    tn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append({"theta": theta, "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "false_merge_rate": round(fmr, 4),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    best_f1 = max(rows, key=lambda r: r["f1"])
    prec_priority = max(
        (r for r in rows if r["false_merge_rate"] <= 0.05),
        key=lambda r: r["recall"], default=rows[0])
    return {"curve": rows, "best_f1": best_f1, "prec_priority_theta": prec_priority}


def main():
    queries = json.load(open(OUT / "verifier_queries.json", encoding="utf-8"))
    pred_files = sorted(OUT.glob("verifier_predictions_*.json"))
    if not pred_files:
        log.warning("No verifier_predictions_*.json found in %s -- run run_verifier.py first", OUT)
        return

    results = {}
    for f in pred_files:
        name = f.stem.replace("verifier_predictions_", "")
        preds = json.load(open(f, encoding="utf-8"))
        n_parse_err = sum(1 for p in preds if p.get("_parse_error"))
        sweep = eval_sweep(queries, preds)
        results[name] = sweep
        bf = sweep["best_f1"]
        pp = sweep["prec_priority_theta"]
        log.info("%-15s  best_F1=%.3f@theta=%d (P=%.3f R=%.3f)  "
                 "prec_priority F1=%.3f@theta=%d (FMR<=5%%)  parse_errors=%d/%d",
                 name, bf["f1"], bf["theta"], bf["precision"], bf["recall"],
                 pp["f1"], pp["theta"], n_parse_err, len(queries))

    log.info("Step 07 threshold baseline (e5_k1, best-F1 point): P=%.3f R=%.3f F1=%.3f",
             BASELINE["precision"], BASELINE["recall"], BASELINE["f1"])

    json.dump({"models": results, "threshold_baseline": BASELINE},
              open(OUT / "verifier_eval_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "verifier_eval_summary.json")


if __name__ == "__main__":
    main()
