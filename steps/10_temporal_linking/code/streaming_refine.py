"""
Step 10 (refinement) — sharpen the winning geometric config from streaming_eval.py.

Three additions:
  1. 2-D gate: distinctiveness margin AND a magnitude floor (best cosine >= floor).
  2. DP without the size prior log(n_k) -- the size prior favors the big generic
     "budget-surplus" cluster, which is exactly the false-merge confound; test the
     pure adaptive-new-option variant.
  3. Finer whitening-dimension sweep around the winner.

CPU-only, reuses streaming_eval's transform + stream_records.
Usage:  python steps/10_temporal_linking/code/streaming_refine.py
"""

import json
import logging
from pathlib import Path

import numpy as np

import streaming_eval as se

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"


def gate_2d(records, floors, margins):
    """LINK to best-cosine entry iff best>=floor AND (best-second)>=margin."""
    best = None
    for floor in floors:
        for m in margins:
            tp = fp = fn = tn = 0
            for r in records:
                c = r["cos"]; k = int(np.argmax(c))
                srt = np.partition(c, -2) if len(c) >= 2 else np.array([c[k], -1])
                link = (srt[-1] >= floor) and (srt[-1] - srt[-2] >= m)
                if link:
                    if r["gold_is_repeat"] and k == r["gold_entry"]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
            row = {"floor": round(float(floor), 3), "margin": round(float(m), 3),
                   **se.confusion(tp, fp, fn, tn)}
            if best is None or row["f1"] > best["f1"]:
                best = row
    # also best subject to fmr<=5% found by re-scanning
    fmr5 = None
    for floor in floors:
        for m in margins:
            tp = fp = fn = tn = 0
            for r in records:
                c = r["cos"]; k = int(np.argmax(c))
                srt = np.partition(c, -2) if len(c) >= 2 else np.array([c[k], -1])
                link = (srt[-1] >= floor) and (srt[-1] - srt[-2] >= m)
                if link:
                    if r["gold_is_repeat"] and k == r["gold_entry"]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
            fmr = fp / (fp + tn) if fp + tn else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            if fmr <= 0.05 and (fmr5 is None or rec > fmr5["recall"]):
                fmr5 = {"floor": round(float(floor), 3), "margin": round(float(m), 3),
                        **se.confusion(tp, fp, fn, tn)}
    return {"best_f1": best, "best_f1_fmr<=5%": fmr5}


def sweep_dp_nosize(records, sigma2, sigma0_2):
    """DP adaptive-new WITHOUT the log(n_k) size prior."""
    rows = []
    for new_bias in np.linspace(-6, 6, 101):
        tp = fp = fn = tn = 0
        for r in records:
            dk_sq = 2 - 2 * r["cos"]
            link_score = -dk_sq / (2 * sigma2)          # no log n_k
            k = int(np.argmax(link_score))
            new_score = new_bias - r["d0_sq"] / (2 * sigma0_2)
            link = link_score[k] > new_score
            if link:
                if r["gold_is_repeat"] and k == r["gold_entry"]:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
        rows.append({"new_bias": round(float(new_bias), 3), **se.confusion(tp, fp, fn, tn)})
    return rows


def main():
    e5 = np.load(STEP07_OUT / "embeddings" / "e5.npy")
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    results = {}

    floors = np.linspace(0.0, 0.9, 19)
    margins = np.linspace(0.0, 0.5, 26)

    # 2-D gate on the top contenders
    log.info("=== 2-D gate (magnitude floor x margin) ===")
    for tf in ["whiten50", "abtt1", "abtt2"]:
        Xt = se.transform(e5, tf)
        for rep in ["median", "centroid"]:
            recs = se.stream_records(Xt, ids, rep)
            s = gate_2d(recs, floors, margins)
            results[f"gate2d|{tf}|{rep}"] = s
            b = s["best_f1"]; f = s["best_f1_fmr<=5%"]
            log.info("  %-9s %-9s best_f1=%.3f (P=%.3f R=%.3f @floor=%.2f,m=%.2f)  fmr5_R=%s",
                     tf, rep, b["f1"], b["precision"], b["recall"], b["floor"], b["margin"],
                     f'{f["recall"]:.3f}' if f else "n/a")

    # DP without size prior
    log.info("=== DP without size prior ===")
    for tf in ["whiten50", "abtt1"]:
        Xt = se.transform(e5, tf)
        recs = se.stream_records(Xt, ids, "median")
        sig2, sig0 = se.estimate_sigmas(Xt, ids)
        s = se.summarize(sweep_dp_nosize(recs, sig2, sig0))
        results[f"dp_nosize|{tf}|median"] = s
        b = s["best_f1"]
        log.info("  %-9s best_f1=%.3f (P=%.3f R=%.3f)", tf, b["f1"], b["precision"], b["recall"])

    # Finer whitening dim sweep with median+margin
    log.info("=== whitening dim sweep (median + margin) ===")
    for d in [20, 30, 40, 60, 80, 100]:
        Xt = se.transform(e5, f"whiten{d}")
        recs = se.stream_records(Xt, ids, "median")
        s = se.summarize(se.sweep_margin(recs))
        results[f"whitendim|whiten{d}|median|margin"] = s
        b = s["best_f1"]
        log.info("  whiten%-3d best_f1=%.3f (P=%.3f R=%.3f)", d, b["f1"], b["precision"], b["recall"])

    json.dump(results, open(OUT / "streaming_refine_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "streaming_refine_results.json")


if __name__ == "__main__":
    main()
