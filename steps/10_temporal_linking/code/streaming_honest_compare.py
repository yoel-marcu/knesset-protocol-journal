"""
Step 10 — which transforms survive streaming-honest fitting?

whiten50 collapsed 0.250 -> 0.094 when refit only on spans-seen-so-far (it needs a
50-dim covariance estimate that doesn't exist early in the stream). This checks the
CHEAPER transforms that need far fewer samples to estimate:
  center  -- just the running mean (estimable from step 2)
  abtt1   -- running mean + top-1 direction (estimable very early)
  abtt2   -- running mean + top-2 directions
all with centroid representation + the margin rule. Reports batch vs honest so we
can pick a flagship that does NOT leak future information.

Usage: python steps/10_temporal_linking/code/streaming_honest_compare.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

import streaming_eval as se

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"


def honest_centroid_records(e5, ids, kind):
    topics = [r["topic"] for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)
    entry_members, topic_to_entry = [], {}
    records = []
    for p, i in enumerate(order):
        seen = order[: p + 1]
        Xt = se.transform(e5[seen], kind)
        x = Xt[p]
        if entry_members:
            cos = np.empty(len(entry_members))
            for k, mem in enumerate(entry_members):
                v = Xt[mem].mean(0); v /= np.linalg.norm(v) + 1e-12
                cos[k] = v @ x
            t = topics[i]; is_rep = first_seen[t] != i
            records.append({"gold_is_repeat": bool(is_rep), "cos": cos,
                            "gold_entry": topic_to_entry.get(t, -1) if is_rep else -1})
        t = topics[i]
        if first_seen[t] == i:
            topic_to_entry[t] = len(entry_members); entry_members.append([p])
        else:
            entry_members[topic_to_entry[t]].append(p)
    return records


def main():
    e5 = np.load(STEP07_OUT / "embeddings" / "e5.npy")
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    out = {}
    for kind in ["center", "abtt1", "abtt2"]:
        batch = se.summarize(se.sweep_margin(se.stream_records(se.transform(e5, kind), ids, "centroid")))
        honest = se.summarize(se.sweep_margin(honest_centroid_records(e5, ids, kind)))
        out[kind] = {"batch": batch["best_f1"], "honest": honest["best_f1"],
                     "batch_fmr5": batch["best_f1_fmr<=5%"], "honest_fmr5": honest["best_f1_fmr<=5%"]}
        log.info("%-7s  batch F1=%.3f (R=%.3f) | honest F1=%.3f (R=%.3f, fmr5_R=%s)",
                 kind, batch["best_f1"]["f1"], batch["best_f1"]["recall"],
                 honest["best_f1"]["f1"], honest["best_f1"]["recall"],
                 f'{honest["best_f1_fmr<=5%"]["recall"]:.3f}' if honest["best_f1_fmr<=5%"] else "n/a")
    json.dump(out, open(OUT / "streaming_honest_compare.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "streaming_honest_compare.json")


if __name__ == "__main__":
    main()
