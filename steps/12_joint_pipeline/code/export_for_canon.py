"""
Step 12 — export OUR gold segments (Step 7) for canonicalization on Colab, so we
can fairly compare Step-10 linking on RAW vs CANONICAL text of the SAME segments.

Our gold segments are coarse (up to ~5000 words) vs Tomer's fine ones, but e5 only
sees the first ~512 tokens of raw text anyway -- so a clean ~600-word canonical
distillation of the whole segment can actually help MORE on our data. We cap the
raw input fed to Gemma at CAP_WORDS (context/speed), noting the tail is dropped for
the few very long segments.

Output: outputs/canon_input.json -- list of {seg_key, doc_id, seg_idx, date, raw}
        (seg_key = "{doc_id}__{seg_idx}"; `raw` is the capped span_text; the gold
        `topic` is deliberately NOT included so it cannot leak into the canonical).
Usage (CPU): python steps/12_joint_pipeline/code/export_for_canon.py
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
GOLD = STEP_DIR.parent / "07_gold_eval" / "outputs" / "gold_segments.json"
CAP_WORDS = 3500


def main():
    segs = json.load(open(GOLD, encoding="utf-8"))
    rows = []
    for s in segs:
        if not s.get("clean_for_eval"):
            continue
        raw = " ".join(s["span_text"].split()[:CAP_WORDS])
        rows.append({"seg_key": f"{s['doc_id']}__{s['seg_idx']}",
                     "doc_id": s["doc_id"], "seg_idx": s["seg_idx"],
                     "date": s["date"][:10], "raw": raw})
    json.dump(rows, open(OUT / "canon_input.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    capped = sum(1 for s in segs if len(s["span_text"].split()) > CAP_WORDS)
    log.info("Exported %d segments (%d exceeded %d-word cap, tail dropped) -> %s",
             len(rows), capped, CAP_WORDS, OUT / "canon_input.json")


if __name__ == "__main__":
    main()
