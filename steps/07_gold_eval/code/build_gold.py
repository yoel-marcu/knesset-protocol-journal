"""
Convert the manual segmentation annotations (Step 06) into gold evaluation data
for both coupled tasks:

  Task 1 (segmentation/topic labeling): gold_segments.json
    Same record shape as steps/02_topic_clustering/outputs/topic_spans.json,
    so it's a drop-in input for embed.py / cluster_eval.py.

  Task 2 (streaming subject linking): gold_chains.json
    Segments grouped by exact label string. A label that recurs across protocols
    IS the gold SAME chain (the annotator explicitly reused the label via the
    "seen elsewhere" picker); a label seen once is a gold NEW with no partner.
    No fuzzy canonicalization is applied -- near-duplicate label strings that
    should have been reused but weren't will surface as extra singleton chains
    (false-split noise). Accepted per project's own asymmetric-cost principle
    (false-split << false-merge).

Utterance parsing must exactly match steps/06_annotation/code/serve.py's
parse_protocol, since before_idx values in segmentation.json were generated
against that indexing -- so it is imported directly rather than reimplemented.

Usage:
    python steps/07_gold_eval/code/build_gold.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]          # ANLP-PROJECT
STEP_DIR = Path(__file__).resolve().parents[1]      # steps/07_gold_eval
PROTOCOLS_DIR = ROOT / "PROTOCOLS"
SEG_FILE = ROOT / "steps" / "06_annotation" / "outputs" / "segmentation.json"
OUTPUTS_DIR = STEP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_OUT = OUTPUTS_DIR / "gold_segments.json"
CHAINS_OUT = OUTPUTS_DIR / "gold_chains.json"

sys.path.insert(0, str(ROOT / "steps" / "06_annotation" / "code"))
from serve import parse_protocol  # noqa: E402


def clean(text: str) -> str:
    return " ".join(text.split())


def segment_protocol(doc: dict, saved: dict) -> list[dict]:
    """One record per annotated segment (mirrors topic_spans.json shape)."""
    _, utterances, _ = parse_protocol(doc.get("text", ""))

    segs = sorted(saved.get("segments", []), key=lambda s: s["before_idx"])
    n = len(utterances)
    meta = {
        "doc_id": doc["doc_id"],
        "committee": doc.get("committee", ""),
        "date": doc.get("date", ""),
    }

    records = []
    for i, seg in enumerate(segs):
        label = seg.get("label", "").strip()
        if not label:
            continue
        start = seg["before_idx"]
        end = segs[i + 1]["before_idx"] if i + 1 < len(segs) else n
        if start >= end:
            continue
        span_text = clean(" ".join(u["text"] for u in utterances[start:end]))
        records.append({
            **meta,
            "seg_idx": i,
            "topic": label,
            "span_text": span_text,
            "n_chars": len(span_text),
            "n_words": len(span_text.split()),
            "segmentation": "gold_manual",
            "auto_unconfirmed": bool(seg.get("auto", False)),
            "clean_for_eval": True,
        })
    return records


def main() -> None:
    seg_data = json.load(open(SEG_FILE, encoding="utf-8"))
    files = sorted(PROTOCOLS_DIR.glob("*.json"))
    log.info("Protocols on disk: %d, annotated: %d", len(files), len(seg_data))

    all_segments: list[dict] = []
    missing = []
    for f in files:
        doc_id = f.stem
        if doc_id not in seg_data:
            missing.append(doc_id)
            continue
        doc = json.load(open(f, encoding="utf-8"))
        all_segments.extend(segment_protocol(doc, seg_data[doc_id]))

    with open(SEGMENTS_OUT, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    # --- gold chains: group by exact label string ---
    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in all_segments:
        by_label[rec["topic"]].append(rec)
    chains = {}
    for label, members in by_label.items():
        ordered = sorted(members, key=lambda r: (r["date"], r["doc_id"], r["seg_idx"]))
        chains[label] = [
            {"doc_id": m["doc_id"], "seg_idx": m["seg_idx"], "date": m["date"]}
            for m in ordered
        ]
    with open(CHAINS_OUT, "w", encoding="utf-8") as f:
        json.dump(chains, f, ensure_ascii=False, indent=2)

    # --- summary ---
    n_unconfirmed_docs = sum(
        1 for doc_id, d in seg_data.items()
        if any(s.get("auto") for s in d.get("segments", []))
    )
    recurring = {l: m for l, m in chains.items() if len(m) > 1}
    log.info("Missing protocols (no annotation): %s", missing)
    log.info("Docs with >=1 unconfirmed (auto=true) segment: %d", n_unconfirmed_docs)
    log.info("Total gold segments: %d", len(all_segments))
    log.info("Unique labels: %d, recurring (chain len>1): %d", len(chains), len(recurring))
    log.info("Total SAME pairs implied (sum of chain_len-1): %d",
              sum(len(m) - 1 for m in recurring.values()))
    log.info("Wrote %s", SEGMENTS_OUT)
    log.info("Wrote %s", CHAINS_OUT)


if __name__ == "__main__":
    main()
