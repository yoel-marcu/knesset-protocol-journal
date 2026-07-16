"""
Step 09a — Build journal chains: for each gold-recurring topic, the full text of
every occurrence in chronological order (input for incremental log-writing).

Built on gold_chains.json / gold_segments.json (Step 07) rather than any predicted
linking output -- Task 3 (log writing) is deliberately isolated from Task 2 (linking)
quality, same precedent as Step 07 isolating Task 1 from Task 2. Method (a)'s best-F1
precision is only 7.7%, so its predicted chains are mostly wrong; using them here
would conflate "is the summarizer bad" with "is the upstream linking bad."

Input:  steps/07_gold_eval/outputs/gold_chains.json   (22 recurring topics, by exact label)
        steps/07_gold_eval/outputs/gold_segments.json (full span_text per occurrence)
Output: outputs/journal_chains.json -- list of chains, each a chronological list of
        {doc_id, seg_idx, date, committee, span_text}
Usage (CPU-only):
    python steps/09_log_writing/code/build_journal_chains.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"


def main():
    chains = json.load(open(STEP07_OUT / "gold_chains.json", encoding="utf-8"))
    segments = json.load(open(STEP07_OUT / "gold_segments.json", encoding="utf-8"))
    span_lookup = {(s["doc_id"], s["seg_idx"]): s for s in segments}

    recurring = {topic: occs for topic, occs in chains.items() if len(occs) >= 2}
    log.info("Recurring topics (chain length >= 2): %d", len(recurring))

    journal_chains = []
    for topic, occs in recurring.items():
        occs_sorted = sorted(occs, key=lambda o: o["date"])
        chain = []
        for o in occs_sorted:
            s = span_lookup.get((o["doc_id"], o["seg_idx"]))
            if s is None:
                log.warning("Missing span for %s#%d, skipping occurrence", o["doc_id"], o["seg_idx"])
                continue
            chain.append({
                "doc_id": o["doc_id"], "seg_idx": o["seg_idx"], "date": o["date"],
                "committee": s.get("committee", ""), "span_text": s["span_text"],
            })
        if len(chain) >= 2:
            journal_chains.append({"topic": topic, "occurrences": chain})

    n_updates = sum(len(c["occurrences"]) - 1 for c in journal_chains)
    log.info("Built %d chains, %d total incremental-update instances",
             len(journal_chains), n_updates)

    json.dump(journal_chains, open(OUT / "journal_chains.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "journal_chains.json")


if __name__ == "__main__":
    main()
