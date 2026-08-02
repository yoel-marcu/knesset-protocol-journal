"""
Step 12 — salvage the 145-entry partial Colab canonicalization run.

Root cause found (not truncation): Gemma wrapped its JSON response in a
```json ... ``` markdown fence; Tomer's _first_canon_json extractor failed to
strip it, fell back to dumping the raw text into `canonical` with `subject`
left empty (90/145 = 62% of the partial run hit this). The JSON itself is
valid once the fence is stripped -- confirmed recoverable, not lost work.

Usage (CPU): python steps/12_joint_pipeline/code/salvage_partial_canon.py
"""
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def salvage(entry):
    """Return a fixed entry dict, or None if genuinely unrecoverable."""
    if entry.get("subject", "").strip():
        return entry  # already fine
    raw = entry.get("canonical", "")
    inner = FENCE_RE.sub("", raw.strip())
    # try direct parse, then brace-balanced substring extraction as fallback
    candidates = [inner]
    m = re.search(r"\{.*\}", inner, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            d = json.loads(cand)
            if "canonical" in d and d["canonical"].strip():
                return {"seg_key": entry["seg_key"], "date": entry["date"],
                        "subject": d.get("subject", ""), "canonical": d["canonical"],
                        "decision": d.get("decision", ""), "amounts": d.get("amounts", [])}
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def main():
    done = json.load(open(OUT / "canonical_gold.json", encoding="utf-8"))
    fixed, unrecoverable = [], []
    n_was_broken = sum(1 for d in done if not d.get("subject", "").strip())
    for d in done:
        r = salvage(d)
        if r is not None:
            fixed.append(r)
        else:
            unrecoverable.append(d["seg_key"])

    log.info("Partial run: %d entries, %d were broken (empty subject)", len(done), n_was_broken)
    log.info("Salvaged: %d fixed, %d genuinely unrecoverable", len(fixed) - (len(done) - n_was_broken), len(unrecoverable))
    if unrecoverable:
        log.info("Unrecoverable seg_keys (will be redone): %s", unrecoverable[:10])

    json.dump(fixed, open(OUT / "canonical_gold_salvaged.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log.info("Wrote %s (%d entries)", OUT / "canonical_gold_salvaged.json", len(fixed))


if __name__ == "__main__":
    main()
