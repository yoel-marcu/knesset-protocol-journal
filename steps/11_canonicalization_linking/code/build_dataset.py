"""
Step 11 — build the raw-vs-canonical evaluation dataset from Tomer's NLP_ADV pipeline.

Tomer's research question: does canonicalizing each segment (rewriting it into neutral
3rd-person Hebrew, stripping the bureaucratic register) improve embedding quality for
topic assignment, vs. embedding the raw segment text? This directly bears on Step 10:
the boilerplate that canonicalization removes is exactly the "budget register" confound
that Step 10's whitening tried (and failed, on honesty grounds) to remove geometrically.

For every segment in NLP_ADV/data/canonicalizations/*.json, pairs:
  raw       = concatenated utterance text (via segmentation utterance_ids -> protocol)
  canonical = Tomer's canonical rewrite
  subject   = Tomer's LLM subject label (used as a pseudo-topic label for recurrence)

Output: outputs/canon_dataset.json -- list of
  {seg_id, file_id, date, subject, raw, canonical, is_procedural}
Usage (CPU): python steps/11_canonicalization_linking/code/build_dataset.py
"""

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
NLP_ADV = STEP_DIR.parents[1] / "NLP_ADV"
PROTO = NLP_ADV / "data" / "protocols"
SEG = NLP_ADV / "data" / "segmentations"
CANON = NLP_ADV / "data" / "canonicalizations"

# obviously-procedural subjects that are not journal "matters" (opening/closing/votes)
PROC_RE = re.compile(r"(פתיחת ישיבה|סיום ישיבת|נעילת ישיבה|הצבעה על|בקשת רביזיה|"
                     r"הסתייגות|רוויזי|סדר היום|הצבעה)")


def main():
    rows = []
    for cf in sorted(CANON.glob("*.canonical.json")):
        file_id = cf.name.replace(".canonical.json", "")
        canon = json.load(open(cf, encoding="utf-8"))
        seg_file = SEG / f"{file_id}.segments.json"
        proto_file = PROTO / f"{file_id}.json"
        if not seg_file.exists() or not proto_file.exists():
            log.warning("skip %s (missing segmentation or protocol)", file_id)
            continue
        seg = json.load(open(seg_file, encoding="utf-8"))
        proto = json.load(open(proto_file, encoding="utf-8"))
        utt = {str(u["id"]): u["text"] for u in proto["utterances"]}
        seg_by_id = {s["seg_id"]: s for s in seg["segments"]}

        for cs in canon["segments"]:
            sid = cs["seg_id"]
            s = seg_by_id.get(sid)
            if s is None:
                continue
            raw = " ".join(utt.get(str(x), "") for x in s["utterance_ids"]).strip()
            canonical = (cs.get("canonical") or "").strip()
            subject = re.sub(r"\s+", " ", (cs.get("subject") or "").strip())
            if not raw or not canonical or not subject:
                continue
            rows.append({
                "seg_id": sid, "file_id": file_id, "date": canon.get("date", "")[:10],
                "subject": subject, "raw": raw, "canonical": canonical,
                "is_procedural": bool(PROC_RE.search(subject)),
            })

    n_proc = sum(r["is_procedural"] for r in rows)
    log.info("Built %d segment records from %d protocols (%d procedural, %d substantive)",
             len(rows), len({r["file_id"] for r in rows}), n_proc, len(rows) - n_proc)
    json.dump(rows, open(OUT / "canon_dataset.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log.info("Wrote %s", OUT / "canon_dataset.json")


if __name__ == "__main__":
    main()
