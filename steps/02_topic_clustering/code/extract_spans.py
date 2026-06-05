"""
Stage 1 of the topic-clustering pipeline: body isolation + per-topic segmentation.

Produces clean, boilerplate-free discussion spans, one per (protocol, topic),
for downstream embedding and clustering.

Pipeline:
  1. Strip preamble  -- everything before the first speaker tag is header + agenda +
                        attendees boilerplate; drop it.
  2. Segment         -- split the body on body-internal topic markers (<< נושא >> / << הצח >>).
                        Single-topic protocols yield one span = whole body.
  3. Clean           -- remove speaker-name tag spans (<< יור >> NAME << יור >>), strip any
                        remaining markers, normalize whitespace. Speech content is preserved;
                        speaker identities are removed (they are constant across this committee
                        and would otherwise dominate the clustering signal).

Output: outputs/topic_spans.json -- list of records:
  {
    "doc_id", "committee", "date",
    "topic",            # canonical topic label (gold)
    "span_text",        # cleaned discussion text
    "n_chars", "n_words",
    "segmentation",     # "single" | "marker_split"
    "clean_for_eval"    # bool: False for front-loaded multi-topic (ambiguous unit)
  }

Usage:
    python src/clustering/extract_spans.py
"""

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]          # ANLP-PROJECT
STEP_DIR = Path(__file__).resolve().parents[1]      # steps/02_topic_clustering
PROTOCOLS_DIR = ROOT / "PROTOCOLS"
CANONICAL_FILE = ROOT / "steps" / "01_topic_preprocessing" / "outputs" / "topics_canonical.json"
OUTPUTS_DIR = STEP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUTS_DIR / "topic_spans.json"

SPEAKER_TAGS = ["דובר", "יור", "אורח", "קריאה", "דובר_המשך", "מנהל", "הלסי"]
TOPIC_TAGS = ["נושא", "הצח"]
STRUCT_TAGS = ["סיום", "הפסקה"]

# Generic tag matcher: << anything >>
ANY_TAG = re.compile(r"<<\s*([^>]+?)\s*>>")

# Speaker-name span: << TAG >> name : << TAG >>  (same tag both sides)
SPEAKER_SPAN = re.compile(
    r"<<\s*(" + "|".join(map(re.escape, SPEAKER_TAGS)) + r")\s*>>"
    r".*?"
    r"<<\s*\1\s*>>",
    re.DOTALL,
)

# Topic header span: << נושא >> title << נושא >>
TOPIC_SPAN = re.compile(
    r"<<\s*(" + "|".join(map(re.escape, TOPIC_TAGS)) + r")\s*>>"
    r"(?P<title>.*?)"
    r"<<\s*\1\s*>>",
    re.DOTALL,
)


def first_speaker_offset(text: str) -> int | None:
    """Offset of the first speaker tag = boundary between preamble and discussion body."""
    for m in ANY_TAG.finditer(text):
        if m.group(1).strip() in SPEAKER_TAGS:
            return m.start()
    return None


def clean_span(text: str) -> str:
    """Remove speaker-name tag spans, any leftover markers, and normalize whitespace."""
    # Remove "<< TAG >> name << TAG >>" spans (drops speaker identities)
    text = SPEAKER_SPAN.sub(" ", text)
    # Remove any remaining tags (unpaired, struct tags, topic headers, etc.)
    text = ANY_TAG.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def body_topic_headers(text: str, body_start: int) -> list[tuple[int, str]]:
    """
    Return body-internal topic headers as (offset, title), i.e. topic markers that occur
    after the preamble. These delimit per-topic discussion sections.
    """
    headers = []
    for m in TOPIC_SPAN.finditer(text):
        if m.start() >= body_start:
            title = re.sub(r"\s+", " ", m.group("title")).strip()
            title = re.sub(r"^\d+\.\s*", "", title)  # strip leading "1. "
            headers.append((m.start(), title))
    return headers


def segment_protocol(doc: dict, gold_topics: list[str]) -> list[dict]:
    """
    Return a list of span records for one protocol.
    """
    text = doc["text"]
    doc_id = doc["doc_id"]
    meta = {
        "doc_id": doc_id,
        "committee": doc.get("committee", ""),
        "date": doc.get("date", ""),
    }

    body_start = first_speaker_offset(text)
    if body_start is None:
        log.warning("%s: no speaker tags found; using whole text as body", doc_id)
        body_start = 0

    headers = body_topic_headers(text, body_start)

    # --- Single-topic protocol: whole body is one span ---
    if len(gold_topics) == 1:
        span_text = clean_span(text[body_start:])
        return [{
            **meta,
            "topic": gold_topics[0],
            "span_text": span_text,
            "n_chars": len(span_text),
            "n_words": len(span_text.split()),
            "segmentation": "single",
            "clean_for_eval": True,
        }]

    # --- Multi-topic with enough body-internal headers: marker split ---
    if len(headers) >= len(gold_topics):
        records = []
        for i, (off, title) in enumerate(headers):
            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            span_text = clean_span(text[off:end])
            records.append({
                **meta,
                "topic": title,            # body-header title (canonicalized downstream if needed)
                "span_text": span_text,
                "n_chars": len(span_text),
                "n_words": len(span_text.split()),
                "segmentation": "marker_split",
                "clean_for_eval": True,
            })
        return records

    # --- Multi-topic, front-loaded (no usable body dividers): one ambiguous span ---
    span_text = clean_span(text[body_start:])
    records = []
    for topic in gold_topics:
        records.append({
            **meta,
            "topic": topic,
            "span_text": span_text,       # same text duplicated across labels
            "n_chars": len(span_text),
            "n_words": len(span_text.split()),
            "segmentation": "frontloaded",
            "clean_for_eval": False,       # excluded from clustering metrics
        })
    return records


def main() -> None:
    canonical = json.load(open(CANONICAL_FILE, encoding="utf-8"))
    files = sorted(PROTOCOLS_DIR.glob("*.json"))
    log.info("Processing %d protocols", len(files))

    all_spans: list[dict] = []
    for f in files:
        doc = json.load(open(f, encoding="utf-8"))
        gold_topics = canonical[doc["doc_id"]]["topics"]
        all_spans.extend(segment_protocol(doc, gold_topics))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_spans, f, ensure_ascii=False, indent=2)

    # Summary
    from collections import Counter
    seg = Counter(s["segmentation"] for s in all_spans)
    clean = sum(s["clean_for_eval"] for s in all_spans)
    word_counts = [s["n_words"] for s in all_spans if s["clean_for_eval"]]
    word_counts.sort()
    median_w = word_counts[len(word_counts) // 2] if word_counts else 0

    log.info("Total spans: %d", len(all_spans))
    log.info("  segmentation: %s", dict(seg))
    log.info("  clean_for_eval spans: %d", clean)
    log.info("  median words (clean spans): %d", median_w)
    log.info("  word range (clean): %d .. %d", word_counts[0], word_counts[-1])
    log.info("Wrote %s", OUT_FILE)


if __name__ == "__main__":
    main()
