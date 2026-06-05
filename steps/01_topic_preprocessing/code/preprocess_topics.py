"""
Preprocess topic headers from Knesset protocol JSON files.

Steps:
  1. Extract topics via << נושא >> markers (primary) or << הצח >> markers (alternate).
  2. Fall back to extracting from the סדר היום: agenda line for files with no marker.
  3. Strip leading numbering (e.g. "1. ", "2. ") from multi-topic agendas.
  4. Fuzzy-cluster near-duplicate topic strings and assign a canonical label.
  5. Write outputs/topics_raw.json  — per-file raw extracted topics (before canonicalization)
     and outputs/topics_canonical.json — per-file canonical topic list + the cluster map.

Usage:
    python src/preprocess_topics.py
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]      # steps/01_topic_preprocessing
ROOT = Path(__file__).resolve().parents[3]          # ANLP-PROJECT
PROTOCOLS_DIR = ROOT / "PROTOCOLS"
OUTPUTS_DIR = STEP_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Similarity threshold above which two topics are considered the same
FUZZY_THRESHOLD = 88  # out of 100; tuned for Hebrew near-duplicates
CORRECTIONS_FILE = STEP_DIR / "inputs" / "topics_corrections.json"  # manual overrides


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _strip_numbering(text: str) -> str:
    """Remove leading list numbers like '1. ', '2. ', '1.' etc."""
    return re.sub(r"^\d+\.\s*", "", text.strip())


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_topics_noshaa(text: str) -> list[str]:
    """Extract topics from << נושא >> ... << נושא >> spans."""
    matches = re.findall(r"<<\s*נושא\s*>>(.*?)<<\s*נושא\s*>>", text, re.DOTALL)
    topics = []
    seen = set()
    for m in matches:
        t = _normalize_whitespace(_strip_numbering(m))
        if t and t not in seen:
            seen.add(t)
            topics.append(t)
    return topics


def extract_topics_hatzach(text: str) -> list[str]:
    """Extract topics from << הצח >> ... << הצח >> spans (budget-law variant)."""
    matches = re.findall(r"<<\s*הצח\s*>>(.*?)<<\s*הצח\s*>>", text, re.DOTALL)
    topics = []
    seen = set()
    for m in matches:
        t = _normalize_whitespace(_strip_numbering(m))
        if t and t not in seen:
            seen.add(t)
            topics.append(t)
    return topics


def extract_topics_agenda(text: str) -> list[str]:
    """
    Fall back: extract from the 'סדר היום:' block.
    Captures everything between 'סדר היום:' and the next section header (נכחו / הערה / blank-line).
    """
    m = re.search(r"סדר היום[:\s]+(.*?)(?=\n\n|\nנכחו|\nהערה|$)", text, re.DOTALL)
    if not m:
        return []
    block = m.group(1).strip()
    # Split on numbered list items if present
    items = re.split(r"\n\s*\d+\.\s+", block)
    topics = []
    seen = set()
    for item in items:
        t = _normalize_whitespace(_strip_numbering(item))
        # Remove trailing dash annotations
        t = re.sub(r"\s*[-–]\s*(הכנה לקריאה|בקשה לדיון|דיון והצבעות|המשך|הצבעות).*$", "", t)
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            topics.append(t)
    return topics


def extract_topics(doc_id: str, text: str) -> tuple[list[str], str]:
    """
    Returns (topics, method) where method is one of:
      'noshaa' | 'hatzach' | 'agenda' | 'none'
    """
    topics = extract_topics_noshaa(text)
    if topics:
        return topics, "noshaa"

    topics = extract_topics_hatzach(text)
    if topics:
        return topics, "hatzach"

    topics = extract_topics_agenda(text)
    if topics:
        return topics, "agenda"

    return [], "none"


# ---------------------------------------------------------------------------
# Fuzzy deduplication / canonicalization
# ---------------------------------------------------------------------------

def build_canonical_map(all_topics: list[str], threshold: int = FUZZY_THRESHOLD) -> dict[str, str]:
    """
    Given a flat list of unique topic strings, cluster near-duplicates and
    return a mapping {variant -> canonical}.
    The canonical representative is the most frequent / longest string in each cluster.
    """
    unique = list(dict.fromkeys(all_topics))  # preserve order, dedupe
    clusters: list[list[str]] = []            # each cluster is a list of variants
    assigned: set[str] = set()

    for topic in unique:
        if topic in assigned:
            continue
        # Find all other topics sufficiently similar to this one
        cluster = [topic]
        assigned.add(topic)
        for other in unique:
            if other in assigned:
                continue
            score = fuzz.ratio(topic, other)
            if score >= threshold:
                cluster.append(other)
                assigned.add(other)
        clusters.append(cluster)

    # Canonical = longest string in cluster (usually the most complete)
    canonical_map: dict[str, str] = {}
    for cluster in clusters:
        canonical = max(cluster, key=len)
        for variant in cluster:
            canonical_map[variant] = canonical

    return canonical_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    files = sorted(PROTOCOLS_DIR.glob("*.json"))
    log.info("Found %d protocol files", len(files))

    raw_records: dict[str, dict] = {}     # doc_id -> {topics, method, committee, date}
    all_topic_strings: list[str] = []

    # Pass 1: extract raw topics from every file
    for fpath in files:
        with open(fpath) as f:
            doc = json.load(f)

        doc_id = doc["doc_id"]
        text = doc["text"]
        topics, method = extract_topics(doc_id, text)

        raw_records[doc_id] = {
            "doc_id": doc_id,
            "committee": doc.get("committee", ""),
            "date": doc.get("date", ""),
            "topics_raw": topics,
            "extraction_method": method,
        }

        if method == "none":
            log.warning("No topics found for %s", doc_id)

        all_topic_strings.extend(topics)

    # Pass 2: build canonical map over all observed topic strings
    unique_topics = list(dict.fromkeys(all_topic_strings))
    log.info("Unique raw topic strings: %d", len(unique_topics))

    canonical_map = build_canonical_map(unique_topics)

    # Apply manual corrections (override false-positive merges)
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE) as f:
            corrections = json.load(f)
        overrides = corrections.get("overrides", {})
        canonical_map.update(overrides)
        log.info("Applied %d manual corrections", len(overrides))
    else:
        log.warning("No corrections file found at %s", CORRECTIONS_FILE)

    # Log detected clusters (non-trivial ones)
    clusters_inv: dict[str, list[str]] = defaultdict(list)
    for variant, canon in canonical_map.items():
        clusters_inv[canon].append(variant)

    log.info("Canonical clusters: %d", len(clusters_inv))
    for canon, variants in sorted(clusters_inv.items()):
        if len(variants) > 1:
            log.info("  CLUSTER -> canonical: %s", canon[:80])
            for v in variants:
                if v != canon:
                    log.info("    variant: %s", v[:80])

    # Pass 3: apply canonical map to each record
    canonical_records: dict[str, dict] = {}
    for doc_id, rec in raw_records.items():
        canonical_topics = list(dict.fromkeys(
            canonical_map.get(t, t) for t in rec["topics_raw"]
        ))
        canonical_records[doc_id] = {
            **rec,
            "topics": canonical_topics,
        }

    # Write outputs
    raw_out = OUTPUTS_DIR / "topics_raw.json"
    canonical_out = OUTPUTS_DIR / "topics_canonical.json"
    cluster_out = OUTPUTS_DIR / "topics_clusters.json"

    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)

    with open(canonical_out, "w", encoding="utf-8") as f:
        json.dump(canonical_records, f, ensure_ascii=False, indent=2)

    with open(cluster_out, "w", encoding="utf-8") as f:
        json.dump(
            {canon: variants for canon, variants in clusters_inv.items() if len(variants) > 1},
            f, ensure_ascii=False, indent=2
        )

    log.info("Wrote %s", raw_out)
    log.info("Wrote %s", canonical_out)
    log.info("Wrote %s", cluster_out)

    # Summary
    methods = defaultdict(int)
    for rec in raw_records.values():
        methods[rec["extraction_method"]] += 1
    log.info("Extraction method counts: %s", dict(methods))

    no_topic = [doc_id for doc_id, rec in raw_records.items() if rec["extraction_method"] == "none"]
    if no_topic:
        log.warning("Files still with NO topic after all fallbacks: %s", no_topic)


if __name__ == "__main__":
    main()
