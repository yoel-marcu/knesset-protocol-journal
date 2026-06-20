"""
Step 06 — Interactive CLI annotator for subject-linking pairs.

Loads the annotation batch, shows each pair, and records your label.
Progress is saved after every pair — you can Ctrl-C and resume anytime.

Labels:
  s / same     — same legislative matter (should be ONE journal entry)
  r / related  — related but distinct (separate entries, may cross-reference)
  n / new      — clearly different subject
  skip / ?     — uncertain; leave for adjudication

Usage:
    python annotate.py --annotator YOURNAME
    python annotate.py --annotator YOURNAME --show-stratum   # reveal sampling stratum
    python annotate.py --annotator YOURNAME --only hard_neg  # annotate one stratum only
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from bidi.algorithm import get_display
    def rtl(text: str) -> str:
        """Reorder Hebrew text for correct display in an LTR terminal."""
        return get_display(text)
except ImportError:
    def rtl(text: str) -> str:
        return text

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
BATCH_FILE = OUT / "annotation_batch.json"

LABEL_MAP = {
    "s": "same", "same": "same",
    "r": "related", "related": "related",
    "n": "new", "new": "new",
    "skip": "skip", "?": "skip", "": "skip",
}

SEP  = "─" * 72
SEP2 = "═" * 72


def load_batch():
    if not BATCH_FILE.exists():
        print(f"ERROR: {BATCH_FILE} not found. Run sample_pairs.py first.")
        sys.exit(1)
    return json.load(open(BATCH_FILE, encoding="utf-8"))


def annotator_path(name: str) -> Path:
    return OUT / f"annotations_{name}.json"


def load_progress(name: str) -> dict:
    path = annotator_path(name)
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    return {"annotator": name, "pairs": {}}


def save_progress(name: str, progress: dict):
    path = annotator_path(name)
    json.dump(progress, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def display_pair(pair: dict, idx: int, total: int, show_stratum: bool):
    a, b = pair["a"], pair["b"]
    print()
    print(SEP2)
    stratum_tag = f"  [{pair['stratum']}]" if show_stratum else ""
    print(f"  Pair {idx}/{total}  (id: {pair['id']}){stratum_tag}")
    print(SEP2)

    for label, span in [("A", a), ("B", b)]:
        print(f"\n  ── Span {label} ──────────────────────────────────────────────")
        print(f"  Date   : {span['date']}")
        print(f"  Topic  : {rtl(span['topic'])}")
        print(f"  Preview: {rtl(span['preview'][:250])}")

    print(f"\n{SEP}")
    print("  Labels: [s]ame  [r]elated  [n]ew  [skip/?]  [q]uit")
    print(SEP)


def run(annotator: str, show_stratum: bool, only_stratum: str | None):
    batch = load_batch()
    progress = load_progress(annotator)
    done = progress["pairs"]

    pairs = batch["pairs"]
    if only_stratum:
        pairs = [p for p in pairs if p["stratum"] == only_stratum]

    remaining = [p for p in pairs if p["id"] not in done]
    already   = len(pairs) - len(remaining)

    print(f"\n{SEP2}")
    print(f"  Annotator : {annotator}")
    print(f"  Total pairs in scope : {len(pairs)}")
    print(f"  Already annotated    : {already}")
    print(f"  Remaining            : {len(remaining)}")
    print(SEP2)

    if not remaining:
        print("\n  ✓ All pairs annotated. Nothing left to do.")
        _print_summary(done)
        return

    print("\n  Press Enter to begin. Ctrl-C to stop (progress is saved).\n")
    try:
        input()
    except KeyboardInterrupt:
        return

    idx_base = already + 1
    try:
        for offset, pair in enumerate(remaining):
            display_pair(pair, idx_base + offset, len(pairs), show_stratum)

            while True:
                try:
                    raw = input("  Your label: ").strip().lower()
                except KeyboardInterrupt:
                    print("\n\n  Interrupted — progress saved.")
                    save_progress(annotator, progress)
                    return

                if raw == "q":
                    print("  Quitting — progress saved.")
                    save_progress(annotator, progress)
                    return

                label = LABEL_MAP.get(raw)
                if label is None:
                    print(f"  ✗ Unknown input '{raw}'. Use s/r/n/skip/q.")
                    continue

                done[pair["id"]] = {
                    "label":   label,
                    "pair_id": pair["id"],
                    "stratum": pair["stratum"],
                }
                save_progress(annotator, progress)

                tag = {"same": "✓ SAME", "related": "~ RELATED",
                       "new": "✗ NEW", "skip": "? SKIP"}[label]
                print(f"  {tag}  (saved)")
                break

    except EOFError:
        pass

    print(f"\n{SEP}")
    print("  Session complete — progress saved.")
    _print_summary(done)


def _print_summary(done: dict):
    from collections import Counter
    counts = Counter(v["label"] for v in done.values())
    strata = Counter(v["stratum"] for v in done.values())
    print(f"\n  Annotations so far: {len(done)} total")
    print(f"  Labels  : {dict(counts)}")
    print(f"  Strata  : {dict(strata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator",    required=True,
                        help="Your name/id (e.g. yoel, tomer, or)")
    parser.add_argument("--show-stratum", action="store_true",
                        help="Show whether pair is positive/hard_neg/easy_neg")
    parser.add_argument("--only",         dest="only_stratum", default=None,
                        choices=["positive", "hard_neg", "easy_neg"],
                        help="Annotate only one stratum")
    args = parser.parse_args()
    run(args.annotator, args.show_stratum, args.only_stratum)
