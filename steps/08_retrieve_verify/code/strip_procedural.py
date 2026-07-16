"""
Step 08g — Strip recurring administrative/procedural boilerplate out of each
gold segment's span_text before re-embedding, to test whether removing the
shared "budget-request reading" register lets e5_k1 focus on actual topic
content instead of genre.

Patterns target connective/procedural phrasing observed directly in
miss_inspection.txt (budget-transfer templates, request-number headers,
voting boilerplate, currency amounts) -- NOT ministry/authority names or
topic-specific nouns, which are exactly what we want to keep.

This is a manual-regex experiment, explicitly a "give it a shot" per user
request -- if it doesn't measurably help recall@1 on the 36 gold repeat
events (see rerank_recall.py / shortlist_recall.py for the baseline), the
plan is to drop this and rely on the LLM verifier instead.

Output: outputs/stripped_segments.json -- same schema as gold_segments.json,
        span_text replaced with the stripped version.
Usage (CPU-only):
    python steps/08_retrieve_verify/code/strip_procedural.py
"""

import json
import re
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
SPANS_FILE = STEP_DIR.parent / "07_gold_eval" / "outputs" / "gold_segments.json"

PATTERNS = [
    # budget-transfer / request boilerplate templates
    r"ה(?:פנייה|בקשה)(?: התקציבית)? נועדה ל(?:תקצב|תקצוב|העביר|העברת|העברה)",
    r"הבקשה דנה בסעיף",
    r"משנת(?: ה)?(?:תקציב)? \d{2,4}'? ל(?:שנת)?(?: ה)?(?:תקציב)? \d{2,4}'?",
    r"בהרשאה להתחייב",
    r"בהתאם ל(?:פירוט|ביצוע)(?: ה)?(?:תוכניות|תכניות)?(?: הבא(?:ה|ות))?",
    r"בחלוקה ל(?:תוכניות|תכניות)(?: הבאות)?",
    r"(?:העברת|העברה של) עודפ(?:ים|י)?",
    r"עודפי? תקציב(?:ית)?",
    # request/proposal number headers
    r"(?:פני(?:ות|יה)|בקש(?:ה|ות))\s+מס'?\s*\d+(?:[-–]\d+)?",
    r"(?:פני(?:ות|יה)|בקש(?:ה|ות))\s+מספר\s*\d+(?:[-–]\d+)?",
    # voting / procedural chatter
    r"מי בעד\??\s*מי נגד\??\s*מי נמנע\??",
    r"ה(?:פנייה|בקשה) אושרה",
    r"תודה רבה",
    r"בבקשה",
    r"בוקר טוב",
    r"צהרים טובים",
    # currency amounts
    r"[\d,]+(?:\.\d+)?\s*(?:אלפי\s*)?(?:מיליון\s*)?(?:שקל(?:ים)?|ש\"ח|₪)",
]

COMPILED = [re.compile(p) for p in PATTERNS]


def strip_procedural(text: str) -> str:
    for pat in COMPILED:
        text = pat.sub(" ", text)
    return " ".join(text.split())


def main():
    segments = json.load(open(SPANS_FILE, encoding="utf-8"))
    for s in segments:
        s["span_text_orig_n_words"] = s["n_words"]
        s["span_text"] = strip_procedural(s["span_text"])
        s["n_words"] = len(s["span_text"].split())

    total_before = sum(s["span_text_orig_n_words"] for s in segments)
    total_after = sum(s["n_words"] for s in segments)
    print(f"Total words: {total_before} -> {total_after} "
          f"({100 * (1 - total_after / total_before):.1f}% removed)")

    out_path = OUT / "stripped_segments.json"
    json.dump(segments, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")

    print("\n--- sample before/after ---")
    orig = json.load(open(SPANS_FILE, encoding="utf-8"))
    for s, o in list(zip(segments, orig))[6:8]:
        print("BEFORE:", o["span_text"][:300])
        print("AFTER :", s["span_text"][:300])
        print()


if __name__ == "__main__":
    main()
