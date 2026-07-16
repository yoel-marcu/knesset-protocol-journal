"""
Step 09b — Generate journal entries: an initial summary for each chain's first
occurrence, then an incremental (RAG-style) update for every subsequent occurrence,
given the running journal so far + the new segment's text.

Per CLAUDE.md Task 3: "Given the existing journal entry and the new protocol
segment, generate only the incremental contribution (new developments, decisions,
status changes)." The model sees the accumulated dated journal (built from its own
prior generations, not raw source text of earlier sessions) plus the new segment's
text -- matching "faithfulness is chain-relative: measured against the chain of
prior summaries, not the raw source document."

Model-agnostic: takes any HF causal-LM chat model, run once per model to compare
(mirrors Step 08's run_verifier.py pattern).

Input:  outputs/journal_chains.json
Output: outputs/generated_logs_<short_name>.json
Usage (GPU):
    python steps/09_log_writing/code/generate_logs.py \
        --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 2
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"

SNIPPET_WORDS = 700

OPENING_PROMPT = """You are opening a journal entry for a legislative matter discussed in an \
Israeli Knesset Finance Committee session. Below is the full text of the session where this \
matter was first raised.

Write a concise opening journal entry (2-4 sentences, in Hebrew) summarizing what this matter \
is about: what is being proposed/discussed, and by whom if relevant. Base it ONLY on the text \
below -- do not invent details not present in it.

Respond with ONLY the journal entry text, nothing else (no headers, no JSON, no preamble).

SESSION TEXT (date {date}):
<segment>
{text}
</segment>

JOURNAL ENTRY:"""

UPDATE_PROMPT = """You are maintaining a chronological journal entry for a legislative matter \
discussed across multiple Israeli Knesset Finance Committee sessions. Below is the EXISTING \
JOURNAL (everything written about this matter so far, each entry dated) and the full text of a \
NEW session where this same matter came up again.

Write ONLY the new incremental update (1-3 sentences, in Hebrew): new developments, decisions, \
or status changes visible in the new session. Do NOT repeat information already in the existing \
journal. Do NOT invent details not present in the new session text below. If the new session \
genuinely adds nothing beyond what's already logged, say so briefly rather than restating old \
content.

Respond with ONLY the new update text, nothing else (no headers, no JSON, no preamble, no date).

EXISTING JOURNAL:
{journal}

NEW SESSION TEXT (date {date}):
<segment>
{text}
</segment>

NEW INCREMENTAL UPDATE:"""

# v2: adds a real worked example (from an earlier run of this exact pipeline) contrasting a
# BAD update -- which opens by re-explaining the existing matter before adding the new fact,
# diluting the update with recap -- against a GOOD one that states only the new fact, with no
# recap sentence at all. Same "worked example fixes behavior better than instruction alone"
# strategy that fixed Step 08's schema-following problems.
UPDATE_PROMPT_V2 = """You are maintaining a chronological journal entry for a legislative matter \
discussed across multiple Israeli Knesset Finance Committee sessions. Below is the EXISTING \
JOURNAL (everything written about this matter so far, each entry dated) and the full text of a \
NEW session where this same matter came up again.

Write ONLY the new incremental update (1-3 sentences, in Hebrew): new developments, decisions, \
or status changes visible in the new session. Do NOT invent details not present in the new \
session text. If the new session genuinely adds nothing beyond what's already logged, say so \
briefly rather than restating old content.

CRITICAL RULE: do not restate, recap, or re-explain ANY fact that is already in the existing \
journal, not even briefly as a lead-in. Do not write sentences like "the matter remains as \
discussed, however..." or "regarding X, which was already covered, a new detail emerged...". \
Jump straight to the new fact itself, with zero framing sentences about what's already known.

WORKED EXAMPLE:
EXISTING JOURNAL:
[2022-11-29] הנושא הנדון בוועדת הכספים הוא ההשפעה המתרחבת של מחירי הדיור הגבוהים. חבר הכנסת משה \
גפני מציע התערבות ממשלתית שתקל על זוגות צעירים לרכוש דירה באמצעות הפחתת ריבית המשכנתאות.

NEW SESSION TEXT (date 2022-12-06):
<segment>
היו"ר משה גפני: בקשה מס' 42-005: מענקי בינוי ושיכון – מענקים לסבסוד ריבית. יפתח עשהאל: חלק \
מהמענקים הללו מיועדים לקשישים עולים הממתינים לדיור ציבורי, ומטרתם להשוות את הסיוע בשכר דירה בין \
קשישים אלה לבין ותיקים הממתינים לדיור ציבורי. ינון אזולאי (ש"ס): האם גם עולים אחרים כמו עולי צרפת \
או עולים מאתיופיה נהנים מהמענקים? יפתח עשהאל: לא, המענקים ניתנו בהתאם להחלטת ממשלה שהתמקדה בקשישים \
עולים בלבד.
</segment>

BAD UPDATE (do NOT write like this -- opens with a recap sentence before the new fact):
"הנושא של מענקי בינוי ושיכון, אשר נידון בוועדת הכספים, נשאר בעינו. עם זאת, מתגלה פרט חדש: חלק \
מהמענקים הללו מיועדים לקשישים עולים הממתינים לדיור ציבורי..."
(Why bad: "נשאר בעינו... עם זאת" is pure recap of the existing journal, adding nothing.)

GOOD UPDATE (write like this -- states only the new fact, no recap):
"מענקי סבסוד הריבית לבינוי ושיכון (בקשה 42-005) מיועדים לקשישים עולים הממתינים לדיור ציבורי, לא \
לכלל העולים -- שאלת ח"כ אזולאי על עולי צרפת ואתיופיה נענתה בשלילה, בהתאם להחלטת הממשלה."

Now write the real update below, following the GOOD example's style exactly.

EXISTING JOURNAL:
{journal}

NEW SESSION TEXT (date {date}):
<segment>
{text}
</segment>

NEW INCREMENTAL UPDATE:"""

# v3: same anti-recap goal as v2, but the worked example content is now fully INVENTED
# (a fictional committee matter, fictional names/numbers) rather than drawn from the real
# corpus, plus an explicit anti-copying instruction. v2 caused 13/36 (36%) of dictalm2's
# updates to be literal verbatim copies of the real-corpus worked example, regardless of
# the actual input -- the model pattern-matched the example instead of generalizing the
# style. A synthetic example gives it nothing plausible to copy into a real answer.
UPDATE_PROMPT_V3 = """You are maintaining a chronological journal entry for a legislative matter \
discussed across multiple Israeli Knesset Finance Committee sessions. Below is the EXISTING \
JOURNAL (everything written about this matter so far, each entry dated) and the full text of a \
NEW session where this same matter came up again.

Write ONLY the new incremental update (1-3 sentences, in Hebrew): new developments, decisions, \
or status changes visible in the new session. Do NOT invent details not present in the new \
session text. If the new session genuinely adds nothing beyond what's already logged, say so \
briefly rather than restating old content.

CRITICAL RULE 1: do not restate, recap, or re-explain ANY fact that is already in the existing \
journal, not even briefly as a lead-in. Do not write sentences like "the matter remains as \
discussed, however..." or "regarding X, which was already covered, a new detail emerged...". \
Jump straight to the new fact itself, with zero framing sentences about what's already known.

CRITICAL RULE 2: the worked example below is COMPLETELY FICTIONAL and shown ONLY to illustrate \
the required STYLE (no-recap, straight to the new fact). It has nothing to do with the real \
case you will answer. Do NOT reuse, adapt, or reference any name, number, or fact from the \
worked example in your real answer -- your real answer must come ENTIRELY from the real NEW \
SESSION TEXT given after it.

WORKED EXAMPLE (fictional, for STYLE ONLY):
EXISTING JOURNAL:
[2023-03-01] הוועדה דנה בהצעה למימון שיפוץ ספריות ציבוריות דיגיטליות. חברת הכנסת רות אבני הציעה \
תקציב של 40 מיליון שקל לפרויקט.

NEW SESSION TEXT (date 2023-03-15):
<segment>
[fictional example text about the fictional library-funding matter, with a fictional new detail]
</segment>

BAD UPDATE (do NOT write like this -- opens with a recap sentence before the new fact):
"הנושא של מימון שיפוץ הספריות הדיגיטליות, שהוצע על ידי חברת הכנסת אבני, נשאר בעינו. עם זאת, \
התווסף פרט חדש: התקציב הועלה ל-55 מיליון שקל."
(Why bad: "נשאר בעינו... עם זאת" is pure recap of the existing journal, adding nothing.)

GOOD UPDATE (write like this -- states only the new fact, no recap):
"התקציב לשיפוץ הספריות הדיגיטליות הועלה ל-55 מיליון שקל, בעקבות בקשת משרד התרבות להרחיב את \
הפרויקט ל-12 ערים נוספות."

Now write the REAL update below, based ENTIRELY on the REAL existing journal and REAL new \
session text -- not on the fictional example above.

EXISTING JOURNAL:
{journal}

NEW SESSION TEXT (date {date}):
<segment>
{text}
</segment>

NEW INCREMENTAL UPDATE:"""


def snippet(text: str, n: int = SNIPPET_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])


def strip_reply(text: str) -> str:
    text = text.strip()
    for stop in ["\nSESSION", "\nNEW SESSION", "\nEXISTING JOURNAL", "\n<segment>",
                 "\nWORKED EXAMPLE", "\nBAD UPDATE", "\nGOOD UPDATE"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx].strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--short_name", required=True, help="output filename tag")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--prompt_version", choices=["original", "v2", "v3"], default="original",
                     help="v2 adds a real-corpus worked BAD-vs-GOOD example (caused 36% verbatim "
                          "copying on dictalm2 -- see README); v3 uses a fictional worked example "
                          "instead, plus an explicit anti-copying instruction")
    args = ap.parse_args()

    update_prompt_template = {
        "original": UPDATE_PROMPT, "v2": UPDATE_PROMPT_V2, "v3": UPDATE_PROMPT_V3,
    }[args.prompt_version]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    chains = json.load(open(OUT / "journal_chains.json", encoding="utf-8"))
    log.info("Loaded %d chains", len(chains))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=6000).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=1.15, pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return strip_reply(tok.batch_decode(gen, skip_special_tokens=True)[0])

    results = []
    for ci, chain in enumerate(chains):
        occs = chain["occurrences"]
        entries = []  # [{date, text}]

        opening_text = snippet(occs[0]["span_text"])
        opening = generate(OPENING_PROMPT.format(date=occs[0]["date"][:10], text=opening_text))
        entries.append({"date": occs[0]["date"], "text": opening, "is_opening": True})

        for occ in occs[1:]:
            journal_str = "\n".join(f"[{e['date'][:10]}] {e['text']}" for e in entries)
            new_text = snippet(occ["span_text"])
            update = generate(update_prompt_template.format(journal=journal_str, date=occ["date"][:10], text=new_text))
            entries.append({"date": occ["date"], "text": update, "is_opening": False})

        results.append({
            "topic": chain["topic"],
            "occurrences": [{"doc_id": o["doc_id"], "seg_idx": o["seg_idx"], "date": o["date"]} for o in occs],
            "entries": entries,
        })
        log.info("  chain %d/%d done (%s, %d occurrences)", ci + 1, len(chains), chain["topic"][:40], len(occs))

    out_path = OUT / f"generated_logs_{args.short_name}.json"
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
