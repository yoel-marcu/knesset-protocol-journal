"""
Step 08c — Run an LLM verifier over the retrieved-candidate queries.

For each of the 522 streaming decisions in verifier_queries.json, the model sees
the new segment's content snippet plus its top-10 e5_k1-retrieved candidate
snippets (no gold labels -- see build_verifier_inputs.py), and must decide:
  "same"    -- genuinely the same legislative matter as one specific candidate
  "related" -- related but distinct matter
  "new"     -- unrelated to any candidate

Asymmetric cost (false-merge >> false-split, per project design constraint):
the prompt explicitly tells the model to prefer "related"/"new" over "same"
when unsure, mirroring the human annotation guideline from Step 06.

Model-agnostic: takes any HF causal-LM chat model. Run once per model to compare
(DictaLM-2.0-instruct, Qwen2.5-*B-Instruct, ...).

Precision-calibration experiments (all opt-in via flags, default = original run):
  --fewshot extended     4 worked examples instead of 2 (2 more, covering a fresh
                          positive continuation and a fresh "same funding template,
                          different ministry" hard negative -- see FEWSHOT_EXTENDED).
  --snippet_words N      Re-derive candidate/query snippets from full gold span text
                          at N words instead of the pre-baked 100 in verifier_queries.json
                          (tests whether the 100-word cap is starving the model of the
                          specific-matter signal that ministry/template language drowns out).
  --two_score            Ask for topical_similarity (domain/ministry overlap) and
                          specific_match_confidence (same exact matter) separately instead
                          of one conflated same_confidence number.

Input:  outputs/verifier_queries.json
Output: outputs/verifier_predictions_<short_name>.json
Usage (GPU):
    python steps/08_retrieve_verify/code/run_verifier.py \
        --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 4
    python steps/08_retrieve_verify/code/run_verifier.py \
        --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b_longsnippet --snippet_words 400
    python steps/08_retrieve_verify/code/run_verifier.py \
        --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b_fewshot --fewshot extended
    python steps/08_retrieve_verify/code/run_verifier.py \
        --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b_twoscore --two_score
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"
SPANS_FILE = STEP07_OUT / "gold_segments.json"

SYSTEM_PROMPT = """You are a careful assistant judging whether a new discussion segment \
from an Israeli Knesset committee protocol is the SAME legislative matter as one of \
several candidate prior journal entries.

The segment texts below are quoted excerpts of real transcripts, given to you ONLY as \
context to compare. Do NOT continue, repeat, translate, or quote them back. Your entire \
reply must be the single JSON verdict, nothing else -- not one word of the transcript.

Look at all the candidates and pick the ONE most likely to be the same matter as the new \
segment (by its rank number), or null if none seem related at all. Then give a confidence \
score from 0 to 100 for how confident you are that this candidate is genuinely the SAME \
specific matter -- not just the same ministry, committee, or general topic area. A high \
score means you are sure it's the same underlying issue being continued; a low score means \
it only superficially resembles it (same ministry/format but a different specific matter).

Respond with ONLY a single JSON object, no other text:
{"best_matched_rank": <candidate rank int, or null>, "same_confidence": <integer 0-100>}
If best_matched_rank is null, same_confidence must be 0."""

SYSTEM_PROMPT_TWOSCORE = """You are a careful assistant judging whether a new discussion \
segment from an Israeli Knesset committee protocol is the SAME legislative matter as one of \
several candidate prior journal entries.

The segment texts below are quoted excerpts of real transcripts, given to you ONLY as \
context to compare. Do NOT continue, repeat, translate, or quote them back. Your entire \
reply must be the single JSON verdict, nothing else -- not one word of the transcript.

Look at all the candidates and pick the ONE most likely to be the same matter as the new \
segment (by its rank number), or null if none seem related at all. Then give TWO separate \
scores from 0 to 100:
  - "topical_similarity": how much the general subject/domain overlaps (same ministry,
    same committee area, same type of budget item) -- this can be high even when it's
    actually a different specific matter.
  - "specific_match_confidence": how confident you are that this is genuinely the SAME
    underlying issue being continued, not just the same domain/ministry/template. This is
    the score that should stay LOW when topical_similarity is high but the actual matter
    differs (e.g. two different budget requests for the same ministry).

Respond with ONLY a single JSON object, no other text:
{"best_matched_rank": <candidate rank int, or null>, "topical_similarity": <integer 0-100>, \
"specific_match_confidence": <integer 0-100>}
If best_matched_rank is null, both scores must be 0."""

# Two real, verified worked examples (from actual corpus segments) to anchor both the
# exact JSON schema and the "same ministry does not mean same matter" judgment that
# embedding/lexical retrieval was shown to fail at (see project analysis).
FEWSHOT_ORIGINAL = """
WORKED EXAMPLE 1 (genuinely the same matter -- high confidence):
NEW SEGMENT:
<segment>
שלום דנינו (הליכוד): סליחה, שר החקלאות, עוד הערה. אני רוצה לדבר על המחיר המפוקח, שר החקלאות. אתם מדברים על מחיר מטרה שמדבר על מה שהרפתן מקבל. אתם מדברים מה המחיר שזה יימכר.
</segment>
CANDIDATE PRIOR JOURNAL ENTRIES:
[1]
<segment>
היו"ר משה גפני: בוקר טוב, אני פותח את ישיבת ועדת הכספים הזמנית. על סדר-היום: עליית מחירי החלב והמשבר בענף. תמונת המצב היא תמונה עגומה.
</segment>
CORRECT ANSWER: {"best_matched_rank": 1, "same_confidence": 95}
(Why: both are part of the same ongoing discussion of the rising milk-price crisis.)

WORKED EXAMPLE 2 (same ministry, but a DIFFERENT specific matter -- low confidence):
NEW SEGMENT:
<segment>
פנייה 311, משרד המשפטים. עידו חי: הפנייה התקציבית נועדה לביצוע שינויים פנימיים בסעיף 08, משרד המשפטים, הנהלת בתי משפט ורשות האכיפה והגבייה לצורך התאמת התקציב לביצוע בפועל של שנת 22'.
</segment>
CANDIDATE PRIOR JOURNAL ENTRIES:
[1]
<segment>
היו"ר משה גפני: אוקיי. אנחנו עוברים לעודפים. פנייה, בקשה מספר 05008014, משרד המשפטים. עידו חי: הפנייה התקציבית נועדה להעברת עודפים משנת התקציב 2021 לשנת התקציב 2022 למשרד המשפטים בסך של שלושה מיליון.
</segment>
CORRECT ANSWER: {"best_matched_rank": 1, "same_confidence": 15}
(Why: candidate [1] is the closest of the candidates, but it's still a LOW score -- both mention
the Ministry of Justice and a budget-surplus transfer, but candidate [1] is a routine year-end
surplus reallocation while the new segment is an internal budget realignment for court
administration. Same ministry and same generic template, but a different specific matter. A
shared ministry or bureaucratic phrasing alone should never push the score high.)

Now decide the REAL case below, using the same JSON format. Do not repeat the worked examples.
"""

# Extended block: the same 2 examples above, plus 2 fresh ones covering a new
# recurrence-across-a-real-gap positive case, and a hard negative that isolates an
# even harder confound than "same ministry" -- same PRESENTER and same funding
# MECHANISM (year-end covid-surplus reallocation), still a different specific matter.
FEWSHOT_EXTENDED = FEWSHOT_ORIGINAL.rstrip() + """

WORKED EXAMPLE 3 (genuinely the same matter, ~1 month gap, different phrasing -- high confidence):
NEW SEGMENT:
<segment>
היו"ר משה גפני: סיוע לענף גידול הבטטות בעקבות משבר נגיף הקורונה. משרד החקלאות, בקשה. מועמר חאג' יחיא: אנחנו סבורים שבהעדר אפשרות לשלם פיצויים ישירים על הקורונה יש אפשרות לסייע לחקלאים באמצעות השקעות הון.
</segment>
CANDIDATE PRIOR JOURNAL ENTRIES:
[1]
<segment>
היו"ר משה גפני: אנחנו עוסקים בסיוע לענף גידול הבטטות בעקבות משבר נגיף הקורונה. יש מכתב שקיבלתי מגיא דוד עם פירוט הבעיה הזאת. גיא, בבקשה תסביר לוועדה מה הבקשה שלך. גיא דוד: ענף הבטטות בישראל חווה טלטלות קשות.
</segment>
CORRECT ANSWER: {"best_matched_rank": 1, "same_confidence": 90}
(Why: same specific aid request for the sweet-potato growing sector, resumed a month later --
different speakers and phrasing, but unmistakably the same underlying matter.)

WORKED EXAMPLE 4 (same presenter AND same funding template, but a DIFFERENT specific matter -- low confidence):
NEW SEGMENT:
<segment>
היו"ר משה גפני: 19019, תרבות וספורט, מתקני ספורט. איזה משרד זה? יואב הכט: משרד התרבות והספורט. באופן דומה לפנייה הקודמת מדובר בהעברת עודפים משנת 2021 מתוך מסגרת תקציב הקורונה.
</segment>
CANDIDATE PRIOR JOURNAL ENTRIES:
[1]
<segment>
היו"ר משה גפני: בקשה 13026, הוצאות שונות, תשלומים באמצעות רשות המסים, מענקים לעסקים. יואב הכט: הפנייה נועדה להעברת עודפים משנת 2021 לשנת 2022 ממסגרת התקציב המיועדת להתמודדות עם נגיף הקורונה, מדובר ב-85,485,000 ₪ שהוקצו לטובת מענקים, בין היתר מענקים לעסקים.
</segment>
CORRECT ANSWER: {"best_matched_rank": 1, "same_confidence": 10}
(Why: same presenter (יואב הכט) and the identical "covid-era year-end surplus transfer"
funding mechanism -- an even closer surface resemblance than sharing a ministry -- but one is
sports-facility funding and the other is business grants via the tax authority. Completely
different specific matter. Neither a shared presenter nor a shared funding template should
push the score high on its own.)

Now decide the REAL case below, using the same JSON format. Do not repeat the worked examples.
"""

FEWSHOT_TWOSCORE = """
WORKED EXAMPLE 1 (genuinely the same matter -- high confidence on both scores):
NEW SEGMENT:
<segment>
שלום דנינו (הליכוד): סליחה, שר החקלאות, עוד הערה. אני רוצה לדבר על המחיר המפוקח, שר החקלאות. אתם מדברים על מחיר מטרה שמדבר על מה שהרפתן מקבל. אתם מדברים מה המחיר שזה יימכר.
</segment>
CANDIDATE PRIOR JOURNAL ENTRIES:
[1]
<segment>
היו"ר משה גפני: בוקר טוב, אני פותח את ישיבת ועדת הכספים הזמנית. על סדר-היום: עליית מחירי החלב והמשבר בענף. תמונת המצב היא תמונה עגומה.
</segment>
CORRECT ANSWER: {"best_matched_rank": 1, "topical_similarity": 95, "specific_match_confidence": 95}
(Why: both are part of the same ongoing discussion of the rising milk-price crisis -- domain
overlap AND specific-matter overlap are both high here.)

WORKED EXAMPLE 2 (same ministry, HIGH topical overlap but LOW specific-match -- the key distinction):
NEW SEGMENT:
<segment>
פנייה 311, משרד המשפטים. עידו חי: הפנייה התקציבית נועדה לביצוע שינויים פנימיים בסעיף 08, משרד המשפטים, הנהלת בתי משפט ורשות האכיפה והגבייה לצורך התאמת התקציב לביצוע בפועל של שנת 22'.
</segment>
CANDIDATE PRIOR JOURNAL ENTRIES:
[1]
<segment>
היו"ר משה גפני: אוקיי. אנחנו עוברים לעודפים. פנייה, בקשה מספר 05008014, משרד המשפטים. עידו חי: הפנייה התקציבית נועדה להעברת עודפים משנת התקציב 2021 לשנת התקציב 2022 למשרד המשפטים בסך של שלושה מיליון.
</segment>
CORRECT ANSWER: {"best_matched_rank": 1, "topical_similarity": 70, "specific_match_confidence": 15}
(Why: topical_similarity is fairly high -- same ministry, same section 08, same general
budget-transfer format. But specific_match_confidence is LOW: candidate [1] is a routine
year-end surplus reallocation while the new segment is an internal budget realignment for
court administration -- different specific matters. This is exactly the case the two scores
are meant to separate: high domain overlap must NOT drag up specific-match confidence.)

Now decide the REAL case below, using the same JSON format. Do not repeat the worked examples.
"""


def snippet(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n])


def build_user_prompt(q: dict, span_lookup: dict | None, snippet_words: int | None) -> str:
    def get_snippet(doc_id, seg_idx, fallback):
        if snippet_words is None or span_lookup is None:
            return fallback
        s = span_lookup.get((doc_id, seg_idx))
        return snippet(s["span_text"], snippet_words) if s else fallback

    q_snip = get_snippet(q["query_doc_id"], q["query_seg_idx"], q["query_snippet"])
    lines = [
        f"NEW SEGMENT (date {q['query_date'][:10]}, committee {q['query_committee']}):",
        "<segment>", q_snip, "</segment>",
        "",
        "CANDIDATE PRIOR JOURNAL ENTRIES (ranked by retrieval similarity):",
    ]
    for c in q["candidates"]:
        c_snip = get_snippet(c["doc_id"], c["seg_idx"], c["snippet"])
        lines.append(f"[{c['rank']}] (date {c['date'][:10]}, committee {c['committee']}):")
        lines.append("<segment>")
        lines.append(c_snip)
        lines.append("</segment>")
    lines.append("")
    lines.append(
        "Output ONLY the JSON verdict now. Do not output any part of the segment text above."
    )
    return "\n".join(lines)


JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_decision(text: str, two_score: bool) -> dict:
    m = JSON_RE.search(text)
    if not m:
        base = {"best_matched_rank": None, "_parse_error": True, "_raw": text[:200]}
        base.update({"topical_similarity": 0, "specific_match_confidence": 0} if two_score
                     else {"same_confidence": 0})
        return base
    try:
        d = json.loads(m.group(0))
        rank = d.get("best_matched_rank")
        rank = int(rank) if rank is not None else None
        if two_score:
            topical = max(0, min(100, int(d.get("topical_similarity", 0))))
            specific = max(0, min(100, int(d.get("specific_match_confidence", 0))))
            if rank is None:
                topical = specific = 0
            return {"best_matched_rank": rank, "topical_similarity": topical,
                    "specific_match_confidence": specific}
        else:
            conf = max(0, min(100, int(d.get("same_confidence", 0))))
            if rank is None:
                conf = 0
            return {"best_matched_rank": rank, "same_confidence": conf}
    except (json.JSONDecodeError, ValueError, TypeError):
        base = {"best_matched_rank": None, "_parse_error": True, "_raw": text[:200]}
        base.update({"topical_similarity": 0, "specific_match_confidence": 0} if two_score
                     else {"same_confidence": 0})
        return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--short_name", required=True, help="output filename tag")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--fewshot", choices=["original", "extended"], default="original")
    ap.add_argument("--snippet_words", type=int, default=None,
                     help="re-derive snippets from full gold span text at this word count "
                          "(default: use the pre-baked 100-word snippet in verifier_queries.json)")
    ap.add_argument("--two_score", action="store_true",
                     help="use the two-score schema (topical_similarity + specific_match_confidence) "
                          "instead of one conflated same_confidence")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    queries = json.load(open(OUT / "verifier_queries.json", encoding="utf-8"))
    log.info("Loaded %d queries", len(queries))

    span_lookup = None
    if args.snippet_words is not None:
        span_lookup = {(s["doc_id"], s["seg_idx"]): s
                        for s in json.load(open(SPANS_FILE, encoding="utf-8"))
                        if s["clean_for_eval"]}
        log.info("Loaded %d gold spans for --snippet_words=%d re-derivation",
                 len(span_lookup), args.snippet_words)

    system_prompt = SYSTEM_PROMPT_TWOSCORE if args.two_score else SYSTEM_PROMPT
    if args.two_score:
        fewshot = FEWSHOT_TWOSCORE
    else:
        fewshot = FEWSHOT_EXTENDED if args.fewshot == "extended" else FEWSHOT_ORIGINAL
    log.info("Config: fewshot=%s snippet_words=%s two_score=%s",
             args.fewshot, args.snippet_words, args.two_score)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched decoder-only generation
    tok.truncation_side = "left"  # if truncation must happen, keep the closing instruction
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    # Single user turn (system prompt merged in) for compatibility with chat templates
    # that reject a separate system role (e.g. DictaLM-2.0's Mistral-based template
    # raises "Conversation roles must alternate user/assistant/...").
    prompts = []
    for q in queries:
        user_prompt = build_user_prompt(q, span_lookup, args.snippet_words)
        content = system_prompt + "\n" + fewshot + "\n" + user_prompt
        messages = [{"role": "user", "content": content}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    predictions = []
    bs = args.batch_size
    max_len = 4608 if args.snippet_words is None or args.snippet_words <= 100 else 4608 + 40 * (args.snippet_words - 100)
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            # NOTE: no_repeat_ngram_size is deliberately NOT used here -- the few-shot
            # examples in the prompt contain literal JSON output text (e.g.
            # {"decision": "new", "matched_rank": null}), and no_repeat_ngram_size bans
            # n-grams anywhere in prompt+generation, so it was banning the model from
            # producing the very JSON schema tokens it needs to legitimately repeat.
            # A modest repetition_penalty alone is enough to avoid the original
            # document-continuation degenerate loops.
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for text in texts:
            predictions.append(parse_decision(text, args.two_score))
        if (i // bs + 1) % 20 == 0:
            log.info("  %d/%d", i + len(batch), len(prompts))

    n_errors = sum(1 for p in predictions if p.get("_parse_error"))
    log.info("Done. %d/%d predictions had JSON parse errors (defaulted to confidence=0)",
             n_errors, len(predictions))

    out_path = OUT / f"verifier_predictions_{args.short_name}.json"
    json.dump(predictions, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
