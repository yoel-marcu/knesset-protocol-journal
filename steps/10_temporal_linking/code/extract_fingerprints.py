"""
Step 10 — Family D: LLM extracts a structured "matter fingerprint" per span.

Motivation: Step 8 showed the hard false-merge confounds are semantically
irreducible by geometry -- e.g. two different ministries' budget requests, or even
two DIFFERENT matters under the SAME ministry, that share the bureaucratic register.
The ministry-only gate failed (+1/36) because a single ministry token is too coarse.
This uses the LLM as an EXTRACTOR (its strength) not a classifier (its weakness in
Step 8): pull the fine-grained, matter-specific identifiers so linking can require
STABLE-identifier overlap, not just gist similarity.

Extracts, per span:
  ministry        -- the government body (weak signal; e.g. Finance is everywhere)
  laws_programs   -- named laws / bills / funds / programs (STABLE across recurrences)
  entities        -- specific named companies/people/places/institutions (STABLE)
  request_numbers -- פנייה/בקשה numbers (TRANSIENT: change every session; a NON-match
                     signal for budget items, kept separate so matching can ignore them)
  matter          -- one-line Hebrew description of the specific matter

Input:  steps/07_gold_eval/outputs/gold_segments.json  (span_text)
        steps/07_gold_eval/outputs/embeddings/ids.json  (order/topics for alignment)
Output: outputs/fingerprints_<short_name>.json  (list aligned to ids.json order)
Usage (GPU):
    python steps/10_temporal_linking/code/extract_fingerprints.py \
        --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 4
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
OUT.mkdir(parents=True, exist_ok=True)
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"

SNIPPET_WORDS = 500

PROMPT = """You extract a structured fingerprint of the SPECIFIC legislative matter discussed in \
an excerpt from an Israeli Knesset committee session. Read the excerpt and output ONLY a JSON \
object, nothing else, with exactly these keys:

{{"ministry": "<the main government ministry/body, or empty string>",
  "laws_programs": ["<names of specific laws, bills, funds, or programs mentioned>"],
  "entities": ["<specific named companies, people, places, or institutions>"],
  "request_numbers": ["<any פנייה/בקשה/סעיף budget-request numbers, digits only>"],
  "matter": "<one short Hebrew sentence naming the specific matter, not the general area>"}}

Rules: use ONLY what appears in the excerpt; do not invent. Keep list items short (a few words). \
If a field has nothing, use an empty string or empty list. Do NOT continue or quote the excerpt.

EXCERPT:
<segment>
{text}
</segment>

JSON FINGERPRINT:"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def snippet(text, n=SNIPPET_WORDS):
    return " ".join(text.split()[:n])


def parse_fp(text):
    m = JSON_RE.search(text)
    empty = {"ministry": "", "laws_programs": [], "entities": [], "request_numbers": [],
             "matter": "", "_parse_error": True}
    if not m:
        return empty
    try:
        d = json.loads(m.group(0))
        return {"ministry": str(d.get("ministry", "")),
                "laws_programs": [str(x) for x in d.get("laws_programs", []) or []],
                "entities": [str(x) for x in d.get("entities", []) or []],
                "request_numbers": [str(x) for x in d.get("request_numbers", []) or []],
                "matter": str(d.get("matter", "")), "_parse_error": False}
    except (json.JSONDecodeError, ValueError, TypeError):
        return empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short_name", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    span_lookup = {(s["doc_id"], s["seg_idx"]): s
                   for s in json.load(open(STEP07_OUT / "gold_segments.json", encoding="utf-8"))}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device)
    model.eval()

    prompts = []
    for rec in ids:
        s = span_lookup[(rec["doc_id"], rec["seg_idx"])]
        content = PROMPT.format(text=snippet(s["span_text"]))
        msg = [{"role": "user", "content": content}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    fps = []
    bs = args.batch_size
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2600).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.15, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for text in tok.batch_decode(gen, skip_special_tokens=True):
            fps.append(parse_fp(text))
        if (i // bs + 1) % 20 == 0:
            log.info("  %d/%d", i + len(batch), len(prompts))

    n_err = sum(f.get("_parse_error") for f in fps)
    log.info("Done. %d/%d parse errors", n_err, len(fps))
    # attach alignment keys
    for rec, fp in zip(ids, fps):
        fp["doc_id"] = rec["doc_id"]; fp["seg_idx"] = rec["seg_idx"]; fp["topic"] = rec["topic"]
    json.dump(fps, open(OUT / f"fingerprints_{args.short_name}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / f"fingerprints_{args.short_name}.json")


if __name__ == "__main__":
    main()
