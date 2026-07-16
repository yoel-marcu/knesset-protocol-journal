"""
Step 09c — LLM-judge evaluation of generated journal entries on two axes:
  faithfulness -- is every claim in the entry actually supported by its source
                  session text (no hallucination)?
  novelty      -- for incremental updates only: does it add genuinely new
                  information beyond what's already in the existing journal, or
                  just repeat prior content? (opening entries have nothing prior
                  to repeat, so novelty is not scored for them.)

No human-written reference summaries exist for this corpus, so faithfulness/
novelty are judged by an LLM against the same two things the generator itself
saw (source text + prior journal) -- consistent with "faithfulness is
chain-relative" (CLAUDE.md): judged against the summary chain, not by re-deriving
ground truth from every prior session's raw transcript.

Input:  outputs/journal_chains.json, outputs/generated_logs_<short_name>.json
Output: outputs/faithfulness_scores_<short_name>.json, outputs/faithfulness_summary.json
Usage (GPU):
    python steps/09_log_writing/code/eval_faithfulness.py \
        --judge_model Qwen/Qwen2.5-7B-Instruct --targets dictalm2 qwen7b
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

SNIPPET_WORDS = 700

JUDGE_PROMPT_OPENING = """You are evaluating a journal opening-entry written about an Israeli \
Knesset committee session, for faithfulness only.

SOURCE SESSION TEXT:
<segment>
{source}
</segment>

CANDIDATE OPENING ENTRY:
<entry>
{entry}
</entry>

Score faithfulness 0-100: is every claim in the candidate entry actually supported by the \
source text above? 100 = fully grounded, no invented details. 0 = contains claims with no \
support in the source at all.

Respond with ONLY JSON: {{"faithfulness": <int 0-100>}}"""

JUDGE_PROMPT_UPDATE = """You are evaluating a journal incremental-update sentence written about \
an Israeli Knesset committee session, for faithfulness and novelty.

EXISTING JOURNAL (written before this update):
{journal}

NEW SOURCE SESSION TEXT (what the update is supposed to be based on):
<segment>
{source}
</segment>

CANDIDATE UPDATE (to evaluate):
<entry>
{entry}
</entry>

Score two things 0-100:
- "faithfulness": is every claim in the candidate update actually supported by the NEW SOURCE \
SESSION TEXT above? 100 = fully grounded, no invented details. 0 = contains claims with no \
support in the source at all.
- "novelty": does the candidate update add genuinely NEW information beyond what's already in \
the EXISTING JOURNAL, or does it just restate/repeat prior content? 100 = entirely new \
substantive content. 0 = pure repetition, adds nothing beyond the existing journal.

Respond with ONLY JSON: {{"faithfulness": <int 0-100>, "novelty": <int 0-100>}}"""


def snippet(text: str, n: int = SNIPPET_WORDS) -> str:
    return " ".join(text.split()[:n])


JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_scores(text: str, keys: list[str]) -> dict:
    m = JSON_RE.search(text)
    out = {k: 0 for k in keys}
    out["_parse_error"] = True
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
        for k in keys:
            out[k] = max(0, min(100, int(d.get(k, 0))))
        out["_parse_error"] = False
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model", required=True, help="HF model id used as judge")
    ap.add_argument("--judge_short_name", required=True, help="tag for the judge in output filenames")
    ap.add_argument("--targets", nargs="+", required=True,
                     help="short_names of generate_logs.py outputs to evaluate")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    chains = {c["topic"]: c for c in json.load(open(OUT / "journal_chains.json", encoding="utf-8"))}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading judge %s on %s", args.judge_model, device)
    tok = AutoTokenizer.from_pretrained(args.judge_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    def judge(prompt: str, keys: list[str]) -> dict:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=6000).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=1.15, pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        raw = tok.batch_decode(gen, skip_special_tokens=True)[0]
        return parse_scores(raw, keys)

    summary_path = OUT / "faithfulness_summary.json"
    summary = {}
    if summary_path.exists():
        prior = json.load(open(summary_path, encoding="utf-8"))
        summary.update(prior.get("results", {}))
        log.info("Merging into existing summary (%d prior entries: %s)",
                 len(summary), list(summary.keys()))

    for target in args.targets:
        gen_path = OUT / f"generated_logs_{target}.json"
        if not gen_path.exists():
            log.warning("Skipping %s: %s not found", target, gen_path)
            continue
        generated = json.load(open(gen_path, encoding="utf-8"))

        scored = []
        for chain_result in generated:
            topic = chain_result["topic"]
            src_chain = chains.get(topic)
            if src_chain is None:
                continue
            src_occs = src_chain["occurrences"]
            entries = chain_result["entries"]

            for i, e in enumerate(entries):
                source_text = snippet(src_occs[i]["span_text"])
                if e.get("is_opening"):
                    scores = judge(JUDGE_PROMPT_OPENING.format(source=source_text, entry=e["text"]),
                                    ["faithfulness"])
                else:
                    prior_journal = "\n".join(f"[{p['date'][:10]}] {p['text']}" for p in entries[:i])
                    scores = judge(
                        JUDGE_PROMPT_UPDATE.format(journal=prior_journal, source=source_text, entry=e["text"]),
                        ["faithfulness", "novelty"])
                scored.append({"topic": topic, "idx": i, "is_opening": e.get("is_opening", False),
                                "entry": e["text"], **scores})

        out_path = OUT / f"faithfulness_scores_{target}.json"
        json.dump(scored, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log.info("Wrote %s (%d entries scored)", out_path, len(scored))

        faith = [s["faithfulness"] for s in scored]
        novelty = [s["novelty"] for s in scored if not s["is_opening"]]
        n_err = sum(1 for s in scored if s.get("_parse_error"))
        summary[target] = {
            "n_entries": len(scored),
            "mean_faithfulness": round(sum(faith) / len(faith), 1) if faith else None,
            "mean_novelty_updates_only": round(sum(novelty) / len(novelty), 1) if novelty else None,
            "n_parse_errors": n_err,
        }
        log.info("%s: mean_faithfulness=%.1f mean_novelty=%.1f parse_errors=%d/%d",
                 target, summary[target]["mean_faithfulness"] or -1,
                 summary[target]["mean_novelty_updates_only"] or -1, n_err, len(scored))

    json.dump({"judge_model": args.judge_model, "results": summary},
              open(summary_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
