"""
Step 10 — Family E: LLM verifier that reads each candidate's FULL ACCUMULATED
TIMELINE, not just its first span.

The project brief's method (b) says "classify given the candidate's full prior
timeline." Step 8's implementation actually showed the verifier only the FIRST
span's 100-word snippet (see build_verifier_inputs.py). This tests the intended
method: each candidate entry is represented by the concatenation of ALL its prior
occurrences' snippets, giving the model the distinctive, matter-specific facts
(bill numbers, entities) that recur across the timeline but may be absent from any
single span.

Self-contained (query building + model run + eval). Oracle-growth journal, same
scoring as Step 8's eval_verifier so it's directly comparable to the baseline and
to the geometric methods in streaming_eval.py.

Usage (GPU):
    python steps/10_temporal_linking/code/timeline_verify.py \
        --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b --batch_size 2
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"

TOP_K = 10
SNIP = 90            # words per occurrence in a timeline
MAX_TL = 500         # cap total words per candidate timeline

SYSTEM = """You judge whether a new discussion segment from an Israeli Knesset committee \
protocol continues the SAME specific legislative matter as one of several candidate journal \
entries. Each candidate is shown as its FULL TIMELINE so far (all prior dated occurrences of \
that matter). Do NOT continue or quote the texts. Reply with ONLY one JSON object:
{"best_matched_rank": <candidate rank int or null>, "same_confidence": <int 0-100>}
A high score means genuinely the same specific matter continuing across the timeline; a low \
score means only the same ministry/format/general area. If null, same_confidence must be 0."""


def abtt(X, k=1):
    from sklearn.preprocessing import normalize
    mu = X.mean(0, keepdims=True); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return normalize(Xc - (Xc @ Vt[:k].T) @ Vt[:k]).astype(np.float32)


def snip(text, n):
    return " ".join(text.split()[:n])


def build_queries():
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    span = {(s["doc_id"], s["seg_idx"]): s
            for s in json.load(open(STEP07_OUT / "gold_segments.json", encoding="utf-8"))}
    X = abtt(np.load(STEP07_OUT / "embeddings" / "e5.npy"))
    topics = [r["topic"] for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)

    entry_members, entry_topic, topic_to_entry = [], [], {}
    queries = []
    for i in order:
        rec = ids[i]; s = span[(rec["doc_id"], rec["seg_idx"])]
        if entry_members:
            cent = np.stack([X[m].mean(0) for m in entry_members])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
            sims = cent @ X[i]
            top = np.argsort(-sims)[:TOP_K]
            t = topics[i]; is_rep = first_seen[t] != i
            correct = topic_to_entry.get(t, -1) if is_rep else -1
            correct_rank = None
            if is_rep and correct in top:
                correct_rank = int(np.where(top == correct)[0][0]) + 1
            cands = []
            for rank, k in enumerate(top, 1):
                # full timeline: concat all member snippets, capped
                tl, used = [], 0
                for m in entry_members[k]:
                    ms = span[(ids[m]["doc_id"], ids[m]["seg_idx"])]
                    w = snip(ms["span_text"], SNIP); used += len(w.split())
                    tl.append(f"({ids[m]['date'][:10]}) {w}")
                    if used >= MAX_TL:
                        break
                cands.append({"rank": rank, "entry": int(k), "timeline": "\n".join(tl),
                              "_gold": (k == correct)})
            queries.append({"qi": int(i), "snippet": snip(s["span_text"], 120),
                            "date": rec["date"][:10], "cands": cands,
                            "gold_is_repeat": bool(is_rep), "gold_rank": correct_rank})
        t = topics[i]
        if first_seen[t] == i:
            topic_to_entry[t] = len(entry_members); entry_members.append([i]); entry_topic.append(t)
        else:
            entry_members[topic_to_entry[t]].append(i)
    return queries


def build_prompt(q):
    lines = [SYSTEM, "", f"NEW SEGMENT (date {q['date']}):", "<segment>", q["snippet"], "</segment>",
             "", "CANDIDATE JOURNAL ENTRIES (each shown as its full timeline):"]
    for c in q["cands"]:
        lines += [f"[{c['rank']}]:", "<timeline>", c["timeline"], "</timeline>"]
    lines += ["", "Output ONLY the JSON verdict now."]
    return "\n".join(lines)


JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse(text):
    m = JSON_RE.search(text)
    if not m:
        return {"rank": None, "conf": 0}
    try:
        d = json.loads(m.group(0)); r = d.get("best_matched_rank")
        r = int(r) if r is not None else None
        c = max(0, min(100, int(d.get("same_confidence", 0))))
        return {"rank": r, "conf": 0 if r is None else c}
    except Exception:
        return {"rank": None, "conf": 0}


def evaluate(queries, preds):
    rows = []
    for theta in range(0, 101, 5):
        tp = fp = fn = tn = 0
        for q, p in zip(queries, preds):
            link = p["rank"] is not None and p["conf"] >= theta
            if link:
                ok = q["gold_is_repeat"] and p["rank"] == q["gold_rank"]
                tp += ok; fp += (not ok)
            else:
                fn += q["gold_is_repeat"]; tn += (not q["gold_is_repeat"])
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fmr = fp / (fp + tn) if fp + tn else 0.0
        rows.append({"theta": theta, "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "false_merge_rate": round(fmr, 4)})
    best = max(rows, key=lambda r: r["f1"])
    ok = [r for r in rows if r["false_merge_rate"] <= 0.05]
    return {"curve": rows, "best_f1": best,
            "best_f1_fmr<=5%": max(ok, key=lambda r: r["recall"]) if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short_name", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    queries = build_queries()
    log.info("Built %d timeline queries", len(queries))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device).eval()

    prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(q)}],
                                       tokenize=False, add_generation_prompt=True) for q in queries]
    preds = []
    bs = args.batch_size
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=5200).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.15, pad_token_id=tok.pad_token_id)
        for text in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            preds.append(parse(text))
        if (i // bs + 1) % 20 == 0:
            log.info("  %d/%d", i + bs, len(prompts))

    res = evaluate(queries, preds)
    json.dump({"model": args.model, **res},
              open(OUT / f"timeline_verify_{args.short_name}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    b = res["best_f1"]
    log.info("timeline-verify %s: best_f1=%.3f (P=%.3f R=%.3f)  fmr5_R=%s", args.short_name,
             b["f1"], b["precision"], b["recall"],
             f'{res["best_f1_fmr<=5%"]["recall"]:.3f}' if res["best_f1_fmr<=5%"] else "n/a")
    log.info("Wrote %s", OUT / f"timeline_verify_{args.short_name}.json")


if __name__ == "__main__":
    main()
