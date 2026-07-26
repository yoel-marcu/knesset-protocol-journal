"""
Step 12 — the JOINT PIPELINE: segments -> embed -> streaming link (Step 10) ->
context-conditioned log writing (Step 9) -> a finished per-topic journal.

This unifies the two halves of the project. It consumes a segments file that can
be EITHER our raw gold segments OR their canonicalized rewrites (Tomer's upstream),
selected with --segments, so the same command produces the journal both ways and
the raw-vs-canonical effect is measured end-to-end, not just on an intrinsic metric.

Stages (all in one job):
  1. EMBED every segment's text with multilingual-e5-large (GPU).
  2. LINK online with the Step-10 method (ABTT-k1 + accumulated centroid + margin
     gate at a fixed operating point): each segment joins the best existing journal
     entry if its distinctiveness margin clears theta, else opens a new entry.
  3. LOG-WRITE per entry with the Step-9 method (dictalm2): an opening summary for
     the entry's first segment, then an incremental update per later segment,
     conditioned on the running journal (GPU).
  4. WRITE journal.json.

Input segments file: list of {seg_key, date, text} (see export_for_canon.py /
build_raw_segments.py). Output: outputs/journal_<tag>.json.

Usage (GPU, one job):
    python steps/12_joint_pipeline/code/run_journal.py \
        --segments outputs/segments_raw.json --tag raw \
        --theta 0.34 --logwrite_model dicta-il/dictalm2.0-instruct
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

E5 = "intfloat/multilingual-e5-large"
SNIPPET_WORDS = 700

OPENING_PROMPT = """You are opening a journal entry for a legislative matter from an Israeli \
Knesset Finance Committee session. Write a concise opening entry (2-4 sentences, Hebrew) of \
what this matter is about, based ONLY on the text. Respond with ONLY the entry text.

SESSION TEXT (date {date}):
<segment>
{text}
</segment>

JOURNAL ENTRY:"""

UPDATE_PROMPT = """You maintain a chronological journal entry for a legislative matter across \
Knesset Finance Committee sessions. Below is the EXISTING JOURNAL and a NEW session on the same \
matter. Write ONLY the new incremental update (1-3 sentences, Hebrew): new developments, \
decisions, status changes. Do NOT repeat what's already logged; do NOT invent. Respond with \
ONLY the update text.

EXISTING JOURNAL:
{journal}

NEW SESSION TEXT (date {date}):
<segment>
{text}
</segment>

NEW INCREMENTAL UPDATE:"""


def abtt(X, k=1):
    mu = X.mean(0, keepdims=True); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return normalize(Xc - (Xc @ Vt[:k].T) @ Vt[:k])


def snippet(t, n=SNIPPET_WORDS):
    return " ".join(t.split()[:n])


def link(Xt, rows, theta):
    """Step-10 online linking: centroid + margin gate. Returns list of entries
    (each a list of row indices in chronological order)."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["date"], rows[i]["seg_key"]))
    entries = []          # list of member-index lists
    assign = {}
    for i in order:
        x = Xt[i]
        if entries:
            cos = np.array([(lambda M: (M.mean(0)/(np.linalg.norm(M.mean(0))+1e-12)) @ x)(Xt[m])
                            for m in entries])
            k = int(np.argmax(cos))
            srt = np.partition(cos, -2) if len(cos) >= 2 else np.array([cos[k], -1])
            if (srt[-1] - srt[-2]) >= theta:
                entries[k].append(i); assign[i] = k; continue
        entries.append([i]); assign[i] = len(entries) - 1
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--theta", type=float, default=0.34)
    ap.add_argument("--logwrite_model", default="dicta-il/dictalm2.0-instruct")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--no_logwrite", action="store_true", help="link only, skip log writing")
    args = ap.parse_args()

    rows = json.load(open(args.segments if Path(args.segments).is_absolute()
                          else STEP_DIR / args.segments, encoding="utf-8"))
    log.info("Loaded %d segments (%s)", len(rows), args.tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. EMBED
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(E5, device=device); enc.max_seq_length = 512
    emb = enc.encode([f"query: {r['text']}" for r in rows], batch_size=32,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float64)
    del enc; torch.cuda.empty_cache()
    Xt = abtt(emb, 1)

    # 2. LINK
    entries = link(Xt, rows, args.theta)
    n_multi = sum(len(e) >= 2 for e in entries)
    log.info("Linked into %d journal entries (%d recurring, %d singletons) @theta=%.2f",
             len(entries), n_multi, len(entries) - n_multi, args.theta)

    # 3. LOG-WRITE
    journal = []
    if args.no_logwrite:
        for eid, mem in enumerate(entries):
            journal.append({"entry_id": eid,
                            "segments": [rows[i]["seg_key"] for i in mem],
                            "dates": [rows[i]["date"] for i in mem]})
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.logwrite_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.logwrite_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device).eval()

        def gen(prompt):
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            e = tok(text, return_tensors="pt", truncation=True, max_length=6000).to(device)
            with torch.no_grad():
                o = model.generate(**e, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   repetition_penalty=1.15, pad_token_id=tok.pad_token_id)
            r = tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            for stop in ["\nSESSION", "\nNEW SESSION", "\nEXISTING JOURNAL", "\n<segment>"]:
                j = r.find(stop)
                if j != -1:
                    r = r[:j]
            return r.strip()

        for eid, mem in enumerate(entries):
            log_entries = []
            for pos, i in enumerate(mem):
                r = rows[i]
                if pos == 0:
                    txt = gen(OPENING_PROMPT.format(date=r["date"], text=snippet(r["text"])))
                else:
                    js = "\n".join(f"[{le['date']}] {le['text']}" for le in log_entries)
                    txt = gen(UPDATE_PROMPT.format(journal=js, date=r["date"], text=snippet(r["text"])))
                log_entries.append({"date": r["date"], "seg_key": r["seg_key"], "text": txt})
            journal.append({"entry_id": eid,
                            "segments": [rows[i]["seg_key"] for i in mem],
                            "log": log_entries})
            if (eid + 1) % 50 == 0:
                log.info("  logged %d/%d entries", eid + 1, len(entries))

    out = OUT / f"journal_{args.tag}.json"
    json.dump({"tag": args.tag, "theta": args.theta, "n_segments": len(rows),
               "n_entries": len(entries), "n_recurring": n_multi, "journal": journal},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
