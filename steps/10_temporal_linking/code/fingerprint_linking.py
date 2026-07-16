"""
Step 10 — Family D linking: combine geometric gist with LLM-extracted STABLE
identifiers (laws/programs/entities), accumulated per journal entry.

Streaming, oracle-growth (matches streaming_eval). Per decision, each candidate
entry gets two affinities:
  gist_cos      -- cosine of the span's embedding to the entry centroid (abtt1)
  id_overlap    -- overlap of the span's stable identifiers with the entry's
                   ACCUMULATED stable-identifier set (union over its members).
                   Uses laws_programs + entities (stable); ignores request_numbers
                   (transient -- they change every session, so matching on them
                   would wrongly split true budget-item recurrences).

Two ways to use the identifiers, both swept and compared to the geometry-only winner:
  (i)  additive:  score = gist_cos + beta * id_overlap ; then a distinctiveness
       margin on that combined score.
  (ii) hard gate: geometric margin must pass AND id_overlap(best entry) >= g.

CPU-only, runs after extract_fingerprints.py.
Usage: python steps/10_temporal_linking/code/fingerprint_linking.py --fp dictalm2
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

import streaming_eval as se

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"


def norm_token(s):
    s = re.sub(r"[^\w֐-׿]+", " ", str(s).lower()).strip()
    return re.sub(r"\s+", " ", s)


def id_set(fp):
    toks = set()
    for x in fp.get("laws_programs", []) + fp.get("entities", []):
        t = norm_token(x)
        if len(t) >= 3:
            toks.add(t)
            toks.update(w for w in t.split() if len(w) >= 4)  # word-level too
    return toks


def stream_fp_records(Xt, ids, fps):
    """Per-decision records with gist cosine AND accumulated id-overlap per entry."""
    topics = [r["topic"] for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)

    entry_members, entry_ids, topic_to_entry = [], [], {}
    records = []
    for i in order:
        x = Xt[i]; xin = id_set(fps[i])
        if entry_members:
            cos = np.array([ (lambda M: (M.mean(0)/(np.linalg.norm(M.mean(0))+1e-12)) @ x)(Xt[mem])
                             for mem in entry_members])
            ov = np.array([len(xin & eid) / (len(xin | eid) + 1e-9) for eid in entry_ids])
            t = topics[i]; is_rep = first_seen[t] != i
            records.append({"gold_is_repeat": bool(is_rep), "cos": cos, "ov": ov,
                            "gold_entry": topic_to_entry.get(t, -1) if is_rep else -1})
        t = topics[i]
        if first_seen[t] == i:
            topic_to_entry[t] = len(entry_members); entry_members.append([i]); entry_ids.append(set(xin))
        else:
            k = topic_to_entry[t]; entry_members[k].append(i); entry_ids[k] |= xin
    return records


def sweep_additive(records, betas, margins):
    best, fmr5 = {"f1": -1}, None
    for beta in betas:
        for m in margins:
            tp = fp = fn = tn = 0
            for r in records:
                score = r["cos"] + beta * r["ov"]
                k = int(np.argmax(score))
                srt = np.partition(score, -2) if len(score) >= 2 else np.array([score[k], -1])
                link = (srt[-1] - srt[-2]) >= m
                if link:
                    tp += (r["gold_is_repeat"] and k == r["gold_entry"])
                    fp += not (r["gold_is_repeat"] and k == r["gold_entry"])
                else:
                    fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
            row = {"beta": round(float(beta), 2), "margin": round(float(m), 3), **se.confusion(tp, fp, fn, tn)}
            if row["f1"] > best["f1"]:
                best = row
            if row["false_merge_rate"] <= 0.05 and (fmr5 is None or row["recall"] > fmr5["recall"]):
                fmr5 = row
    return {"best_f1": best, "best_f1_fmr<=5%": fmr5}


def sweep_gate(records, margins, gates):
    best, fmr5 = {"f1": -1}, None
    for m in margins:
        for g in gates:
            tp = fp = fn = tn = 0
            for r in records:
                c = r["cos"]; k = int(np.argmax(c))
                srt = np.partition(c, -2) if len(c) >= 2 else np.array([c[k], -1])
                link = ((srt[-1] - srt[-2]) >= m) and (r["ov"][k] >= g)
                if link:
                    tp += (r["gold_is_repeat"] and k == r["gold_entry"])
                    fp += not (r["gold_is_repeat"] and k == r["gold_entry"])
                else:
                    fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
            row = {"margin": round(float(m), 3), "gate": round(float(g), 3), **se.confusion(tp, fp, fn, tn)}
            if row["f1"] > best["f1"]:
                best = row
            if row["false_merge_rate"] <= 0.05 and (fmr5 is None or row["recall"] > fmr5["recall"]):
                fmr5 = row
    return {"best_f1": best, "best_f1_fmr<=5%": fmr5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", required=True, help="fingerprint short_name (e.g. dictalm2)")
    args = ap.parse_args()

    e5 = np.load(STEP07_OUT / "embeddings" / "e5.npy")
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    fps = json.load(open(OUT / f"fingerprints_{args.fp}.json", encoding="utf-8"))
    assert len(fps) == len(ids)

    Xt = se.transform(e5, "abtt1")   # robust transform (survives streaming honesty)
    recs = stream_fp_records(Xt, ids, fps)

    # id coverage sanity
    nonempty = sum(1 for r in recs if r["ov"].max(initial=0) > 0)
    log.info("decisions with any id-overlap to some entry: %d/%d", nonempty, len(recs))

    results = {}
    geo = se.summarize(se.sweep_margin(stream_records := se.stream_records(Xt, ids, "centroid")))
    results["geometry_only|abtt1|centroid|margin"] = geo
    log.info("geometry-only (abtt1+centroid+margin): best_f1=%.3f (P=%.3f R=%.3f)",
             geo["best_f1"]["f1"], geo["best_f1"]["precision"], geo["best_f1"]["recall"])

    add = sweep_additive(recs, betas=np.linspace(0, 1.0, 11), margins=np.linspace(0, 0.5, 26))
    results[f"additive|abtt1|{args.fp}"] = add
    b = add["best_f1"]
    log.info("additive (gist+beta*id):  best_f1=%.3f (P=%.3f R=%.3f @beta=%.2f,m=%.3f)  fmr5_R=%s",
             b["f1"], b["precision"], b["recall"], b["beta"], b["margin"],
             f'{add["best_f1_fmr<=5%"]["recall"]:.3f}' if add["best_f1_fmr<=5%"] else "n/a")

    gate = sweep_gate(recs, margins=np.linspace(0, 0.4, 21), gates=np.linspace(0, 0.5, 26))
    results[f"gate|abtt1|{args.fp}"] = gate
    b = gate["best_f1"]
    log.info("hard gate (margin & id>=g): best_f1=%.3f (P=%.3f R=%.3f @m=%.3f,g=%.3f)  fmr5_R=%s",
             b["f1"], b["precision"], b["recall"], b["margin"], b["gate"],
             f'{gate["best_f1_fmr<=5%"]["recall"]:.3f}' if gate["best_f1_fmr<=5%"] else "n/a")

    json.dump(results, open(OUT / f"fingerprint_linking_{args.fp}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / f"fingerprint_linking_{args.fp}.json")


if __name__ == "__main__":
    main()
