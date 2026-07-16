"""
Step 10 — figures for the report. Recomputes the needed curves from the frozen
embeddings + saved fingerprints and writes PNGs to outputs/.
Usage: python steps/10_temporal_linking/code/make_plots.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streaming_eval as se
from fingerprint_linking import stream_fp_records

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07 = STEP_DIR.parent / "07_gold_eval" / "outputs"

BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#c53030"

e5 = np.load(STEP07 / "embeddings" / "e5.npy")
ids = json.load(open(STEP07 / "embeddings" / "ids.json", encoding="utf-8"))
fps = json.load(open(OUT / "fingerprints_dictalm2.json", encoding="utf-8"))


def pr_points(rows, xkey="recall", ykey="precision"):
    pts = sorted([(r[xkey], r[ykey]) for r in rows])
    return [p[0] for p in pts], [p[1] for p in pts]


# ── curves ───────────────────────────────────────────────────────────────────
base_rows = se.sweep_threshold(se.stream_records(se.transform(e5, "abtt1"), ids, "first"))
cent_rows = se.sweep_threshold(se.stream_records(se.transform(e5, "center"), ids, "centroid"))
geo_rows = se.sweep_margin(se.stream_records(se.transform(e5, "center"), ids, "centroid"))
recs_fp = stream_fp_records(se.transform(e5, "center"), ids, fps)
# additive fp curve at beta=0.7, sweeping margin
fp_rows = []
for m in np.linspace(0, 0.5, 51):
    tp = fp = fn = tn = 0
    for r in recs_fp:
        score = r["cos"] + 0.7 * r["ov"]; k = int(np.argmax(score))
        srt = np.partition(score, -2) if len(score) >= 2 else np.array([score[k], -1])
        link = (srt[-1] - srt[-2]) >= m
        if link:
            ok = r["gold_is_repeat"] and k == r["gold_entry"]; tp += ok; fp += (not ok)
        else:
            fn += r["gold_is_repeat"]; tn += (not r["gold_is_repeat"])
    fp_rows.append(se.confusion(tp, fp, fn, tn))


# ── Fig 1: leaderboard ────────────────────────────────────────────────────────
methods = ["baseline\n(first-span + threshold)", "Step 8 best\nLLM verifier",
           "Family E\nLLM verifier (full timeline)", "geometry\n(centroid + margin)",
           "geometry + LLM\nfingerprints"]
f1s = [0.103, 0.111, 0.068, 0.190, 0.211]
colors = [GREY, GREY, GREY, BLUE, GREEN]
fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(methods))
ax.barh(y, f1s, color=colors)
ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Best F1 (streaming linking, gold eval)")
ax.axvline(0.103, color=RED, ls="--", lw=1, label="baseline")
for yi, v in zip(y, f1s):
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 0.25); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Geometry beats every LLM verifier; LLM helps only as an extractor", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_leaderboard.png", dpi=150); plt.close(fig)


# ── Fig 2: precision–recall ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.2))
for rows, lab, col in [(base_rows, "baseline (first-span + threshold)", GREY),
                       (geo_rows, "geometry (centroid + margin)", BLUE),
                       (fp_rows, "geometry + LLM fingerprints", GREEN)]:
    xs, ys = pr_points(rows)
    ax.plot(xs, ys, "-o", ms=3, color=col, label=lab)
ax.set_xlabel("Recall (fraction of 36 true repeats caught)")
ax.set_ylabel("Precision (1 − false-merge share of links)")
ax.set_title("Precision–recall: the accumulated-topic geometry\ndominates the frozen first-span baseline", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_pr_curves.png", dpi=150); plt.close(fig)


# ── Fig 3: WHY the margin gate works — margin distribution by ground truth ─────
recs = se.stream_records(se.transform(e5, "center"), ids, "centroid")
mrg_rep, mrg_new = [], []
for r in recs:
    c = r["cos"]
    if len(c) < 2:
        continue
    srt = np.partition(c, -2); margin = srt[-1] - srt[-2]
    (mrg_rep if r["gold_is_repeat"] else mrg_new).append(margin)
fig, ax = plt.subplots(figsize=(6, 3.8))
bins = np.linspace(0, 0.6, 31)
ax.hist(mrg_new, bins=bins, density=True, alpha=0.6, color=GREY, label=f"should be NEW (n={len(mrg_new)})")
ax.hist(mrg_rep, bins=bins, density=True, alpha=0.6, color=GREEN, label=f"should LINK (n={len(mrg_rep)})")
ax.axvline(0.21, color=RED, ls="--", lw=1, label="margin gate")
ax.set_xlabel("distinctiveness margin  (top-1 − top-2 cosine)")
ax.set_ylabel("density")
ax.set_title("Why the margin gate works: true repeats are DISTINCTLY\nclosest to one entry; boilerplate 'new' spans are not", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_margin_hist.png", dpi=150); plt.close(fig)


# ── Fig 4: the recall fix (first-span vs centroid) ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.8))
for rows, lab, col in [(base_rows, "frozen first-span", GREY),
                       (cent_rows, "accumulated centroid", BLUE)]:
    xs, ys = pr_points(rows, xkey="theta", ykey="recall")
    ax.plot(xs, ys, "-", color=col, label=lab)
ax.set_xlabel("cosine threshold θ")
ax.set_ylabel("recall (true repeats caught)")
ax.set_title("Representing a topic by its accumulated centroid instead of a\nfrozen first span lifts recall from ~32% to ~89%", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_recall_fix.png", dpi=150); plt.close(fig)

print("wrote fig_leaderboard, fig_pr_curves, fig_margin_hist, fig_recall_fix")
