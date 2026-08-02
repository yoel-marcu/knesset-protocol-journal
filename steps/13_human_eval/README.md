# Step 13 — Human Assessment of Longitudinal Coherence & Faithfulness

The project abstract promises "we evaluate ... the resulting journal end-to-end through
human assessment of longitudinal coherence and faithfulness." Everything built so far
(Step 9's faithfulness/novelty scores, Step 8's LLM verifier) is **LLM-as-judge**, not human
judgment. This step is that missing piece.

## What's being rated

All **22 gold recurring chains** from Step 9 (dictalm2, original prompt — the best-scoring
config, no few-shot contamination), built directly on Step 7's real, independently-annotated
gold linking (not on any model's predicted linking, so summarization quality is isolated from
linking noise, same design choice as Step 9). This is the full recurring subset of the
523-segment gold benchmark — not a sub-sample, so there's no sampling decision to defend.
Chain lengths: 16 pairs, 4 triples, one 5-chain, one 9-chain (58 entries total).

## Files

- `code/sample_chains.py` — builds the packet from Step 9's output (already run).
- `outputs/annotation_packet_readable.txt` — **read this** to see each chain's entries in
  order (opening entry, then each incremental update, with dates).
- `outputs/annotation_packet.json` — **fill in your scores here**, under
  `scores.<your name>.longitudinal_coherence` and `scores.<your name>.faithfulness` for each
  chain (leave `notes` for anything a number can't capture — a hallucinated fact, a decision
  that got dropped, a chain that just repeats itself).
- `code/compute_agreement.py` — run once all three of you have scored every chain; reports
  per-rater means and pairwise Cohen's kappa.

## Rubric

Score each chain as a whole (i.e., read all its entries in order, then give one pair of
scores for the whole chain), 1–5, using `annotation_packet_readable.txt`:

**Longitudinal coherence** — does the sequence of entries read as a single, continuous,
non-contradictory unfolding story about one matter?
- **5** — Clearly the same matter throughout; each update logically follows from what came
  before; no contradictions or unexplained jumps.
- **3** — Recognizably the same matter, but some updates feel disconnected, repetitive, or
  fail to build on prior entries.
- **1** — Reads as unrelated or contradictory entries stitched together; no sense of
  progression.

**Faithfulness** — does each *update* entry (everything after the opening) contain only
content that is (a) actually supported by that meeting's segment and (b) genuinely
incremental, not a restatement of the opening or prior updates?
- **5** — Every update is accurate and adds only new information (decisions, status changes,
  new figures) not already in the journal.
- **3** — Mostly accurate, but re-summarizes prior content to a moderate degree, or contains
  a minor unsupported detail.
- **1** — Fabricates content not present in the source segment, or is essentially a full
  re-summary from scratch rather than an incremental update.

## Running it

```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
python steps/13_human_eval/code/sample_chains.py     # already run; re-run only if Step 9's
                                                       # best config changes
# --- each rater edits outputs/annotation_packet.json by hand ---
python steps/13_human_eval/code/compute_agreement.py  # after all 3 have scored everything
```

## Status

**Not yet scored.** The packet is generated and ready (`outputs/annotation_packet.json`,
`outputs/annotation_packet_readable.txt`); this step needs the three of you to actually read
and rate the 22 chains before it can be written up.
