# CLAUDE.md — ANLP-PROJECT

## IMPORTANT: Project Isolation

This is a **completely separate project** from any other work in the parent directory.
Do NOT reference, assume, or carry over any context from:
- The thesis project (`bayes_al_2025/`, active learning, SLURM experiments that are directly under /cs/labs/daphna/yoel.marcu2003. This project's slurm experiment files will be in this directory.)
- Any other subdirectory in the parent directory
- Any previously saved memories about thesis research, AL strategies, CIFAR, etc.

Treat this as a **fresh, independent project** scoped entirely to ANLP-PROJECT.

---

## Overview

**Project:** Knesset Protocol Journal Project
**Authors:** Tomer Morad, Or Israeli, Yoel Marko (May 2026)
**Course:** Advanced NLP (ANLP)

**Goal:** Build a longitudinal, subject-coherent journal from a stream of Israeli Knesset committee meeting transcripts. The output is a per-committee index where each entry corresponds to a single underlying legislative matter and accumulates a chronological log of progress as the matter recurs across meetings.

### Three Coupled NLP Tasks

1. **Topic and Segment Extraction** — Partition each protocol into subject-specific spans; label each span with a canonical subject title + key entities. Recover coherent subject boundaries in long, multi-speaker Hebrew discourse.

2. **Streaming Subject Linking** — Cross-document event coreference operating online over a growing index. Classify each newly extracted subject as `same` / `related` / `new` relative to existing journal entries. Anti-duplication is the central design constraint: false-merges damage the journal more than false-splits.

3. **Context-Conditioned Log Writing** — Update summarization with retrieval augmentation (RAG). Given the existing journal entry and the new protocol segment, generate only the incremental contribution (new developments, decisions, status changes).

---

## Data

**Location:** `PROTOCOLS/` — ~200 JSON files, each a single Knesset committee meeting.

**Format per file:**
```json
{
  "knesset_num": 25,
  "committee": "ועדת_הכספים",
  "doc_id": "25_ptv_1219729",
  "date": "2022-11-21T12:30:00+02:00",
  "source_file": "https://...",
  "text": "<full Hebrew transcript as a single string>"
}
```

**Language:** Hebrew. Use Hebrew-capable multilingual encoders (e.g., `intfloat/multilingual-e5-large`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, or `onlplab/alephbert-base`).

**Naming convention:** `25_ptv_<doc_id>.json` — prefix `25` = Knesset session 25.

---

## Milestones

1. **Annotated evaluation set** — Manually label a held-out subset with gold subjects per meeting and gold cross-meeting linking decisions. Required before any quantitative evaluation.

2. **Topic and segment extraction** — First-pass segmentation of each protocol into subject spans. Choose simplest approach consistent with the dataset structure (e.g., prompt-based with GPT/Claude, or rule-based on `<< נושא >>` markers).

3. **Streaming subject linking** — Two methods:
   - **(a) Similarity threshold baseline** — Embed subject (title + summary + entities) with Hebrew-capable multilingual encoder; cosine similarity vs. top-1 journal neighbor; threshold = same/new. No LLM.
   - **(b) Retrieve-then-verify** — Top-K dense retrieval to shortlist; LLM verifier classifies same/related/new given candidate's full prior timeline.

4. **Asymmetric evaluation harness** — Precision, recall, F1 with separate accounting for false-merge and false-split errors. Run both linking methods and report the gap.

---

## Project Structure

The project is organized as a **sequence of numbered steps** under `steps/`, so the
progression line is explicit. **Each step is self-contained**: its own `code/`, `outputs/`,
`report/`, and a `README.md`. A later step reads a previous step's `outputs/` as input.

```
ANLP-PROJECT/
├── CLAUDE.md
├── Knesset_Protocol_Journal_Project.pdf   # Project description
├── PROTOCOLS/                             # Raw data (~200 JSON files)
├── steps/                                 # ← project progression, one folder per step
│   ├── 01_topic_preprocessing/
│   │   ├── code/        # preprocess_topics.py
│   │   ├── inputs/      # manual artifacts (topics_corrections.json)
│   │   ├── outputs/     # topics_canonical.json, ...
│   │   ├── report/      # data_summary_report.pdf
│   │   └── README.md
│   └── 02_topic_clustering/
│       ├── code/        # extract_spans.py → embed.py → cluster_eval.py
│       ├── outputs/     # topic_spans.json, embeddings/, clustering_*.{json,png}
│       ├── report/      # clustering_report.pdf
│       └── README.md
├── sbatch/                                # SLURM job submission scripts
├── logs/slurm/                            # SLURM job logs (%j.log)
├── data/                                  # shared raw/auxiliary data (if needed)
└── notebooks/                             # ad-hoc exploration
```

**Rule for new work:** each new step gets its own `steps/NN_<name>/` folder with the
`code/ outputs/ report/ README.md` layout. Do not write results into a shared `outputs/`.

---

## Environment

```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
```

**Key packages:** `transformers`, `datasets`, `torch`, `evaluate`, `accelerate`, `sentence-transformers`

---

## SLURM Cluster Jobs

SLURM scripts live in `sbatch/`. Submit with `sbatch sbatch/my_job.sh`. Logs go to `logs/slurm/`.

### Before submitting — always run first:
```bash
nodes      # shows node states, GPU types, current utilization
slimits    # shows lab budget and current usage
```

### GPU types available:
| Model | Group | Memory | Notes |
|-------|-------|--------|-------|
| l40s  | gg:g4 | 45G | Most available (firefoot nodes) |
| a40   | gg:g4 | 45G | epona nodes |
| a5000 | gg:g0 | 24G | binky, drape |
| a10   | gg:g0 | 22G | ampere, arion |
| l4    | gg:g0 | 23G | hasufel, incitatus |
| h200  | gg:g10| 140G | joey-01, rare |

**Always specify GPU by model or group — never bare `gpu:N`:**
```bash
#SBATCH --gres=gpu:l40s:1   # correct
#SBATCH --gres=gpu:1         # WRONG — will fail
```

### Partition choice:
- `short` / `medium` — uses lab quota (check `slimits`)
- `killable` — uses excess cluster resources beyond quota; jobs can be preempted

### Standard sbatch template:
```bash
#!/bin/bash
#SBATCH --job-name=anlp_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT
python src/my_script.py
```

---

## Course Exercises

Subdirectories may exist under `ANLP-PROJECT/` for standalone ANLP course exercises, unrelated to the Knesset protocols project. If working inside such a subdirectory, treat it as its own isolated scope — do not assume it is part of the journal pipeline.

---

## Key Design Constraints

- **Anti-duplication is asymmetric:** false-merges >> false-splits in cost. Tune thresholds accordingly.
- **Streaming setting:** linking must work online — no look-ahead over future protocols.
- **Hebrew-first:** all text is Hebrew; ASCII-only models will fail silently.
- **Faithfulness is chain-relative:** log writing faithfulness is measured against the chain of prior summaries, not the raw source document.
