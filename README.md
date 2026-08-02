# Knesset Protocol Journal Project

**Tomer Morad, Or Israeli, Yoel Marko** — Advanced NLP course project

Full task description: [`Knesset_Protocol_Journal_Project.pdf`](Knesset_Protocol_Journal_Project.pdf).

## What this is

We build and maintain a longitudinal, subject-coherent journal from a stream of Israeli
Knesset Finance Committee meeting transcripts (`PROTOCOLS/`, ~200 protocols, Hebrew). Each
journal entry corresponds to one underlying legislative matter and accumulates a
chronological log of progress as that matter recurs across meetings. The problem splits
into three coupled tasks:

1. **Topic and segment extraction** — partition each protocol into subject-specific spans,
   label each with a canonical subject title and key entities.
2. **Streaming subject linking** — for each new span, decide online (no look-ahead) whether
   it continues an existing journal entry or starts a new one. Anti-duplication is the
   central constraint: a false merge is far worse than a false split.
3. **Context-conditioned log writing** — given the matched journal entry and the new span,
   generate only the *incremental* contribution (RAG-style, but the retrieved context is the
   journal's own accumulated log, not an external corpus).

## Where to start

- **Read the reports in order** (`steps/NN_*/report/*.pdf`) — each is self-contained and
  builds on the previous one. Steps 01–09 build the core pipeline and its first evaluation;
  Step 10 reframes linking as online clustering (2× the baseline); Steps 11–12 run the
  project's definitive experiment: does canonicalizing text before embedding actually help
  linking, tested end-to-end on the full, independently-annotated 523-segment gold set.
- **`steps/12_joint_pipeline/report/joint_pipeline_report.pdf`** is the most important single
  document if you only read one — it contains the final answer on canonicalization and ties
  the whole pipeline together.

| Step | What it does |
|---|---|
| [01_topic_preprocessing](steps/01_topic_preprocessing/) | Protocol → topic marker extraction, first data pass |
| [02_topic_clustering](steps/02_topic_clustering/) | Span embedding + clustering, first Task 1 pass |
| [03_dense_deanisotrize](steps/03_dense_deanisotrize/) | ABTT whitening to fix embedding anisotropy |
| [04_hybrid_tfidf_dense](steps/04_hybrid_tfidf_dense/) | TF-IDF + dense score fusion |
| [05_linking_baseline](steps/05_linking_baseline/) | Task 2a: similarity-threshold linking baseline |
| [06_annotation](steps/06_annotation/) | Milestone 1: full-protocol gold segmentation (211/213 protocols) |
| [07_gold_eval](steps/07_gold_eval/) | Re-validates Steps 02–05 against real gold (523 segments, 22 recurring topics) |
| [08_retrieve_verify](steps/08_retrieve_verify/) | Task 2b: LLM retrieve-then-verify linker |
| [09_log_writing](steps/09_log_writing/) | Task 3: context-conditioned incremental log writing |
| [10_temporal_linking](steps/10_temporal_linking/) | Linking reframed as online clustering (centroid + margin gate); ~2× baseline F1 |
| [11_canonicalization_linking](steps/11_canonicalization_linking/) | First (caveated) look: does canonicalizing text help linking? |
| [12_joint_pipeline](steps/12_joint_pipeline/) | Full segments→journal pipeline; **definitive raw-vs-canonical test on all 523 gold segments** |
| [13_human_eval](steps/13_human_eval/) | Human assessment protocol for longitudinal coherence & faithfulness — packet ready, not yet scored |

## Headline results

- **Milestone 4 (asymmetric linking gap)**: the proposed retrieve-then-verify LLM linker
  (Step 08) did **not** beat the plain similarity-threshold baseline (Step 05/07) on the real
  gold set — F1 0.111 vs 0.125. A genuine, documented negative result.
- **Streaming linking as online clustering** (Step 10) nearly doubles the baseline: accumulated
  centroid + margin gate reaches F1 ≈ 0.19–0.21 (honest, streaming-fit), 4× the baseline recall
  at a matched false-merge rate.
- **Canonicalization does not help, and modestly hurts, linking** (Steps 11→12): an initial,
  caveated test (Step 11, small circular pseudo-gold) suggested a possible precision gain from
  rewriting protocol text into neutral third-person Hebrew before embedding. The definitive
  re-run on the full, independently-annotated 523-segment gold set (Step 12) reverses this:
  canonical text gets **worse** nearest-neighbor recall on true recurring subjects (recall@10
  drops 94.8% → 86.2%, concentrated in the common two-occurrence case) and does not show
  cleaner merge precision on manual inspection either.

## What's not yet done

- **Human assessment of longitudinal coherence and faithfulness** of the end-to-end journal —
  promised in the project abstract, not yet run (see `steps/13_human_eval/` for the protocol
  and instructions).

## Environment

```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
```

Key packages: `transformers`, `torch`, `sentence-transformers`, `scikit-learn`. SLURM jobs
live in `sbatch/`, submitted with `sbatch sbatch/<job>.sh`; logs go to `logs/slurm/`. See
`CLAUDE.md` for cluster/GPU details.

## Data

`PROTOCOLS/` — ~200 Knesset Finance Committee meeting transcripts, one JSON file per
protocol, Hebrew. See `CLAUDE.md` for the exact schema.
