"""
Step 07b — Embed gold spans (steps/07_gold_eval/outputs/gold_segments.json).

Gold segment boundaries differ from the marker-derived spans in Step 02, so
embeddings must be recomputed; Step 02's vectors are not reusable here.

Only e5 and alephbert are computed: mpnet was excluded from every winning
representation in Steps 03-05 (never best on ARI, AUC, or the hybrid), and
gold data is a confirmatory re-eval, not a new representation sweep.

Outputs: outputs/embeddings/{e5,alephbert}.npy  -- float32 [N, d]
         outputs/embeddings/ids.json             -- [{doc_id, topic, seg_idx, date}] row order

Usage (GPU):
    python steps/07_gold_eval/code/embed_gold.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]      # steps/07_gold_eval
SPANS_FILE = STEP_DIR / "outputs" / "gold_segments.json"
EMB_DIR = STEP_DIR / "outputs" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_WORDS = 350
BATCH = 32

MODELS = {
    "e5": {
        "hf": "intfloat/multilingual-e5-large",
        "prefix": "passage: ",
        "wrap": False,
    },
    "alephbert": {
        "hf": "imvladikon/alephbertgimmel-base-512",
        "prefix": "",
        "wrap": True,
    },
}


def load_model(cfg, device):
    from sentence_transformers import SentenceTransformer, models
    if cfg["wrap"]:
        word = models.Transformer(cfg["hf"], max_seq_length=512)
        pool = models.Pooling(word.get_word_embedding_dimension(),
                              pooling_mode_mean_tokens=True)
        model = SentenceTransformer(modules=[word, pool], device=device)
    else:
        model = SentenceTransformer(cfg["hf"], device=device)
    model.max_seq_length = 512
    return model


def chunk_words(text: str, n: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def embed_spans(model, spans, prefix):
    vecs = np.zeros((len(spans), model.get_sentence_embedding_dimension()), dtype=np.float32)
    for i, span in enumerate(spans):
        chunks = chunk_words(span["span_text"])
        weights = np.array([max(len(c.split()), 1) for c in chunks], dtype=np.float32)
        texts = [prefix + c for c in chunks]
        chunk_vecs = model.encode(
            texts, batch_size=BATCH, convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=False,
        )
        pooled = (chunk_vecs * weights[:, None]).sum(0) / weights.sum()
        norm = np.linalg.norm(pooled)
        vecs[i] = pooled / norm if norm > 0 else pooled
        if (i + 1) % 50 == 0:
            log.info("  encoded %d/%d spans", i + 1, len(spans))
    return vecs


def main():
    spans_all = json.load(open(SPANS_FILE, encoding="utf-8"))
    spans = [s for s in spans_all if s["clean_for_eval"]]
    log.info("Embedding %d gold spans", len(spans))

    ids = [{"doc_id": s["doc_id"], "topic": s["topic"], "seg_idx": s["seg_idx"],
            "date": s["date"]} for s in spans]
    json.dump(ids, open(EMB_DIR / "ids.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    for name, cfg in MODELS.items():
        log.info("=== %s (%s) ===", name, cfg["hf"])
        model = load_model(cfg, device)
        vecs = embed_spans(model, spans, cfg["prefix"])
        np.save(EMB_DIR / f"{name}.npy", vecs)
        log.info("Saved %s.npy  shape=%s", name, vecs.shape)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info("Done. Embeddings in %s", EMB_DIR)


if __name__ == "__main__":
    main()
