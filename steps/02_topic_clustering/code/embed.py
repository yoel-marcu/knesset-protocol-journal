"""
Stage 2 (dense) of the topic-clustering pipeline.

Encodes each clean per-topic span with three Hebrew-capable encoders, comparing them:
  * intfloat/multilingual-e5-large            (e5)       -- contrastive, prefix "passage: "
  * paraphrase-multilingual-mpnet-base-v2     (mpnet)    -- contrastive sentence-transformer
  * imvladikon/alephbertgimmel-base-512       (alephbert)-- Hebrew BERT, mean-pooled

Long spans (median ~5k words) exceed the 512-token window, so each span is split into
word chunks, every chunk is encoded, and the chunk vectors are length-weighted mean-pooled
into one span vector (then L2-normalized).

Outputs (one per model): outputs/embeddings/<name>.npy  -- float32 [N, d]
Plus a shared          : outputs/embeddings/ids.json     -- [{doc_id, topic}] row order

Usage:
    python src/clustering/embed.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]      # steps/02_topic_clustering
SPANS_FILE = STEP_DIR / "outputs" / "topic_spans.json"
EMB_DIR = STEP_DIR / "outputs" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_WORDS = 350   # ~512 Hebrew subword tokens, conservative
BATCH = 32

MODELS = {
    "e5": {
        "hf": "intfloat/multilingual-e5-large",
        "prefix": "passage: ",
        "wrap": False,
    },
    "mpnet": {
        "hf": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "prefix": "",
        "wrap": False,
    },
    "alephbert": {
        "hf": "imvladikon/alephbertgimmel-base-512",
        "prefix": "",
        "wrap": True,    # plain HF model -> wrap with mean pooling
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
    """Return [N, d] L2-normalized span vectors via length-weighted chunk pooling."""
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
    log.info("Embedding %d clean spans", len(spans))

    ids = [{"doc_id": s["doc_id"], "topic": s["topic"]} for s in spans]
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
