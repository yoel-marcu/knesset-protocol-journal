"""
Step 11 — embed raw and canonical segment text with multilingual-e5-large
(the representation Step 10 uses), so the two are directly comparable.

Input:  outputs/canon_dataset.json
Output: outputs/emb_raw.npy, outputs/emb_canonical.npy  (aligned to the dataset order)
Usage (GPU): python steps/11_canonicalization_linking/code/embed_variants.py
"""
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
MODEL = "intfloat/multilingual-e5-large"
MAXLEN = 512


def main():
    from sentence_transformers import SentenceTransformer
    rows = json.load(open(OUT / "canon_dataset.json", encoding="utf-8"))
    log.info("Embedding %d segments x 2 variants with %s", len(rows), MODEL)
    model = SentenceTransformer(MODEL)
    model.max_seq_length = MAXLEN
    for field, out in [("raw", "emb_raw.npy"), ("canonical", "emb_canonical.npy")]:
        texts = [f"query: {r[field]}" for r in rows]   # e5 convention
        emb = model.encode(texts, batch_size=32, normalize_embeddings=False,
                           show_progress_bar=False, convert_to_numpy=True)
        np.save(OUT / out, emb.astype(np.float32))
        log.info("Wrote %s  shape=%s", OUT / out, emb.shape)


if __name__ == "__main__":
    main()
