"""Recall@k analysis: for every segment belonging to a gold recurring chain (chain
length >= 2), check whether at least one true chain-mate falls within its top-k
nearest neighbors by e5 cosine similarity. Compare raw text vs canonical text.
Also reports per-chain-length breakdown (chains are NOT just pairs -- up to length 9).
"""
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
GOLD_DIR = STEP_DIR.parents[0] / "07_gold_eval" / "outputs"

K_VALUES = [1, 3, 5, 10]


def load_gold_recurring():
    gold = json.load(open(GOLD_DIR / "gold_chains.json", encoding="utf-8"))
    recurring = {topic: v for topic, v in gold.items() if len(v) > 1}
    seg_to_topic = {}
    for topic, spans in recurring.items():
        for s in spans:
            key = f"{s['doc_id']}__{s['seg_idx']}"
            seg_to_topic[key] = topic
    return recurring, seg_to_topic


def recall_at_k(keys, seg_to_topic, embeddings):
    """embeddings: dict seg_key -> vector (already normalized)."""
    vecs = np.stack([embeddings[k] for k in keys])
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1.0)  # exclude self

    hits = {k: 0 for k in K_VALUES}
    n = len(keys)
    for i, key in enumerate(keys):
        topic = seg_to_topic[key]
        true_mates = {j for j, k2 in enumerate(keys) if k2 != key and seg_to_topic[k2] == topic}
        if not true_mates:
            continue
        order = np.argsort(-sims[i])
        for k in K_VALUES:
            topk = set(order[:k].tolist())
            if topk & true_mates:
                hits[k] += 1
    return {k: hits[k] / n for k in K_VALUES}, n


def main():
    recurring, seg_to_topic = load_gold_recurring()
    keys = sorted(seg_to_topic.keys())
    log.info("Recurring gold segments: %d across %d topics (chain lengths: %s)",
              len(keys), len(recurring), sorted(len(v) for v in recurring.values()))

    raw_segs = {r["seg_key"]: r["text"] for r in json.load(open(OUT / "segments_raw.json", encoding="utf-8"))}
    canon_segs = {r["seg_key"]: r["text"] for r in json.load(open(OUT / "segments_canonical.json", encoding="utf-8"))}

    missing = [k for k in keys if k not in raw_segs or k not in canon_segs]
    if missing:
        log.warning("Missing segments (skipped): %s", missing)
        keys = [k for k in keys if k not in missing]

    log.info("Loading multilingual-e5-large ...")
    enc = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")

    def embed(texts):
        vecs = enc.encode([f"query: {t}" for t in texts], batch_size=32,
                           show_progress_bar=False, normalize_embeddings=True)
        return vecs

    raw_vecs = embed([raw_segs[k] for k in keys])
    canon_vecs = embed([canon_segs[k] for k in keys])
    raw_emb = {k: v for k, v in zip(keys, raw_vecs)}
    canon_emb = {k: v for k, v in zip(keys, canon_vecs)}

    raw_recall, n = recall_at_k(keys, seg_to_topic, raw_emb)
    canon_recall, _ = recall_at_k(keys, seg_to_topic, canon_emb)

    print(f"\nRecall@k (fraction of the {n} recurring gold segments whose true chain-mate\n"
          f"appears in the top-k nearest neighbors by e5 cosine similarity):\n")
    print(f"{'k':>4} {'raw':>8} {'canonical':>10}")
    for k in K_VALUES:
        print(f"{k:>4} {raw_recall[k]:>8.3f} {canon_recall[k]:>10.3f}")

    # per-chain-length breakdown, k=10
    print("\nPer chain-length breakdown (k=10):")
    by_len = {}
    for topic, spans in recurring.items():
        chain_keys = [f"{s['doc_id']}__{s['seg_idx']}" for s in spans if f"{s['doc_id']}__{s['seg_idx']}" in keys]
        if len(chain_keys) < 2:
            continue
        by_len.setdefault(len(chain_keys), []).append((topic, chain_keys))

    for length in sorted(by_len):
        chains = by_len[length]
        raw_hits = canon_hits = total = 0
        for topic, chain_keys in chains:
            for key in chain_keys:
                total += 1
                mates = set(chain_keys) - {key}
                mate_idx = {keys.index(m) for m in mates}

                sims_raw = np.array([raw_emb[key] @ raw_emb[k2] if k2 != key else -1 for k2 in keys])
                order_raw = np.argsort(-sims_raw)[:10]
                if set(order_raw.tolist()) & mate_idx:
                    raw_hits += 1

                sims_canon = np.array([canon_emb[key] @ canon_emb[k2] if k2 != key else -1 for k2 in keys])
                order_canon = np.argsort(-sims_canon)[:10]
                if set(order_canon.tolist()) & mate_idx:
                    canon_hits += 1
        print(f"  chain_len={length:>2}  n_chains={len(chains):>2}  n_segments={total:>3}  "
              f"raw_recall@10={raw_hits/total:.3f}  canon_recall@10={canon_hits/total:.3f}")


if __name__ == "__main__":
    main()
