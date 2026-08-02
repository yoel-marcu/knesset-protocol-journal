"""Build the human-annotation packet: all 22 gold recurring journal chains (Step 9's
best config, dictalm2, no few-shot) with blank score fields for 3 raters.

We use ALL 22 recurring topics rather than a sub-sample -- it's a small, complete,
non-arbitrary set (the full recurring subset of the 523-segment gold benchmark), so
there's no sampling decision to defend and every rater sees the same evidence used
throughout Steps 7-12.
"""
import json
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parents[1]
SRC = STEP_DIR.parents[0] / "09_log_writing" / "outputs" / "generated_logs_dictalm2.json"
OUT = STEP_DIR / "outputs"

RATERS = ["tomer", "or", "yoel"]


def main():
    chains = json.load(open(SRC, encoding="utf-8"))
    packet = []
    for i, chain in enumerate(chains):
        packet.append({
            "chain_id": i,
            "topic": chain["topic"],
            "n_entries": len(chain["entries"]),
            "entries": [
                {"date": e["date"][:10], "is_opening": e.get("is_opening", False), "text": e["text"]}
                for e in chain["entries"]
            ],
            "scores": {
                r: {"longitudinal_coherence": None, "faithfulness": None, "notes": ""}
                for r in RATERS
            },
        })

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(packet, open(OUT / "annotation_packet.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # readable rendering for actually reading the entries while scoring
    lines = []
    for c in packet:
        lines.append(f"{'='*70}")
        lines.append(f"CHAIN {c['chain_id']}  ({c['n_entries']} entries)")
        lines.append(f"Topic: {c['topic']}")
        lines.append(f"{'='*70}")
        for j, e in enumerate(c["entries"]):
            kind = "OPENING" if e["is_opening"] else f"UPDATE {j}"
            lines.append(f"\n--- {kind} | {e['date']} ---")
            lines.append(e["text"])
        lines.append("\n")
    open(OUT / "annotation_packet_readable.txt", "w", encoding="utf-8").write("\n".join(lines))

    print(f"wrote {len(packet)} chains ({sum(c['n_entries'] for c in packet)} total entries) to:")
    print(f"  {OUT / 'annotation_packet.json'} (fill in scores here)")
    print(f"  {OUT / 'annotation_packet_readable.txt'} (read entries here)")


if __name__ == "__main__":
    main()
