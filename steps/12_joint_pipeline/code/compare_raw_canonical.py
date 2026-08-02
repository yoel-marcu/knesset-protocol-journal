"""Compare raw vs canonical journal linking: which segments got merged, and whether
the same recurring groups form under both text representations."""
import json
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"


def recurring_groups(journal):
    return [tuple(sorted(e["segments"])) for e in journal if len(e["segments"]) > 1]


def seg_to_group(groups):
    m = {}
    for g in groups:
        for s in g:
            m[s] = g
    return m


def main():
    raw = json.load(open(OUT / "journal_raw.json", encoding="utf-8"))
    canon = json.load(open(OUT / "journal_canonical.json", encoding="utf-8"))

    raw_groups = recurring_groups(raw["journal"])
    canon_groups = recurring_groups(canon["journal"])

    print(f"raw: {len(raw_groups)} recurring groups, canonical: {len(canon_groups)} recurring groups")
    print()
    print("=== RAW recurring groups ===")
    for g in raw_groups:
        print(" ", g)
    print()
    print("=== CANONICAL recurring groups ===")
    for g in canon_groups:
        print(" ", g)

    raw_map = seg_to_group(raw_groups)
    canon_map = seg_to_group(canon_groups)
    all_segs = set(raw_map) | set(canon_map)

    print()
    print("=== differences (segment-level group membership) ===")
    diffs = 0
    for s in sorted(all_segs):
        rg = raw_map.get(s)
        cg = canon_map.get(s)
        if rg != cg:
            diffs += 1
            print(f"  {s}: raw={rg} canonical={cg}")
    if diffs == 0:
        print("  none — identical linking decisions on every segment that appears in a recurring group in either run")

    print()
    print(f"total distinct segments touching a recurring group (raw ∪ canonical): {len(all_segs)}")
    print(f"segments with differing group membership: {diffs}")


if __name__ == "__main__":
    main()
