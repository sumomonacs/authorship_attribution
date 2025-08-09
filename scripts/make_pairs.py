import os
import json
import sys
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pairs import generate_contrastive_pairs
from src.config import PAIR_FILE, EXCERPT_FILE

def main():
    df = pd.read_csv(EXCERPT_FILE)

    # 1 000 positives per author  
    pairs = generate_contrastive_pairs(df, pos_per_author=1000, neg_pairs=5000,seed=42)

    # save as a JSON list of dicts
    out_path = Path(PAIR_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"text1": a, "text2": b, "label": lbl, "author1": author1, "author2": author2} for a, b, lbl, author1, author2 in pairs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # quick summary
    n_pos = sum(lbl for _, _, lbl, _, _ in pairs)
    n_neg = len(pairs) - n_pos
    print(f"Saved {len(pairs):,} pairs to {out_path}  (pos: {n_pos:,}  neg: {n_neg:,})")

if __name__ == "__main__":
    main()