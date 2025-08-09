'''
    This file evluates the model.
'''
import argparse
from pathlib import Path
from pprint import pprint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# after the path tweak we can import normally
from src.evaluate import evaluate  # evaluate.py lives at project root


def main():
    metrics = evaluate(
        ckpt_path="models/best.pt",
        pairs_json="data/pairs.json",
        plots=True,     
        refs_json="data/author_refs.json",                    
        tsne_out="eval_plots/tsne_all_authors.png",
    )
    print("\nEval metrics:")
    for k, v in metrics.items():
        print(f"  {k:>9}: {v:.4f}" if isinstance(v, float) else f"  {k:>9}: {v}")

if __name__ == "__main__":
    main()