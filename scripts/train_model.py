import argparse
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import fit

p = argparse.ArgumentParser(description="Train Siamese Model without leakage")
p.add_argument("--pairs", required=True, help="Path to pairs.json")
p.add_argument("--val-ratio", type=float, default=0.2,
               help="Fraction of data to use for validation (0–1)")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--freeze", type=int, default=0)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--grad-accum", type=int, default=1)
p.add_argument("--out", default="models")
args = p.parse_args()

ckpt = fit(
    pairs_json=args.pairs,
    val_ratio=args.val_ratio,
    epochs=args.epochs,
    batch_size=args.batch,
    lr=args.lr,
    freeze_layers=args.freeze,
    grad_accum=args.grad_accum,
    out_dir=args.out
)

print(f"\nFinished. Best model saved to: {ckpt.resolve()}")

