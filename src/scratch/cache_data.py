"""
Decodifica los splits una sola vez y los guarda en data/cache/*.npy (uint8, NCHW).
Al cargarlos se dividen entre 255 y dan exactamente lo mismo que decode_split.

    python src/scratch/cache_data.py [--workers N]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "data" / "cache"


def load_cached(split):
    """(X float32 (N,3,64,64) en [0,1], y int64 (N,)). Los lotes se arman luego en CPU."""
    X = np.load(CACHE / f"{split}_X.npy")
    return X.astype(np.float32) / np.float32(255.0), np.load(CACHE / f"{split}_y.npy")


def main():
    from asl.data import load_splits
    from asl.torch_data import decode_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, df in zip(("train", "val", "test"), load_splits()):
        X, y, t = decode_split(df, args.workers)
        np.save(CACHE / f"{name}_X.npy", (X * 255).round().to(__import__("torch").uint8).numpy())
        np.save(CACHE / f"{name}_y.npy", y.numpy())
        print(f"{name}: {tuple(X.shape)} en {t:.1f} s")


if __name__ == "__main__":
    main()
