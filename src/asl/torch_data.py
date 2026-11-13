"""
Fase de datos para PyTorch (CPU). No toca la GPU.

Decodifica todos los JPEG con PIL en `workers` procesos y deja cada split como un tensor
float32 (N, 3, 64, 64) en [0, 1] en memoria. La GPU solo entra después, en el entrenamiento.
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import IMG_SIZE


class _JpegDataset(Dataset):
    def __init__(self, paths, size=IMG_SIZE):
        self.paths = list(paths)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize(self.size, Image.BILINEAR)
        return torch.from_numpy(np.asarray(img).copy())          # (H, W, 3) uint8


def decode_split(df: pd.DataFrame, workers: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Lee, decodifica, redimensiona y normaliza un split.
    Devuelve (X float32 NCHW en [0,1], y int64, segundos).
    """
    t0 = time.perf_counter()
    loader = DataLoader(_JpegDataset(df["path"].values), batch_size=256, shuffle=False,
                        num_workers=workers, persistent_workers=False)
    X = torch.cat([b for b in loader])                            # (N, H, W, 3) uint8
    X = X.permute(0, 3, 1, 2).float().div_(255.0).contiguous()    # (N, 3, H, W)
    y = torch.from_numpy(df["label_id"].values.astype(np.int64))
    return X, y, time.perf_counter() - t0
