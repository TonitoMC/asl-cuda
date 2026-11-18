"""
Entrena CNN-2 (la mejor del proyecto) en la GPU y guarda los pesos para el demo.
Reusa train_model de scripts/train_full.py (Early Stopping + ReduceLROnPlateau, igual que el informe).

    python demo/train_demo.py        # ~1 minuto en una RTX 4070 SUPER -> demo/weights/cnn2.pt
"""

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from asl.data import load_splits                  # noqa: E402
from asl.torch_data import decode_split           # noqa: E402
from asl.torch_models import MODELS               # noqa: E402
from train_full import train_model                # noqa: E402

NAME = "CNN-2 Profunda"
OUT = Path(__file__).resolve().parent / "weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_train, df_val, df_test = load_splits()
w = os.cpu_count() or 4
X_tr, y_tr, _ = decode_split(df_train, w)         # fase de datos, CPU
X_va, y_va, _ = decode_split(df_val, w)
X_te, y_te, _ = decode_split(df_test, w)

res = train_model(NAME, MODELS[NAME], X_tr, y_tr, X_va, y_va, X_te, y_te, device,
                  save_path=OUT / "cnn2.pt")      # fase de entrenamiento, CUDA
(OUT / "cnn2.json").write_text(json.dumps(
    {"modelo": NAME, "test_accuracy": res["test_accuracy"], "test_f1_macro": res["test_f1_macro"],
     "epocas": res["epochs_trained"]}, indent=2))
