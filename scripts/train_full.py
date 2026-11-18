"""
Entrenamiento completo de las tres CNN en GPU (PyTorch/CUDA) con Early Stopping
y ReduceLROnPlateau, replicando exactamente la configuración del Lab 3.

Demuestra que la versión acelerada en GPU alcanza la precisión esperada (~99 %)
en cuestión de segundos/minutos.

    python scripts/train_full.py
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asl.data import load_splits, RANDOM_SEED                     # noqa: E402
from asl.torch_data import decode_split                          # noqa: E402
from asl.torch_models import MODELS                              # noqa: E402

EPOCHS = {
    "CNN-1 Baseline": 20,
    "CNN-2 Profunda": 25,
    "CNN-3 Regularizada": 25,
}
BATCH_SIZE = 64
PATIENCE = 5


def evaluate_split(model, X, y, device, batch_size=256):
    model.eval()
    all_preds = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * len(xb)
            all_preds.append(logits.argmax(dim=1).cpu())
    all_preds = torch.cat(all_preds).numpy()
    y_np = y.numpy()
    acc = float((all_preds == y_np).mean())
    macro_f1 = float(f1_score(y_np, all_preds, average="macro"))
    avg_loss = total_loss / len(X)
    return acc, macro_f1, avg_loss


def train_model(name, model_fn, X_tr, y_tr, X_val, y_val, X_te, y_te, device):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2, min_lr=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    max_epochs = EPOCHS[name]
    best_val_acc = -1.0
    best_weights = None
    epochs_no_improve = 0

    epoch_times = []
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n=======================================================")
    print(f" Entrenando {name} en {device} (Max épocas: {max_epochs})")
    print(f"=======================================================")

    t_start = time.perf_counter()
    for ep in range(1, max_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train()
        perm = torch.randperm(len(X_tr))
        train_loss = 0.0

        for i in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_duration = time.perf_counter() - t_ep_start
        epoch_times.append(ep_duration)

        train_loss /= len(X_tr)
        val_acc, val_f1, val_loss = evaluate_split(model, X_val, y_val, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"  Época {ep:02d}/{max_epochs:02d} - {ep_duration:.2f}s - "
              f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc*100:.2f}% (lr: {lr_curr:.1e})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  --> Early stopping activado en época {ep} (paciencia: {PATIENCE})")
                break

    total_time = time.perf_counter() - t_start

    # Restaurar mejores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_acc, test_f1, test_loss = evaluate_split(model, X_te, y_te, device)
    print(f"\nResultado final {name}:")
    print(f"  Test Accuracy: {test_acc*100:.2f}% | Test F1 Macro: {test_f1:.4f}")
    print(f"  Tiempo total: {total_time:.1f} s ({len(epoch_times)} épocas, media {np.mean(epoch_times):.2f} s/época)")

    return {
        "epochs_trained": len(epoch_times),
        "total_seconds": round(total_time, 2),
        "avg_epoch_seconds": round(float(np.mean(epoch_times)), 3),
        "best_val_acc": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "history": history
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Preparar datos (en CPU, usando cache si está disponible)
    cache_dir = ROOT / "data" / "cache"
    if (cache_dir / "train_X.npy").exists():
        print("Cargando datos desde caché data/cache/...", flush=True)
        from scratch.cache_data import load_cached
        X_tr, y_tr = [torch.from_numpy(arr) for arr in load_cached("train")]
        X_val, y_val = [torch.from_numpy(arr) for arr in load_cached("val")]
        X_te, y_te = [torch.from_numpy(arr) for arr in load_cached("test")]
    else:
        df_train, df_val, df_test = load_splits()
        workers = os.cpu_count() or 4
        print(f"Decodificando imágenes en CPU con {workers} workers...", flush=True)
        X_tr, y_tr, t_tr = decode_split(df_train, workers=workers)
        X_val, y_val, t_val = decode_split(df_val, workers=workers)
        X_te, y_te, t_te = decode_split(df_test, workers=workers)
    print(f"Datos listos: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}", flush=True)

    results = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "models": {}
    }

    for name, model_fn in MODELS.items():
        res = train_model(name, model_fn, X_tr, y_tr, X_val, y_val, X_te, y_te, device)
        results["models"][name] = res

    out_file = ROOT / "results" / "full_training_gpu.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados exitosamente en: {out_file}")


if __name__ == "__main__":
    main()
