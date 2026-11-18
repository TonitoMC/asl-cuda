"""
Benchmark PyTorch por fases: preparación de datos (CPU), transferencia CPU->GPU,
entrenamiento y evaluación.

    python scripts/benchmark.py --device cuda --run 1
    python scripts/benchmark.py --device cpu  --run 1

Salida: results/torch_<device>_run<N>.json con la misma forma que los de Keras.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asl.data import load_splits, RANDOM_SEED                     # noqa: E402
from asl.torch_data import decode_split                          # noqa: E402
from asl.torch_models import MODELS, KERAS_PARAMS, count_params  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, X, y, device, batch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(device)
            correct += (model(xb).argmax(1).cpu() == y[i:i + batch]).sum().item()
    return correct / len(X)


def train_epoch(model, opt, loss_fn, X, y, device, batch, gen):
    """
    Una época. Devuelve (total, gather CPU, transferencia H2D, cómputo de la red), en segundos.

    - gather: armar el lote en la CPU (indexar el tensor).
    - h2d: copia CPU->GPU, con sync antes y después para aislarla.
    - step: forward + backward + Adam. En GPU se mide con eventos CUDA (tiempo real de los
      kernels); en CPU con el reloj, porque ahí no hay trabajo asíncrono.
    """
    model.train()
    perm = torch.randperm(len(X), generator=gen)
    t_gather = t_h2d = t_step_cpu = 0.0
    events = []
    cuda = device.type == "cuda"
    sync(device)
    t0 = time.perf_counter()
    for i in range(0, len(X), batch):
        idx = perm[i:i + batch]
        tg = time.perf_counter()
        xb_cpu, yb_cpu = X[idx], y[idx]
        t_gather += time.perf_counter() - tg

        sync(device)
        th = time.perf_counter()
        xb, yb = xb_cpu.to(device), yb_cpu.to(device)
        sync(device)
        t_h2d += time.perf_counter() - th

        if cuda:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        else:
            ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()
        if cuda:
            e1.record()
            events.append((e0, e1))
        else:
            t_step_cpu += time.perf_counter() - ts
    sync(device)
    total = time.perf_counter() - t0
    t_step = sum(a.elapsed_time(b) for a, b in events) / 1000.0 if cuda else t_step_cpu
    return total, t_gather, t_h2d, t_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--run", default="1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="procesos para decodificar JPEG en la fase de datos")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"torch {torch.__version__}, dispositivo {device}, hilos CPU {torch.get_num_threads()}")

    # ---- Fase 1: preparación de datos (CPU) --------------------------------------------
    df_train, df_val, df_test = load_splits()
    prep = {}
    for w in sorted({1, args.workers}):
        _, _, t = decode_split(df_train, w)
        prep[f"train_decode_{w}_workers_s"] = round(t, 3)
        print(f"decodificar train con {w} worker(s): {t:.2f} s")
    X_tr, y_tr, t1 = decode_split(df_train, args.workers)
    X_te, y_te, t2 = decode_split(df_test, args.workers)
    prep["data_prep_seconds"] = round(t1 + t2, 3)     # lo que usa el entrenamiento: train + test

    # ---- Fase 2: entrenamiento (CUDA si corresponde) ------------------------------------
    resultados = {"device": args.device, "run": args.run, "framework": "torch",
                  "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  "batch": args.batch, **prep, "modelos": {}}

    for nombre in args.models:
        torch.manual_seed(RANDOM_SEED)
        gen = torch.Generator().manual_seed(RANDOM_SEED)
        model = MODELS[nombre]().to(device)
        n_par = count_params(model)
        assert n_par == KERAS_PARAMS[nombre], (nombre, n_par, KERAS_PARAMS[nombre])

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        print(f"\n=== {nombre}, {n_par:,} parámetros ===")

        t_ep, t_gather, t_h2d, t_step = [], [], [], []
        t0 = time.perf_counter()
        for ep in range(args.epochs):
            te, tg, th, ts = train_epoch(model, opt, loss_fn, X_tr, y_tr, device, args.batch, gen)
            t_ep.append(te); t_gather.append(tg); t_h2d.append(th); t_step.append(ts)
            print(f"  época {ep + 1}: {te:.2f} s (gather {tg:.2f}, H2D {th:.2f}, red {ts:.2f})")
        t_train = time.perf_counter() - t0

        sync(device)
        te0 = time.perf_counter()
        acc = evaluate(model, X_te, y_te, device, 256)
        sync(device)
        t_eval = time.perf_counter() - te0

        resultados["modelos"][nombre] = {
            "parametros": n_par, "epocas": args.epochs,
            "train_seconds": round(t_train, 3),
            "epoch_seconds": [round(t, 3) for t in t_ep],
            "gather_seconds": [round(t, 3) for t in t_gather],
            "h2d_seconds": [round(t, 3) for t in t_h2d],
            "step_seconds": [round(t, 3) for t in t_step],
            "eval_seconds": round(t_eval, 3), "accuracy": acc,
        }
        print(f"{nombre}: {t_train:.1f} s, accuracy {acc * 100:.2f} %, eval {t_eval:.2f} s")

    out = ROOT / "results" / f"torch_{args.device}{args.tag}_run{args.run}.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
