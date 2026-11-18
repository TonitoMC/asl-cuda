"""
Evidencia de uso de hardware por fase.

Fase "data":  decodifica el train varias veces con N workers (CPU multinúcleo, GPU ociosa).
Fase "train": entrena una CNN en CUDA (GPU ocupada, CPU casi ociosa).

Muestrea cada 0.5 s los núcleos de CPU (psutil) y la GPU (nvidia-smi) a un CSV y genera una
gráfica. Escribe results/evidence/phase.txt para poder tomar capturas de pantalla por fase.

    python scripts/evidence.py --epochs 40
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asl.data import load_splits, RANDOM_SEED        # noqa: E402
from asl.torch_data import decode_split              # noqa: E402
from asl.torch_models import MODELS                  # noqa: E402
from benchmark import train_epoch                    # noqa: E402

OUT = ROOT / "results" / "evidence"
PHASE = {"name": "idle"}


def set_phase(name):
    PHASE["name"] = name
    (OUT / "phase.txt").write_text(name)
    print(f"[fase] {name}", flush=True)


def gpu_sample():
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    u, m, p = [float(x) for x in r.stdout.strip().split(",")]
    return u, m, p


def sampler(stop, rows, t_start):
    psutil.cpu_percent(percpu=True)               # primer valor descartado
    while not stop.is_set():
        time.sleep(0.5)
        cores = psutil.cpu_percent(percpu=True)
        u, m, p = gpu_sample()
        rows.append([round(time.perf_counter() - t_start, 2), PHASE["name"], u, m, p,
                     round(sum(cores) / len(cores), 1)] + cores)


def plot(rows, ncores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.array([r[0] for r in rows])
    ph = [r[1] for r in rows]
    gpu_u = np.array([r[2] for r in rows])
    gpu_m = np.array([r[3] for r in rows])
    cpu = np.array([r[6:6 + ncores] for r in rows]).T

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1.6]})
    ax[0].plot(t, gpu_u, color="#2a7ae2", lw=2)
    ax[0].set_ylabel("GPU util. (%)"); ax[0].set_ylim(0, 105)
    ax[1].plot(t, gpu_m, color="#d2691e", lw=2)
    ax[1].set_ylabel("Memoria GPU (MiB)")
    im = ax[2].imshow(cpu, aspect="auto", cmap="viridis", vmin=0, vmax=100,
                      extent=[t[0], t[-1], ncores, 0], interpolation="nearest")
    ax[2].set_ylabel("Núcleo de CPU"); ax[2].set_xlabel("Tiempo (s)")
    fig.colorbar(im, ax=ax, label="Uso de CPU (%)", pad=0.01)

    # sombrear fases
    cambios = [0] + [i for i in range(1, len(ph)) if ph[i] != ph[i - 1]] + [len(ph) - 1]
    colores = {"data": "#f5d76e", "train": "#9ad0a0"}
    for a, b in zip(cambios[:-1], cambios[1:]):
        for a_ in ax[:2]:
            a_.axvspan(t[a], t[b], color=colores.get(ph[a], "none"), alpha=0.25)
        if ph[a] in colores:
            ax[0].text((t[a] + t[b]) / 2, 60, {"data": "Preparación\nde datos (CPU)",
                                               "train": "Entrenamiento (GPU)"}[ph[a]],
                       ha="center", va="top", fontsize=9)
    fig.suptitle("Uso de hardware por fase", fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--model", default="CNN-3 Regularizada")
    ap.add_argument("--data-repeats", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df_train, _, _ = load_splits()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    rows, stop, t_start = [], threading.Event(), time.perf_counter()
    th = threading.Thread(target=sampler, args=(stop, rows, t_start), daemon=True)
    th.start()
    time.sleep(2)                                         # reposo inicial

    set_phase("data")
    for _ in range(args.data_repeats):
        X, y, _ = decode_split(df_train, args.workers)

    time.sleep(1)
    set_phase("train")
    torch.manual_seed(RANDOM_SEED)
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    model = MODELS[args.model]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        te, tg, tt, ts = train_epoch(model, opt, loss_fn, X, y, device, args.batch, gen)
        if ep == args.epochs // 2:
            (OUT / "nvidia_smi_training.txt").write_text(
                subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
        print(f"época {ep + 1}/{args.epochs}: {te:.2f} s", flush=True)

    set_phase("idle")
    time.sleep(2)
    stop.set(); th.join()

    ncores = len(rows[0]) - 6
    with open(OUT / "usage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "gpu_util", "gpu_mem_mib", "gpu_power_w", "cpu_avg"]
                   + [f"cpu{i}" for i in range(ncores)])
        w.writerows(rows)
    plot(rows, ncores, OUT / "usage.png")
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
