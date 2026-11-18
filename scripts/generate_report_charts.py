"""
Genera gráficos de alta calidad para el informe final a partir de results/summary.json
y los resultados detallados de corridas.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores consistente
COLORS = {
    "keras_cpu": "#7f7f7f",     # Gris
    "torch_cpu": "#1f77b4",     # Azul oscuro
    "torch_cuda": "#2ca02c",    # Verde
    "scratch_numpy": "#d62728", # Rojo
    "scratch_cupy": "#ff7f0e",  # Naranja
    "scratch_cupy_kernels": "#9467bd",  # Morado
}

LABELS = {
    "keras_cpu": "Keras (CPU)",
    "torch_cpu": "PyTorch (CPU)",
    "torch_cuda": "PyTorch (CUDA cuDNN)",
    "scratch_numpy": "NumPy (CPU desde cero)",
    "scratch_cupy": "CuPy (GPU desde cero)",
    "scratch_cupy_kernels": "CuPy + kernels CUDA propios",
}


def load_summary():
    with open(RESULTS_DIR / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_times_comparison(summary):
    """Comparativa de s/época entre configuraciones."""
    models = ["CNN-1 Baseline", "CNN-2 Profunda", "CNN-3 Regularizada"]
    configs = ["keras_cpu", "torch_cpu", "torch_cuda", "scratch_cupy", "scratch_cupy_kernels"]

    x = np.arange(len(models))
    width = 0.16

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, cfg in enumerate(configs):
        if cfg not in summary:
            continue
        vals = []
        errs = []
        for m in models:
            if m in summary[cfg]["modelos"]:
                mean, sd = summary[cfg]["modelos"][m]["epoch_s"]
                vals.append(mean)
                errs.append(sd)
            else:
                vals.append(0)
                errs.append(0)

        offset = (idx - len(configs) / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                       label=LABELS.get(cfg, cfg), color=COLORS.get(cfg, "#333333"), alpha=0.9)

        # Etiquetas encima de las barras
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}s",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_ylabel("Segundos por Época (promedio)", fontsize=11, fontweight="bold")
    ax.set_title("Comparativa de Tiempo de Entrenamiento por Época", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    p = OUT_DIR / "epoch_times_comparison.png"
    plt.savefig(p, dpi=180)
    plt.close()
    print(f"Guardado: {p}")


def plot_speedups(summary):
    """Speedup contra Keras CPU."""
    models = ["CNN-1 Baseline", "CNN-2 Profunda", "CNN-3 Regularizada"]
    base = summary.get("keras_cpu")
    if not base:
        return

    configs = ["torch_cuda", "scratch_cupy", "scratch_cupy_kernels"]
    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, cfg in enumerate(configs):
        if cfg not in summary:
            continue
        sp_vals = []
        for m in models:
            t_base = base["modelos"][m]["epoch_s"][0]
            t_cfg = summary[cfg]["modelos"][m]["epoch_s"][0]
            sp_vals.append(t_base / t_cfg)

        offset = (idx - len(configs) / 2 + 0.5) * width
        rects = ax.bar(x + offset, sp_vals, width, label=f"Speedup {LABELS[cfg]}",
                       color=COLORS[cfg], alpha=0.9)

        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}x",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", lw=1.2, label="Línea base (1.0x Keras CPU)")
    ax.set_ylabel("Factor de Speedup (S = T_base / T_paralelo)", fontsize=11, fontweight="bold")
    ax.set_title("Aceleración (Speedup) obtenida con CUDA vs Keras CPU", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    p = OUT_DIR / "speedup_vs_baseline.png"
    plt.savefig(p, dpi=180)
    plt.close()
    print(f"Guardado: {p}")


def plot_phase_breakdown(summary):
    """Desglose por fase: Gather CPU, H2D transfer, Step GPU."""
    cfg = "torch_cuda"
    if cfg not in summary:
        return

    # Usar datos de torch_cuda_run1.json para desglose exacto
    run1 = json.load(open(RESULTS_DIR / "torch_cuda_run1.json", "r", encoding="utf-8"))
    models = ["CNN-1 Baseline", "CNN-2 Profunda", "CNN-3 Regularizada"]

    gather = [np.mean(run1["modelos"][m]["gather_seconds"][1:]) for m in models]
    h2d = [np.mean(run1["modelos"][m]["h2d_seconds"][1:]) for m in models]
    step = [np.mean(run1["modelos"][m]["step_seconds"][1:]) for m in models]

    x = np.arange(len(models))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5.5))
    p1 = ax.bar(x, gather, width, label="Gather (CPU: indexado y lote)", color="#6baed6")
    p2 = ax.bar(x, h2d, width, bottom=gather, label="Transferencia H2D (PCIe Host->GPU)", color="#fd8d3c")
    bottom_step = [g + h for g, h in zip(gather, h2d)]
    p3 = ax.bar(x, step, width, bottom=bottom_step, label="Cálculo Matemático Red (GPU)", color="#74c476")

    # Etiquetas de totales
    totals = [g + h + s for g, h, s in zip(gather, h2d, step)]
    for i, tot in enumerate(totals):
        ax.annotate(f"Total:\n{tot:.2f}s", xy=(x[i], tot), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Segundos por Época", fontsize=11, fontweight="bold")
    ax.set_title("Desglose del Tiempo por Fase en GPU (PyTorch CUDA)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    p = OUT_DIR / "phase_breakdown_gpu.png"
    plt.savefig(p, dpi=180)
    plt.close()
    print(f"Guardado: {p}")


def plot_training_convergence():
    """Curvas de pérdida y validación del entrenamiento completo en GPU."""
    json_path = RESULTS_DIR / "full_training_gpu.json"
    if not json_path.exists():
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_data = data.get("models", {})
    if not models_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        "CNN-1 Baseline": {"color": "#1f77b4", "line": "-"},
        "CNN-2 Profunda": {"color": "#2ca02c", "line": "-"},
        "CNN-3 Regularizada": {"color": "#ff7f0e", "line": "-"},
    }

    for name, info in models_data.items():
        hist = info["history"]
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        st = styles.get(name, {"color": "#333333", "line": "-"})

        # Loss
        ax1.plot(epochs, hist["train_loss"], label=f"{name} (Train)", color=st["color"], linestyle="--", alpha=0.7)
        ax1.plot(epochs, hist["val_loss"], label=f"{name} (Val)", color=st["color"], linestyle=st["line"], linewidth=2)

        # Accuracy
        ax2.plot(epochs, [a * 100 for a in hist["val_acc"]], label=f"{name} (Final: {info['test_accuracy']*100:.2f}%)",
                 color=st["color"], linestyle=st["line"], linewidth=2)

    ax1.set_xlabel("Época", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Convergencia de Función de Pérdida", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Época", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Exactitud de Validación (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Evolución de Exactitud de Validación (Val Acc)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Entrenamiento Completo en GPU (NVIDIA RTX 4070 SUPER) con Early Stopping y ReduceLROnPlateau", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = OUT_DIR / "training_convergence.png"
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {p}")


def main():
    summary = load_summary()
    plot_times_comparison(summary)
    plot_speedups(summary)
    plot_phase_breakdown(summary)
    plot_training_convergence()
    print("Todos los gráficos han sido generados exitosamente en results/charts/")


if __name__ == "__main__":
    main()
