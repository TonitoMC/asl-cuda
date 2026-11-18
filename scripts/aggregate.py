"""
Agrega results/*_run*.json: media y desviación por configuración, descartando la época 1
(calentamiento) y calculando el speedup contra Keras CPU.

    python scripts/aggregate.py
"""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = "keras_cpu"


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


ORDEN = ["keras_cpu", "torch_cpu", "scratch_numpy", "scratch_cupy", "scratch_cupy_kernels", "torch_cuda"]
NOMBRES = {"keras_cpu": "Keras CPU", "torch_cpu": "PyTorch CPU", "scratch_numpy": "NumPy",
           "scratch_cupy": "CuPy", "scratch_cupy_kernels": "CuPy+kernels", "torch_cuda": "PyTorch CUDA"}


def matriz_speedup(resumen):
    """Markdown con, por modelo, T_fila / T_columna: cuántas veces es más rápida la columna que la fila."""
    out = []
    for m in ["CNN-1 Baseline", "CNN-2 Profunda", "CNN-3 Regularizada"]:
        cfgs = [c for c in ORDEN if c in resumen and m in resumen[c]["modelos"]]
        t = {c: resumen[c]["modelos"][m]["epoch_s"][0] for c in cfgs}
        out.append(f"### {m}\n\nCada celda es T_fila / T_columna (cuántas veces es más rápida la columna que la fila).\n")
        out.append("| s/época | " + " | ".join(NOMBRES[c] for c in cfgs) + " |")
        out.append("|:---|" + ":---:|" * len(cfgs))
        for a in cfgs:
            fila = [("-" if a == b else f"{t[a] / t[b]:.2f}x") for b in cfgs]
            out.append(f"| **{NOMBRES[a]}** ({t[a]:.3f}) | " + " | ".join(fila) + " |")
        out.append("")
    return "\n".join(out)


def main():
    grupos = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "results" / "*_run*.json"))):
        nombre = Path(f).stem.rsplit("_run", 1)[0]
        grupos[nombre].append(json.load(open(f)))

    resumen = {}
    for cfg, runs in grupos.items():
        r = {"corridas": len(runs), "modelos": {}}
        for k in ("data_prep_seconds", "train_decode_1_workers_s"):
            vals = [x[k] for x in runs if k in x]
            if vals:
                r[k] = ms(vals)
        for m in runs[0]["modelos"]:
            # segundos por época sin la primera, promediando todas las épocas de todas las corridas
            eps = [t for x in runs for t in x["modelos"][m]["epoch_seconds"][1:]]
            h2d = [t for x in runs for t in x["modelos"][m].get("h2d_seconds", [0])[1:]]
            r["modelos"][m] = {"epoch_s": ms(eps),
                               "h2d_s": ms(h2d) if h2d else None,
                               "acc": ms([x["modelos"][m]["accuracy"] for x in runs])[0]}
        resumen[cfg] = r

    base = resumen.get(BASELINE)
    print(f"{'Config':<22}{'Modelo':<22}{'s/época':>16}{'H2D s/ép':>10}{'Speedup':>9}{'Acc':>8}")
    print("-" * 77)
    for cfg, r in resumen.items():
        for m, v in r["modelos"].items():
            mean, sd = v["epoch_s"]
            sp = (base["modelos"][m]["epoch_s"][0] / mean) if base else float("nan")
            h = f"{v['h2d_s'][0]:.3f}" if v["h2d_s"] else "-"
            print(f"{cfg:<22}{m:<22}{mean:>9.2f} ± {sd:<5.2f}{h:>10}{sp:>8.1f}x{v['acc'] * 100:>7.1f}%")
        extra = [f"{k}={r[k][0]:.2f}s" for k in ("data_prep_seconds", "train_decode_1_workers_s") if k in r]
        print(f"{'':<22}({r['corridas']} corridas; {', '.join(extra)})\n")

    (ROOT / "results" / "summary.json").write_text(json.dumps(resumen, indent=2))
    (ROOT / "results" / "speedup_matrix.md").write_text(matriz_speedup(resumen), encoding="utf-8")


if __name__ == "__main__":
    main()
