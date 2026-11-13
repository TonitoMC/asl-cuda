"""
Benchmark de la CNN desde cero (misma forma de salida que src/benchmark.py).

    python src/scratch/bench_scratch.py --backend numpy --run 1 --models "CNN-1 Baseline"
    python src/scratch/bench_scratch.py --backend numpy --run 1 --train-subset 2000

Protocolo: lote 64, float32, semilla 42, Adam lr=1e-3, épocas fijas, sin early stopping ni
validación. El lote se arma en CPU (gather); con CuPy se copia a la GPU dentro del ciclo (h2d).
Salida: results/scratch_<backend>[tag]_run<N>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scratch.cache_data import load_cached                    # noqa: E402

SEED = 42


def get_backend(name):
    if name == "numpy":
        from scratch import numpy_cnn as impl
        return impl, np, (lambda: None), (lambda a: a)
    from scratch import cupy_cnn as impl                      # noqa: E402
    import cupy as cp
    return impl, cp, (lambda: cp.cuda.Device().synchronize()), cp.asarray


def evaluate(impl, model, xp, to_dev, X, y, batch=256):
    correct = 0
    for i in range(0, len(X), batch):
        pred = model.forward(to_dev(X[i:i + batch]), train=False).argmax(1)
        if xp is not np:
            pred = pred.get()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    ap.add_argument("--run", default="1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--train-subset", type=int, default=None, help="usar solo N imágenes de train")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    impl, xp, sync, to_dev = get_backend(args.backend)
    models = args.models or list(impl.MODELS)
    X_tr, y_tr = load_cached("train")
    X_te, y_te = load_cached("test")
    if args.train_subset:
        keep = np.random.RandomState(SEED).permutation(len(X_tr))[:args.train_subset]
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    print(f"backend {args.backend}, train {len(X_tr)}, test {len(X_te)}")

    res = {"device": "cpu" if args.backend == "numpy" else "cuda", "run": args.run,
           "framework": args.backend, "batch": args.batch, "train_size": len(X_tr),
           "modelos": {}}
    for nombre in models:
        model = impl.MODELS[nombre](SEED)
        opt = impl.Adam(model.params(), lr=1e-3)
        n_par = model.count_params()
        rng = np.random.RandomState(SEED)
        print(f"\n=== {nombre}, {n_par:,} parámetros ===")
        t_ep, t_g, t_h, t_s, losses = [], [], [], [], []
        t0 = time.perf_counter()
        for ep in range(args.epochs):
            perm = rng.permutation(len(X_tr))
            g = h = s = 0.0
            acc_loss = 0.0           # en CuPy se queda en GPU hasta el final de la época
            sync()
            te = time.perf_counter()
            for i in range(0, len(X_tr), args.batch):
                idx = perm[i:i + args.batch]
                a = time.perf_counter()
                xb, yb = X_tr[idx], y_tr[idx]
                g += time.perf_counter() - a
                sync(); a = time.perf_counter()
                xb, yb = to_dev(xb), to_dev(yb)
                sync(); h += time.perf_counter() - a
                a = time.perf_counter()
                acc_loss = acc_loss + impl.train_step(model, opt, xb, yb) * len(idx)
                sync(); s += time.perf_counter() - a
            sync()
            t_ep.append(time.perf_counter() - te); t_g.append(g); t_h.append(h); t_s.append(s)
            losses.append(float(acc_loss) / len(X_tr))
            print(f"  época {ep + 1}: {t_ep[-1]:.2f} s (gather {g:.2f}, H2D {h:.2f}, red {s:.2f})"
                  f", loss {losses[-1]:.4f}")
        t_train = time.perf_counter() - t0
        sync(); a = time.perf_counter()
        acc = evaluate(impl, model, xp, to_dev, X_te, y_te)
        sync(); t_eval = time.perf_counter() - a
        res["modelos"][nombre] = {
            "parametros": n_par, "epocas": args.epochs, "train_seconds": round(t_train, 3),
            "epoch_seconds": [round(t, 3) for t in t_ep],
            "gather_seconds": [round(t, 3) for t in t_g],
            "h2d_seconds": [round(t, 3) for t in t_h],
            "step_seconds": [round(t, 3) for t in t_s],
            "loss": [round(l, 4) for l in losses],
            "eval_seconds": round(t_eval, 3), "accuracy": acc}
        print(f"{nombre}: {t_train:.1f} s, accuracy {acc * 100:.2f} %")

    out = ROOT / "results" / f"scratch_{args.backend}{args.tag}_run{args.run}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
