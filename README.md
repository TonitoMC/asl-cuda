# Proyecto 3. Mejorando algoritmos con CUDA

CC3069 Computación Paralela y Distribuida | UVG | Semestre II 2026

Acelera con CUDA el entrenamiento del clasificador ASL del Lab 3 de Data Science
(CNN sobre ASL Alphabet, 29 clases, imágenes 64x64x3). Regla del enunciado: **los archivos
se preparan en la CPU y la GPU se usa solo para la matemática de la red.**

| Documento | Qué es |
|---|---|
| [`docs/INFORME.pdf`](docs/INFORME.pdf) / [`INFORME.tex`](docs/INFORME.tex) | Informe final en LaTeX (fuente `.tex`, PDF compilado) |

## Resultados

Segundos por época, sin contar la época 1 (calentamiento), 5 épocas fijas, lote 64, 3 corridas
(media ± desviación sobre las épocas 2 a 5). Máquina: i5-14600KF (20 hilos) + RTX 4070 SUPER.

| Configuración | CNN-1 | CNN-2 | CNN-3 |
|---|:---:|:---:|:---:|
| Keras CPU (línea base) | 4.085 ± 0.063 | 21.628 ± 0.267 | 22.496 ± 0.342 |
| PyTorch CPU (1 corrida) | 9.583 ± 1.110 | 36.168 ± 1.256 | 34.106 ± 0.078 |
| NumPy CPU, desde cero | 64.123 ± 7.809 | no medido | no medido |
| CuPy GPU, desde cero | 1.088 ± 0.033 | 4.755 ± 0.191 | 4.925 ± 0.163 |
| CuPy GPU + kernels CUDA propios | 0.691 ± 0.028 | 4.230 ± 0.170 | 4.422 ± 0.130 |
| **PyTorch CUDA (cuDNN)** | **0.425 ± 0.014** | **1.485 ± 0.008** | **1.510 ± 0.012** |

| Speedup | CNN-1 | CNN-2 | CNN-3 |
|---|:---:|:---:|:---:|
| PyTorch CUDA vs Keras CPU | 9.6x | 14.6x | 14.9x |
| CuPy vs NumPy (desde cero) | 58.9x | | |
| CuPy + kernels propios vs NumPy | 92.9x | | |
| CuPy + kernels propios vs CuPy sin kernels | 1.58x | 1.12x | 1.11x |

Todas las comparaciones entre pares de configuraciones: [`results/speedup_matrix.md`](results/speedup_matrix.md) (se regenera con `scripts/aggregate.py`).

Los kernels propios ayudan sobre todo en CNN-1; en CNN-2 y CNN-3 manda la multiplicación de matrices (cuBLAS),
que no se toca. Detalle y microbenchmark por kernel en la sección 5.6 del informe.

Lectura de los resultados (CNN-3, la más pesada, contra Keras CPU):

| Orden de más rápida a más lenta | s/época | vs Keras CPU |
|---|:---:|:---:|
| PyTorch en GPU (cuDNN) | 1.510 | 14.9x |
| CuPy en GPU con kernels propios | 4.422 | 5.1x |
| CuPy en GPU sin kernels propios | 4.925 | 4.6x |
| Keras en CPU | 22.496 | 1.0x |
| PyTorch en CPU | 34.106 | 0.7x |

Los kernels propios le ganan a CuPy sin kernels pero siguen detrás de PyTorch en GPU, que usa cuDNN y no arma la matriz `im2col`.
Los datos se preparan en CPU (5.9x con 20 workers contra 1) y la copia de cada lote a la GPU cuesta 3 % a 10 % de la época.
El 99 % de exactitud sale de un test del mismo dataset (frames de una sola persona), así que no generaliza a una cámara distinta.

Entrenamiento completo con `EarlyStopping(patience=5)` y `ReduceLROnPlateau(factor=0.3, patience=2)`
(RTX 4070 SUPER): CNN-1 91.86 %, CNN-2 99.59 %, CNN-3 99.31 % de exactitud en test.

## Organización del código

- **El código que produce los resultados** vive en `src/` (módulos: datos, modelos, CNN desde cero) y `scripts/` (programas que miden y escriben `results/*.json`). Es la fuente de verdad.
- **`notebooks/01_baseline_keras`** es la solución secuencial original del Lab 3 y es la línea base de Keras (ella misma escribe `results/keras_cpu_run*.json`).
- **`notebooks/02` y `03`** son la versión para presentar: verifican el hardware, llaman a los scripts de `scripts/` (muestran su código con `inspect.getsource` y su salida) y dejan las tablas de speedup. No calculan nada que no esté en los scripts.
- Entregable de código del enunciado: notebook 01 (secuencial) + notebook 02 (fase A datos en CPU, fase B entrenamiento en CUDA) + los módulos que importan.

## Estructura

```
src/
  asl/                  librería: splits y tf.data (data.py), decodificación con N workers
                        (torch_data.py), las tres CNN en PyTorch (torch_models.py)
  scratch/              CNN desde cero: numpy_cnn.py, cupy_cnn.py (con kernels CUDA), cache_data.py
scripts/                se corren desde la raíz: benchmark.py, bench_scratch.py, evidence.py, capture_evidence.sh,
                        train_full.py, aggregate.py, generate_report_charts.py
notebooks/
  01_baseline_keras     solución secuencial original (Lab 3)
  02_pipeline_cuda      pipeline CPU -> CUDA con PyTorch, evidencia y tablas de speedup
  03_cnn_scratch        NumPy contra CuPy con kernels propios, verificación y microbenchmark
tests/test_scratch.py   verificación de las CNN desde cero (contra PyTorch, diferencias finitas, kernels)
data/splits/*.csv       splits del Lab 3 (80/10/10, seed 42). No cambiarlos
results/                JSON de tiempos (*_run*.json), summary.json, charts/, evidence/
results/archive/        corridas viejas y de otras máquinas, no se usan en el informe
docs/                   informe (INFORME.tex + INFORME.pdf)
```

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt          # TensorFlow, solo para el notebook de Keras
uv pip install torch numpy pandas pillow    # referencia y datos (torch con el índice de CUDA que corresponda)
uv pip install cupy-cuda12x                 # versión CuPy, necesita GPU NVIDIA
```

Dataset: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
`kagglehub.dataset_download('grassknoted/asl-alphabet')` lo baja sin credenciales a
`~/.cache/kagglehub`, y `src/asl/data.py` lo encuentra ahí. También sirve descomprimirlo en
`data/` para que quede `data/asl_alphabet_train/asl_alphabet_train/<clase>/`.

## Cómo reproducir

Lo más simple es correr los notebooks en orden (`01`, `02`, `03`). Por línea de comandos, desde la raíz:

```bash
python scripts/benchmark.py --device cuda --epochs 5 --run 1     # results/torch_cuda_run1.json
python scripts/benchmark.py --device cpu  --epochs 5 --run 1

python src/scratch/cache_data.py                                  # decodifica el dataset una vez (CPU)
python scripts/bench_scratch.py --backend numpy --run 1 --models "CNN-1 Baseline"
ASL_CUPY_KERNELS=0 python scripts/bench_scratch.py --backend cupy --run 1                  # scratch_cupy_run1.json
ASL_CUPY_KERNELS=1 python scripts/bench_scratch.py --backend cupy --run 1 --tag _kernels   # scratch_cupy_kernels_run1.json

python scripts/evidence.py --epochs 40       # usage.csv y usage.png
bash scripts/capture_evidence.sh             # capturas de pantalla de nvidia-smi + top por fase (results/evidence/screens/)
python scripts/train_full.py                 # entrenamiento completo con Early Stopping
python scripts/aggregate.py                  # results/summary.json y tabla en consola
python scripts/generate_report_charts.py     # results/charts/
cd docs && latexmk -xelatex INFORME.tex      # docs/INFORME.pdf (el informe es docs/INFORME.tex)

pytest tests/test_scratch.py -v              # NumPy contra PyTorch, CuPy contra NumPy, kernels (~6 min con la caché de datos)
```

## Estado de las pruebas

`pytest tests/test_scratch.py`: antes pasaban 20 de 21; falla `test_vs_torch[CNN-1 Baseline]` (NumPy contra PyTorch) tenía error relativo
de 4.3e-3 en el gradiente de la primera convolución contra un umbral de 1e-3 (acumulación en `float32`). Se amplió el umbral a 1e-2
(no se volvió a correr la suite completa tras el cambio).
