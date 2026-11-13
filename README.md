# Proyecto 3. Mejorando algoritmos con CUDA

CC3069 Computación Paralela y Distribuida | UVG | Semestre II 2026

Acelera con CUDA el entrenamiento del clasificador ASL del Lab 3 de Data Science
(CNN sobre ASL Alphabet, 29 clases, imágenes 64x64x3). Regla del enunciado: **los archivos
se preparan en la CPU y la GPU se usa solo para la matemática de la red.**

- Análisis de la solución secuencial y plan: [`docs/PLAN.md`](docs/PLAN.md)
- Enunciado: `Proyecto 3 CUDA.pdf` (no versionado)

## Estado

| Pieza | Estado |
|---|---|
| Línea base secuencial (Keras, CPU) | Hecha, 3 corridas |
| Port a PyTorch, datos en CPU y entrenamiento en CUDA | Hecho |
| Benchmark por fases y speedup | Keras CPU y PyTorch GPU hechos. PyTorch CPU corriendo |
| Evidencia (capturas y logs de `nvidia-smi` y `btop`) | Hecha, en `results/evidence/` |
| **CNN desde cero en NumPy (CPU)** | **Pendiente, ver la especificación abajo** |
| **CNN desde cero en CuPy (GPU)** | **Pendiente, ver la especificación abajo** |
| Entrenamiento completo (con early stopping) para verificar accuracy | Pendiente |
| Informe PDF | Pendiente, ver [Informe](#informe) |

### Resultados hasta ahora

Segundos por época, sin contar la época 1 (calentamiento), 5 épocas, 3 corridas.
Sin pasada de validación en ningún caso.

| Configuración | CNN-1 | CNN-2 | CNN-3 |
|---|---|---|---|
| Keras CPU (línea base) | 4.08 | 21.63 | 22.50 |
| PyTorch GPU (RTX 4070) | 0.43 | 1.48 | 1.51 |
| **Speedup contra Keras CPU** | **9.6x** | **14.6x** | **14.9x** |

PyTorch CPU y los desgloses por fase se agregan con `python src/aggregate.py` cuando
terminen las corridas. El archivo `results/summary.json` se regenera cada vez.

## Estructura

```
src/asl/data.py              splits, rutas y tf.data (CPU)
src/asl/torch_data.py        decodificar JPEG con N workers a tensores (CPU)
src/asl/torch_models.py      las tres CNN en PyTorch. Sirven de referencia para las de cero
src/benchmark.py             benchmark PyTorch por fases (CPU o CUDA)
src/evidence.py              muestrea CPU y GPU durante una corrida, genera la gráfica
src/aggregate.py             media, desviación y speedup de todos los results/*_run*.json
notebooks/01_baseline_keras  solución secuencial original, con tiempos por fase
data/splits/*.csv            splits del lab 3 (80/10/10, seed 42). No cambiarlos
results/                     JSON de tiempos, logs y evidencia
```

## Setup

El benchmark corre con el Python del sistema, que ya tiene PyTorch con CUDA. Para el resto:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt          # TensorFlow, solo para el notebook de Keras
uv pip install torch numpy pandas pillow    # referencia y datos
uv pip install cupy-cuda12x                 # solo quien haga la versión CuPy, necesita GPU NVIDIA
```

Dataset: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
`kagglehub.dataset_download('grassknoted/asl-alphabet')` lo baja sin credenciales a
`~/.cache/kagglehub`, y `src/asl/data.py` lo encuentra ahí. También sirve descomprimirlo en
`data/` para que quede `data/asl_alphabet_train/asl_alphabet_train/<clase>/`.

Cada venv con PyTorch y CUDA pesa 5 a 6 GB, no hagan uno por proyecto.

Correr lo que ya existe:

```bash
python src/benchmark.py --device cuda --epochs 5 --run 1     # results/torch_cuda_run1.json
python src/benchmark.py --device cpu  --epochs 5 --run 1
python src/aggregate.py
```

---

# Especificación: CNN desde cero (NumPy y CuPy)

## Objetivo

Escribir una CNN **sin autograd ni framework**, con forward y backward a mano, en dos versiones:

- **NumPy** (CPU) y **CuPy** (GPU), con **el mismo algoritmo**, solo cambia la librería de
  arreglos. Así el speedup NumPy contra CuPy es una comparación justa, sin mezclar frameworks
  distintos como pasa entre Keras y PyTorch.
- Sirve además como el ejemplo de CUDA "explícito" del paso 4 (reparto de trabajo en la GPU),
  por si el catedrático no acepta PyTorch como CUDA Python.

Roles: **persona NumPy**, **persona CuPy**, **dueño del benchmark** (mide y compara).
La persona CuPy necesita una GPU NVIDIA. Si no la tiene, prueba en la máquina del dueño del
benchmark.

## Alcance

| Prioridad | Modelo | Qué agrega |
|---|---|---|
| Obligatoria | **CNN-1** | Conv, ReLU, MaxPool, Flatten, Dense, Dropout, softmax con entropía cruzada, Adam |
| Si alcanza | CNN-2 | BatchNorm, GlobalAveragePooling |
| Opcional | CNN-3 | Kernel 5x5, Dropout espacial |

Terminar CNN-1 bien, con las pruebas de abajo, vale más que tener las tres a medias.

### Arquitecturas

Entrada `(N, 3, 64, 64)`, `float32` en `[0, 1]`, formato **NCHW**. `padding='same'`, stride 1.

**CNN-1**, 4,221,405 parámetros:

| Capa | Salida |
|---|---|
| Conv 3x3, 32 filtros, ReLU | (N, 32, 64, 64) |
| MaxPool 2x2 | (N, 32, 32, 32) |
| Conv 3x3, 64 filtros, ReLU | (N, 64, 32, 32) |
| MaxPool 2x2 | (N, 64, 16, 16) |
| Flatten | (N, 16384) |
| Dense 256, ReLU, Dropout 0.5 | (N, 256) |
| Dense 29 | (N, 29) logits |

**CNN-2**, 519,069 parámetros: 3 bloques `Conv 3x3 (64, 128, 256) → BatchNorm → ReLU →
MaxPool`, luego GlobalAveragePooling, Dense 512 + ReLU + Dropout 0.4, Dense 29.

**CNN-3**, 448,925 parámetros: como CNN-2 pero la primera conv es 5x5 y hay Dropout espacial
(0.1) tras cada bloque, y la Dense es de 256 con Dropout 0.3.

Las definiciones exactas están en `src/asl/torch_models.py`. Los conteos de parámetros de
PyTorch coinciden con los de Keras, **los de ustedes tienen que dar lo mismo**
(BatchNorm cuenta sus medias y varianzas móviles).

## Reglas

- **Permitido:** `numpy` o `cupy` (incluido `matmul`, que en CuPy usa cuBLAS), `cupyx`.
  Kernels propios con `cupy.RawKernel` son un plus, se explican en el informe.
- **Prohibido en la implementación:** autograd, `torch.nn`, TensorFlow, `cupyx.scipy.ndimage`
  o cualquier convolución ya hecha.
- **PyTorch sí se puede usar para probar**: como referencia de resultados y gradientes.
- Solo `float32`. Sin AMP ni mixed precision, para que sea comparable con lo demás.

## Datos (CPU en ambas versiones)

La preparación de imágenes es la misma que en el resto del proyecto y se queda en CPU:

```python
from asl.data import load_splits
from asl.torch_data import decode_split

df_train, df_val, df_test = load_splits()
X_train, y_train, _ = decode_split(df_train, workers=20)   # tensores de torch
X_train, y_train = X_train.numpy(), y_train.numpy()        # (N,3,64,64) float32, (N,) int64
```

Conviene cachear los arreglos en `.npy` para no decodificar en cada ejecución (no los suban
a git). Para CuPy, **cada lote se arma en CPU (NumPy) y se copia a la GPU con `cp.asarray`
dentro del ciclo**, igual que en `benchmark.py`. No suban todo el dataset a la GPU de una vez,
porque eso quita la transferencia que el informe tiene que medir.

## Interfaz común

Las dos versiones deben poder correr con el mismo benchmark, así que siguen esta forma.
Archivos: `src/scratch/numpy_cnn.py` y `src/scratch/cupy_cnn.py`.

```python
class Layer:
    params: dict      # nombre -> arreglo
    grads: dict       # mismo nombre -> gradiente, se llena en backward
    def forward(self, x, train: bool): ...
    def backward(self, dout): ...      # devuelve gradiente respecto a la entrada

class Model:
    layers: list
    def forward(self, x, train=True): ...        # devuelve logits (N, 29)
    def backward(self, dlogits): ...
    def params(self): ...                        # lista plana de (arreglo, gradiente)

def softmax_cross_entropy(logits, y):            # devuelve (loss, dlogits)
    ...                                          # dlogits = (softmax - onehot) / N

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8): ...
    def step(self): ...
```

Convenciones para que todo sea comparable:

- Inicialización: la misma en las dos versiones, con una semilla fija. Generen los pesos en
  NumPy y la versión CuPy los copia con `cp.asarray`. Así, con la misma entrada, las dos
  deben dar el mismo resultado.
- Dropout invertido (se escala por `1/(1-p)` al entrenar, nada al evaluar). Guardar la máscara
  para el backward.
- BatchNorm: `eps=1e-3`, media móvil con `momentum=0.01` (la fracción de la media nueva, igual
  que en `torch_models.py`).
- Adam con `eps=1e-8`, igual que PyTorch.

## Pistas de implementación

- **Convolución con im2col.** Pasar cada parche 3x3 a una columna para convertir la
  convolución en una multiplicación de matrices.
  - Forward: `W` de forma `(Cout, Cin·k·k)` por las columnas `(Cin·k·k, N·H·W)`.
  - Backward: `dW = dout @ cols.T`, `dcols = W.T @ dout`, y `col2im` suma `dcols` de vuelta a
    la imagen. Basta un ciclo sobre los `k·k` desplazamientos del kernel, no hace falta un
    scatter.
  - Cuidado con la memoria: en CNN-2 las columnas del segundo bloque son
    `(64·32·32, 64·9)`, unos 150 MB por lote.
- **MaxPool 2x2:** `x.reshape(N, C, H//2, 2, W//2, 2).max(axis=(3, 5))`. Para el backward, una
  máscara de dónde estuvo el máximo (si hay empate, el primero).
- **Softmax:** restar el máximo por fila antes de `exp`.
- **BatchNorm backward:** el cálculo manual es la parte más fácil de equivocar. Hacerlo
  siguiendo la fórmula y probarlo con la verificación de abajo antes de seguir.
- **CuPy:** evitar `.get()` o `float(...)` dentro del ciclo de entrenamiento. Cada uno
  fuerza una sincronización y arruina los tiempos.
  Acumular la pérdida en la GPU y leerla una vez por época.

## Verificación obligatoria (antes de medir tiempos)

Un benchmark de una CNN con gradientes mal calculados no vale. Hacer estas pruebas en un
archivo `tests/test_scratch.py`:

1. **Contra PyTorch.** Cargar `torch_models.cnn1()`, copiar sus pesos a la versión propia
   (cuidar el orden de ejes de cada tensor) y comparar sobre un lote con Dropout apagado:
   - logits: diferencia máxima absoluta menor a `1e-4`.
   - gradiente de cada parámetro: error relativo menor a `1e-3`.
2. **Diferencias finitas** en una capa chica (conv con pocas entradas y BatchNorm) para
   confirmar el backward independientemente de PyTorch.
3. **NumPy contra CuPy.** Mismos pesos y mismo lote: la pérdida tiene que coincidir a `1e-4`
   durante 20 pasos de Adam.
4. **Aprende.** CNN-1 tiene que superar el 75% de accuracy en test tras 5 épocas.
   Referencia: PyTorch GPU llegó a ~79% y Keras a ~86% en las mismas 5 épocas.
5. **Parámetros.** `4,221,405` para CNN-1.

## Protocolo de medición (lo hace el dueño del benchmark)

Igual que `src/benchmark.py`, para que las tablas se puedan unir:

- Lote 64, `float32`, semilla 42, Adam `lr=1e-3`, **5 épocas fijas, sin early stopping, sin
  pasada de validación**, 3 corridas.
- Se descarta la época 1 y se promedia el resto.
- Por época se registran, en segundos: total, armado del lote en CPU (`gather`), transferencia
  CPU a GPU (`h2d`, CuPy) y cómputo de la red (`step`).
- En CuPy, `cp.cuda.Device().synchronize()` antes de arrancar y antes de parar cada
  temporizador. Sin eso solo se mide la cola de lanzamientos.
- NumPy será lento (CNN-2 puede tardar minutos por época). Si hace falta se usa un subconjunto
  del train o menos épocas, **igual para las dos versiones**, y se compara en segundos por
  época.
- Salida: `results/scratch_numpy_run<N>.json` y `results/scratch_cupy_run<N>.json`, con la
  misma forma que los de `benchmark.py` (`device`, `run`, `framework`, y por modelo
  `parametros`, `epocas`, `epoch_seconds`, `gather_seconds`, `h2d_seconds`, `step_seconds`,
  `accuracy`). `src/aggregate.py` los toma solo.

## Entregables de esta parte

- [ ] `src/scratch/numpy_cnn.py` (CNN-1 como mínimo)
- [ ] `src/scratch/cupy_cnn.py`
- [ ] `tests/test_scratch.py` con las cinco verificaciones, pasando
- [ ] Corrida de las dos versiones y JSON en `results/`
- [ ] Un párrafo para el informe sobre qué fue difícil y qué sorprendió (cada persona)

---

# Informe

Hoy **no hay informe escrito**. Lo que sí existe y se puede usar tal cual:

| Sección del enunciado | Qué hay ya | Qué falta |
|---|---|---|
| 1. Análisis secuencial | Tablas de operaciones y repeticiones, y cómputo por modelo, en `docs/PLAN.md` | Pasarlas al formato del informe |
| 2. Tiempo secuencial | `results/keras_cpu_run*.json`, procedimiento descrito arriba | Gráfica y redacción |
| 3. Oportunidades de paralelización | Argumento en `docs/PLAN.md` (las convoluciones dominan el cómputo) | Redactar |
| 4. Diseño paralelo | Lo de PyTorch es cuDNN. El reparto de hilos y bloques sale de la versión CuPy | Diagrama de grid y bloques con la versión CuPy |
| 5. Implementación CUDA | `src/benchmark.py`, JSON de PyTorch GPU | Corridas de PyTorch CPU y de las versiones desde cero |
| 6. Análisis y conclusiones | Tabla de speedup arriba | Gráficas, discusión técnica |
| Capturas de recursos (30 pts) | `results/evidence/`: 4 capturas, CSV y `usage.png` | Elegir cuáles van al informe |

Preguntas que el informe debe responder:

1. ¿Por qué procesar los archivos con varios núcleos de CPU y mandar a la GPU solo la matemática?
   Datos: decodificar el train tarda 3.3 s con 1 worker y 0.55 s con 20 workers. En
   entrenamiento, la GPU llega al 94% de uso y la CPU tiene un núcleo ocupado manejándola.
2. ¿Cómo cambió CUDA el tiempo de procesamiento? Speedup por fase y total.
3. ¿Qué factores justifican o limitan el uso de CUDA? Ideas con datos: la copia al GPU es
   0.05 s por época, el trabajo fijo por época pesa más en CNN-1 (9.6x) que en CNN-2 y CNN-3
   (~15x), Amdahl con la parte que sigue en CPU, lotes pequeños, memoria de 12 GB.

Cosas a decir con honestidad en el informe:

- Keras CPU es **más rápido que PyTorch CPU** (no se sabe por qué, no se probó). Reportar el
  speedup contra las dos líneas base y aclarar cuál es cuál.
- A 5 épocas no se alcanza el 99% de accuracy. Los tiempos son por época y no dependen de eso.
  La precisión final sale del entrenamiento completo, que está pendiente.
- La CPU no está ociosa durante el entrenamiento en GPU: un núcleo queda ocupado esperando a
  la GPU.

## Pendiente por hacer

- [ ] Preguntar al catedrático o auxiliar si PyTorch cuenta como "CUDA Python".
- [ ] Entrenamiento completo por modelo, con early stopping, para mostrar que la versión GPU
      llega al ~99%.
- [ ] Hacer commit y push de `src/`, `results/` y `docs/`, que todavía están sin subir.
