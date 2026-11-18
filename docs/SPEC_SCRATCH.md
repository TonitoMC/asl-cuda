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

Igual que `scripts/benchmark.py`, para que las tablas se puedan unir:

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
  `accuracy`). `scripts/aggregate.py` los toma solo.
