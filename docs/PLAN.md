# Plan: Proyecto 3 CUDA (CC3069)

Objetivo: acelerar el entrenamiento del clasificador ASL del lab 3 y medir el speedup.
Regla del enunciado: **los archivos/imágenes se preparan en la CPU (varios núcleos) y la GPU
se usa solo para la matemática de la red.** Esa separación es el 40 % de la nota.

## 1. Qué es secuencial hoy (paso 1 del enunciado)

Fuente: `notebooks/01_baseline_keras.ipynb` (código del lab 3, Keras) y `src/asl/data.py`.
Datos: 14,500 imágenes de 29 clases, 64x64x3. Splits 11,600 train / 1,450 val / 1,450 test.

### Operaciones del programa

| No. | Operación | Qué hace | Dónde |
|---|---|---|---|
| 1 | Lectura de splits | `load_splits()` lee los CSV y arma rutas | CPU |
| 2 | Lectura de archivo | `tf.io.read_file`, un JPEG por imagen | CPU, I/O |
| 3 | Decodificación JPEG | `decode_jpeg` a tensor uint8 | CPU |
| 4 | Redimensionado | `tf.image.resize` a 64x64 | CPU |
| 5 | Normalización y one-hot | `/255`, etiqueta a vector de 29 | CPU |
| 6 | cache / shuffle / batch / prefetch | arma lotes de 64 | CPU |
| 7 | Forward de la red | Conv2D, BatchNorm, ReLU, MaxPool, GAP, Dense, Dropout, softmax | **candidata a GPU** |
| 8 | Pérdida | cross-entropy categórica | **GPU** |
| 9 | Backward | gradientes de todas las capas | **GPU** |
| 10 | Actualización | Adam sobre todos los pesos | **GPU** |
| 11 | Control por época | EarlyStopping, ReduceLROnPlateau, pasada de validación | CPU/GPU |
| 12 | Evaluación | `predict`, `argmax`, accuracy y F1 | CPU/GPU |

### Operaciones repetitivas

| Operación repetitiva | Datos que procesa | Repeticiones |
|---|---|---|
| Lectura + decode + resize de un JPEG | 1 imagen | 14,500 (una vez, luego `cache()`) |
| Paso de entrenamiento (forward + backward + Adam) | 1 lote de 64 | 182 por época, hasta 25 épocas |
| Convolución 3x3 / 5x5 | mapa de activaciones por lote | 2 a 3 capas por paso, cada paso |
| MaxPool / BatchNorm / ReLU | mapa de activaciones | 2 a 3 capas por paso |
| Pasada de validación | 1,450 imágenes (23 lotes) | 1 por época |

### Dónde se va el tiempo

Las convoluciones dominan. MACs por imagen, solo forward:

| Modelo | Conv (MACs) | Dense (MACs) | Parámetros |
|---|---|---|---|
| CNN-1 | 22 M | 4.2 M | 4,221,405 |
| CNN-2 | 158 M | 0.15 M | 519,069 |
| CNN-3 | 171 M | 0.07 M | 448,925 |

CNN-2 y CNN-3 hacen ~7x más cómputo por imagen que CNN-1 aunque tengan 8x menos parámetros,
y eso explica por qué en el lab 3 tardaron mucho más (2.3, 18.4 y 31.2 min; la diferencia
entre CNN-2 y CNN-3 viene sobre todo de cuántas épocas corrió cada una, no se midió el
dispositivo). El cómputo es denso, regular y sin dependencias entre imágenes de un lote, o
sea lo que mejor encaja en una GPU.

## 2. Qué se hace después

### Fase 0. Preparar entorno
- Descargar el dataset de Kaggle a `data/` (no está en esta máquina todavía).
- `uv venv --python 3.12`, instalar `requirements.txt`, confirmar que TF corre.
- Confirmar `torch.cuda.is_available()` (ya da True con torch 2.9.1+cu128, RTX 4070 12 GB, 20 hilos de CPU).

### Fase 1. Línea base (paso 2)
- Correr `01_baseline_keras.ipynb` con `FORCE_CPU = True`.
- Protocolo de tiempos: épocas fijas, **sin early stopping** (`EARLY_STOPPING = False`,
  `EPOCHS_OVERRIDE = 5`), 3 a 5 corridas, mismo split. Early stopping cambia el número de
  épocas y arruina la comparación.
- Se cronometran por separado preparación de datos y entrenamiento, el notebook ya lo hace.
- Corrida extra con `FORCE_CPU = False` como referencia "Keras en GPU".
- Salida en `results/keras_{cpu,gpu}_run{N}.json`.

### Fase 2. Port a PyTorch (pasos 3 y 4)
- `src/asl/torch_data.py` (CPU): `Dataset` + `DataLoader` con `num_workers` y `pin_memory`.
  Decodifica con PIL, mismo recorte y resize que el lab 3.
- `src/asl/torch_models.py`: las tres CNN como `nn.Module`. Chequeo de paridad: mismos
  conteos de parámetros que arriba y accuracy parecida a la de Keras.
- `src/asl/train.py` (CUDA): loop de entrenamiento, el modelo y los lotes pasan a `cuda`.
  Solo aquí se toca la GPU.
- Diseño de hilos, bloques y grids (paso 4): lo que usa cuDNN para las convs más lo que
  se escriba a mano en la fase 4.

### Fase 3. Benchmark (paso 5)
Tres configuraciones sobre los mismos datos:

| Config | Qué mide |
|---|---|
| Keras CPU | solución original, el denominador del speedup |
| PyTorch CPU | comparación justa, misma implementación sin GPU |
| PyTorch GPU | la versión acelerada |

Por fase: preparación de datos (con 1 worker contra N workers), transferencia CPU a GPU,
entrenamiento por época, evaluación. En GPU usar `torch.cuda.synchronize()` antes de
cada medición y descartar la primera época (calentamiento y autotuning de cuDNN).

### Fase 4. Kernel propio con Numba (opcional pero da el paso 4 real)
- `conv2d` forward en `@cuda.jit`: primero versión ingenua (un hilo por pixel de salida),
  luego con memoria compartida. ReLU y maxpool si da tiempo.
- Validar contra `torch.nn.functional.conv2d` y medirlo contra cuDNN.
- Esperamos perder contra cuDNN. Eso es parte del análisis, no un problema.
- El entrenamiento principal sigue usando cuDNN.

### Fase 5. Evidencia (30 puntos)
- `nvidia-smi dmon` o `--query-gpu=... -l 1 --format=csv` a archivo durante cada corrida.
- Capturas de `btop`/`htop` en la fase de datos (núcleos CPU saturados) y de `nvidia-smi`
  en entrenamiento (utilización y memoria de la GPU).
- Guardar todo en `results/evidencia/`.

### Fase 6. Informe PDF (paso 6)
- Tabla comparativa de tiempos y speedup por fase: `S = T_secuencial / T_paralelo`.
- Speedup total limitado por la parte que sigue en CPU (Amdahl).
- Discusión: por qué archivos en CPU multinúcleo y matemática en GPU, qué cambió CUDA,
  qué limita su uso (transferencias, modelos chicos, lotes chicos, memoria de la GPU).

## 3. Decisiones abiertas

1. **¿Cuenta PyTorch como "CUDA Python"?** Preguntar al catedrático o auxiliar. Si exigen
   código CUDA explícito, la fase 4 pasa de opcional a obligatoria.
2. **Redes densas (FC-1..3):** no se portaron. El enunciado habla del algoritmo, con las
   CNN alcanza. Se pueden agregar después.
3. **Modelo principal del informe:** CNN-3 (la mejor, 99.45 %) o CNN-2 (más rápida de entrenar).
   Probablemente las tres, el costo por modelo es chico.
