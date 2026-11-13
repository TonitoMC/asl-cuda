# Proyecto 3. Mejorando algoritmos con CUDA

CC3069 Computación Paralela y Distribuida | UVG | Semestre II 2026

Acelera con CUDA el entrenamiento del clasificador ASL del Lab 3 de Data Science
(CNN sobre ASL Alphabet, 29 clases, 64x64). La preparación de imágenes se queda en la CPU y la
GPU se usa solo para la matemática de la red.

Plan completo y desglose de la solución secuencial: [`docs/PLAN.md`](docs/PLAN.md).
Enunciado: `Proyecto 3 CUDA.pdf` (no versionado aquí).

## Estructura

```
src/asl/data.py                      fase de datos (CPU): splits, lectura, preprocesamiento
notebooks/01_baseline_keras.ipynb    solución secuencial original (Keras), con tiempos por fase
data/splits/*.csv                    mismos splits del lab 3 (80/10/10, seed 42)
results/                             JSON de tiempos y evidencia de ejecución
docs/PLAN.md                         análisis, plan y decisiones abiertas
```

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Dataset: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet),
descomprimir en `data/` para que quede `data/asl_alphabet_train/asl_alphabet_train/<clase>/`.

## Estado

- [x] Línea base Keras limpia, sin correr todavía
- [ ] Mediciones secuenciales
- [ ] Port a PyTorch (CPU para datos, CUDA para entrenamiento)
- [ ] Benchmark y speedup
- [ ] Informe PDF
