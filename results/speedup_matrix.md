### CNN-1 Baseline

Cada celda es T_fila / T_columna (cuántas veces es más rápida la columna que la fila).

| s/época | Keras CPU | PyTorch CPU | NumPy | CuPy | CuPy+kernels | PyTorch CUDA |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Keras CPU** (4.085) | - | 0.43x | 0.06x | 3.75x | 5.92x | 9.60x |
| **PyTorch CPU** (9.583) | 2.35x | - | 0.15x | 8.81x | 13.88x | 22.53x |
| **NumPy** (64.123) | 15.70x | 6.69x | - | 58.94x | 92.87x | 150.73x |
| **CuPy** (1.088) | 0.27x | 0.11x | 0.02x | - | 1.58x | 2.56x |
| **CuPy+kernels** (0.691) | 0.17x | 0.07x | 0.01x | 0.63x | - | 1.62x |
| **PyTorch CUDA** (0.425) | 0.10x | 0.04x | 0.01x | 0.39x | 0.62x | - |

### CNN-2 Profunda

Cada celda es T_fila / T_columna (cuántas veces es más rápida la columna que la fila).

| s/época | Keras CPU | PyTorch CPU | CuPy | CuPy+kernels | PyTorch CUDA |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Keras CPU** (21.628) | - | 0.60x | 4.55x | 5.11x | 14.57x |
| **PyTorch CPU** (36.168) | 1.67x | - | 7.61x | 8.55x | 24.36x |
| **CuPy** (4.755) | 0.22x | 0.13x | - | 1.12x | 3.20x |
| **CuPy+kernels** (4.230) | 0.20x | 0.12x | 0.89x | - | 2.85x |
| **PyTorch CUDA** (1.485) | 0.07x | 0.04x | 0.31x | 0.35x | - |

### CNN-3 Regularizada

Cada celda es T_fila / T_columna (cuántas veces es más rápida la columna que la fila).

| s/época | Keras CPU | PyTorch CPU | CuPy | CuPy+kernels | PyTorch CUDA |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Keras CPU** (22.496) | - | 0.66x | 4.57x | 5.09x | 14.90x |
| **PyTorch CPU** (34.106) | 1.52x | - | 6.93x | 7.71x | 22.59x |
| **CuPy** (4.925) | 0.22x | 0.14x | - | 1.11x | 3.26x |
| **CuPy+kernels** (4.422) | 0.20x | 0.13x | 0.90x | - | 2.93x |
| **PyTorch CUDA** (1.510) | 0.07x | 0.04x | 0.31x | 0.34x | - |
