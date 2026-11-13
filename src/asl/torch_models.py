"""Las tres CNN del lab 3 en PyTorch. Salen logits; la softmax va dentro de la pérdida."""

import torch.nn as nn

from .data import NUM_CLASSES


def _block(cin, cout, k):
    return [nn.Conv2d(cin, cout, k, padding=k // 2), nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
            nn.ReLU(), nn.MaxPool2d(2)]


def cnn1():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES))


def cnn2():
    return nn.Sequential(
        *_block(3, 64, 3), *_block(64, 128, 3), *_block(128, 256, 3),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, NUM_CLASSES))


def cnn3():
    return nn.Sequential(
        *_block(3, 64, 5), nn.Dropout2d(0.1),
        *_block(64, 128, 3), nn.Dropout2d(0.1),
        *_block(128, 256, 3), nn.Dropout2d(0.1),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))


MODELS = {"CNN-1 Baseline": cnn1, "CNN-2 Profunda": cnn2, "CNN-3 Regularizada": cnn3}

# Parámetros de las versiones Keras (count_params incluye las medias/varianzas de BatchNorm)
KERAS_PARAMS = {"CNN-1 Baseline": 4_221_405, "CNN-2 Profunda": 519_069,
                "CNN-3 Regularizada": 448_925}


def count_params(model) -> int:
    """Igual que Keras: pesos entrenables más buffers de BatchNorm (sin num_batches_tracked)."""
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for name, b in model.named_buffers() if "num_batches" not in name)
    return n
