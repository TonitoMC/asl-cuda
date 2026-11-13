"""
Fase de preparación de datos (CPU) del clasificador ASL.

Portado de lab3-ds/notebooks/utils/data_loader.py, dejando solo lo que usa el
entrenamiento de las CNN: constantes, lectura de los splits y datasets de TensorFlow.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Si es None, se busca el dataset en data/ y luego en la caché de kagglehub.
DATASET_ROOT: Optional[str] = None

# El zip de Kaggle trae la carpeta anidada dos veces.
TRAIN_SUBDIR = "asl_alphabet_train/asl_alphabet_train"

IMG_SIZE: Tuple[int, int] = (64, 64)
RANDOM_SEED: int = 42

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]
NUM_CLASSES = len(CLASS_NAMES)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def resolve_dataset_root(custom_path: Optional[str] = None) -> Path:
    """Devuelve la carpeta que contiene asl_alphabet_train/."""
    candidate = custom_path or DATASET_ROOT
    if candidate:
        p = Path(candidate)
        if (p / TRAIN_SUBDIR).exists():
            return p
        raise FileNotFoundError(f"No se encontró {TRAIN_SUBDIR} en {candidate}")

    local_data = PROJECT_ROOT / "data"
    if (local_data / TRAIN_SUBDIR).exists():
        return local_data

    cache_base = (Path.home() / ".cache" / "kagglehub" / "datasets" / "grassknoted"
                  / "asl-alphabet" / "versions")
    if cache_base.exists():
        for version in sorted(cache_base.iterdir(), reverse=True):
            if (version / TRAIN_SUBDIR).exists():
                return version

    raise FileNotFoundError(
        f"No se encontró el dataset. Se esperaba {local_data / TRAIN_SUBDIR}. "
        "Descargarlo de kaggle.com/datasets/grassknoted/asl-alphabet y descomprimir en data/."
    )


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga data/splits/{train,val,test}.csv (los mismos del lab 3, 80/10/10 estratificado,
    seed 42) y reconstruye las rutas absolutas.
    """
    root = resolve_dataset_root()

    def _cargar(nombre: str) -> pd.DataFrame:
        df = pd.read_csv(SPLITS_DIR / f"{nombre}.csv")
        df["path"] = df["path_rel"].map(lambda r: str(root / r))
        return df

    df_train, df_val, df_test = (_cargar(n) for n in ("train", "val", "test"))

    if any(not os.path.exists(p) for p in df_test["path"].head(20)):
        raise FileNotFoundError("Los splits apuntan a imágenes que no existen en el dataset.")

    print(f"[data] Train {len(df_train):,} | Val {len(df_val):,} | Test {len(df_test):,}")
    return df_train, df_val, df_test


def get_tf_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = 64,
    num_classes: int = NUM_CLASSES,
):
    """
    tf.data.Dataset de train, val y test. Val y test no se barajan, así las predicciones
    quedan alineadas con df_test['label_id'].
    """
    import tensorflow as tf

    def _make_dataset(df: pd.DataFrame, shuffle: bool = False):
        paths = df["path"].values
        labels = df["label_id"].values.astype(np.int32)

        def load_fn(path, label):
            img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.image.resize(img, img_size)
            return tf.cast(img, tf.float32) / 255.0, tf.one_hot(label, num_classes)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
        # cache() evita volver a decodificar los JPEG en cada época
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (_make_dataset(df_train, shuffle=True),
            _make_dataset(df_val),
            _make_dataset(df_test))
