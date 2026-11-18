"""
Demo del clasificador ASL (CNN-2) con cámara en vivo, inferencia en la GPU.

    streamlit run demo/app.py

Pestaña 1: cámara en vivo (se limita a 30 fps). Se recorta un cuadrado central, se redimensiona a
64x64 igual que en el entrenamiento y se clasifica con la CNN en CUDA.
Pestaña 2: sin cámara, con una imagen subida o una del conjunto de prueba.
"""

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asl.data import CLASS_NAMES, IMG_SIZE            # noqa: E402
from asl.torch_models import MODELS                   # noqa: E402

WEIGHTS = Path(__file__).resolve().parent / "weights" / "cnn2.pt"
MODEL_NAME = "CNN-2 Profunda"
MAX_FPS = 30

st.set_page_config(page_title="ASL en vivo (CUDA)", layout="wide")


@st.cache_resource
def load_model(device_name: str):
    model = MODELS[MODEL_NAME]()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.to(device_name).eval()


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """Mismo preprocesado del entrenamiento: resize bilineal a 64x64, [0, 1], NCHW."""
    img = Image.fromarray(rgb).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255.0)
    return x.unsqueeze(0)


@torch.no_grad()
def predict(model, rgb: np.ndarray, device_name: str, k: int = 3):
    x = preprocess(rgb).to(device_name)
    t0 = time.perf_counter()
    probs = torch.softmax(model(x), dim=1)[0]
    if device_name == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    p, i = probs.topk(k)
    return [(CLASS_NAMES[j], float(q)) for q, j in zip(p.cpu(), i.cpu())], ms


# ------------------------------- barra lateral -------------------------------
st.sidebar.title("Opciones")
devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
device_name = st.sidebar.selectbox("Dispositivo de inferencia", devices)
cam_index = st.sidebar.number_input("Índice de la cámara", 0, 9, 0)
roi_pct = st.sidebar.slider("Tamaño del recuadro (% del alto)", 30, 100, 70)
mirror = st.sidebar.checkbox("Espejo", True)

if not WEIGHTS.exists():
    st.error(f"No hay pesos en {WEIGHTS}. Correr primero: python demo/train_demo.py")
    st.stop()

model = load_model(device_name)
info = json.loads((WEIGHTS.parent / "cnn2.json").read_text()) if (WEIGHTS.parent / "cnn2.json").exists() else {}
gpu = torch.cuda.get_device_name(0) if device_name == "cuda" else "CPU"
st.title("Reconocimiento de ASL en vivo")
st.caption(f"{MODEL_NAME} | inferencia en: {gpu} | exactitud en test: "
           f"{info.get('test_accuracy', 0) * 100:.2f} %")

tab_live, tab_img = st.tabs(["Cámara en vivo", "Imagen"])

# --------------------------------- cámara ------------------------------------
with tab_live:
    st.write("Pon la mano dentro del recuadro verde, sobre un fondo liso. El modelo se entrenó con "
             "fotos del dataset ASL Alphabet, así que con cámara real funciona mejor con fondo y luz parecidos.")
    run = st.toggle("Encender cámara")
    col_video, col_pred = st.columns([2, 1])
    frame_box, pred_box, stat_box = col_video.empty(), col_pred.empty(), col_pred.empty()

    if run:
        cap = cv2.VideoCapture(int(cam_index), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # sin MJPG la cámara da ~4 fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, MAX_FPS)
        if not cap.isOpened():
            st.error(f"No se pudo abrir la cámara {cam_index}")
        fps_ema, last = 0.0, time.perf_counter()
        while run and cap.isOpened():
            t_frame = time.perf_counter()
            ok, bgr = cap.read()
            if not ok:
                break
            if mirror:
                bgr = cv2.flip(bgr, 1)
            h, w = bgr.shape[:2]
            side = int(h * roi_pct / 100)
            x0, y0 = (w - side) // 2, (h - side) // 2
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            top, ms = predict(model, rgb[y0:y0 + side, x0:x0 + side], device_name)
            cv2.rectangle(rgb, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 3)
            frame_box.image(rgb, channels="RGB", use_container_width=True)
            pred_box.markdown(f"## {top[0][0]}  ({top[0][1] * 100:.0f} %)\n\n" +
                              "\n\n".join(f"{c}: {p * 100:.0f} %" for c, p in top[1:]))
            now = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 / max(now - last, 1e-6) if fps_ema else 1 / max(now - last, 1e-6)
            last = now
            stat_box.markdown(f"Inferencia: **{ms:.1f} ms** | {fps_ema:.0f} fps (tope {MAX_FPS})")
            # tope de fps: dormir lo que sobre del presupuesto de 1/30 s
            time.sleep(max(0.0, 1 / MAX_FPS - (time.perf_counter() - t_frame)))
        cap.release()

# --------------------------------- imagen ------------------------------------
with tab_img:
    up = st.file_uploader("Subir una imagen", type=["jpg", "jpeg", "png"])
    use_test = st.button("Imagen al azar del conjunto de prueba")
    rgb, label = None, None
    if up is not None:
        rgb = np.asarray(Image.open(up).convert("RGB"))
    elif use_test:
        from asl.data import load_splits
        _, _, df_test = load_splits()
        row = df_test.iloc[random.randrange(len(df_test))]
        rgb, label = np.asarray(Image.open(row["path"]).convert("RGB")), row["label"]
    if rgb is not None:
        top, ms = predict(model, rgb, device_name)
        c1, c2 = st.columns(2)
        c1.image(rgb, use_container_width=True)
        c2.markdown(f"## {top[0][0]}  ({top[0][1] * 100:.0f} %)")
        if label is not None:
            c2.write(f"Etiqueta real: {label} {'(correcto)' if label == top[0][0] else '(incorrecto)'}")
        c2.write(f"Inferencia: {ms:.1f} ms en {gpu}")
