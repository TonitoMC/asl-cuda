# Demo: ASL en vivo con CUDA

App de Streamlit que clasifica señas ASL con la CNN-2 del proyecto (99.59 % en test), con inferencia en la GPU.

```bash
pip install -r demo/requirements.txt     # torch con el índice de CUDA que corresponda
python demo/train_demo.py                # opcional: re-entrena y guarda demo/weights/cnn2.pt (~1 min)
streamlit run demo/app.py
```

- Pestaña "Cámara en vivo": recorta un cuadrado central, lo pasa a 64x64 (igual que el entrenamiento) y muestra
  el top 3, el tiempo de inferencia y los fps (tope de 30). Se puede alternar entre `cuda` y `cpu` en la barra lateral.
- Pestaña "Imagen": sube una imagen o toma una al azar del conjunto de prueba.
- Los pesos (`demo/weights/cnn2.pt`, 2 MB) vienen en el repo.
- La CNN se entrenó con las fotos del dataset ASL Alphabet (fondo y luz uniformes); con una cámara normal la exactitud baja.

## Imágenes de entrenamiento
`demo/muestras_dataset.png` tiene una imagen por clase. El dataset completo (ASL Alphabet de Kaggle) queda en
`~/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train/<clase>/`
(una carpeta por clase, 500 imágenes de 200x200 de las que se usan en los splits de `data/splits/`).
