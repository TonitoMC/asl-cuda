"""
Carga sostenida para tomar capturas de pantalla de nvidia-smi y top (ver capture_evidence.sh).

    python scripts/evidence_workload.py data    # fase A: decodificar JPEG con todos los hilos (CPU)
    python scripts/evidence_workload.py train   # fase B: entrenar CNN-2 en CUDA
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch                                          # noqa: E402
import torch.nn as nn                                 # noqa: E402
from asl.data import load_splits, RANDOM_SEED         # noqa: E402
from asl.torch_data import decode_split               # noqa: E402

phase = sys.argv[1]
df_train, _, _ = load_splits()
if phase == "data":
    for _ in range(25):
        decode_split(df_train, os.cpu_count())        # solo CPU, la GPU no se toca
else:
    from asl.torch_models import MODELS
    from benchmark import train_epoch
    X, y, _ = decode_split(df_train, os.cpu_count())  # fase A, antes de tocar la GPU
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(RANDOM_SEED)
    model = MODELS["CNN-2 Profunda"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    for _ in range(40):
        train_epoch(model, opt, nn.CrossEntropyLoss(), X, y, device, 64, gen)
