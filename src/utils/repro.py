"""Reprodutibilidade: fixa seeds e registra versões, hardware e configuração.

Uso nos experimentos:
    from src.utils.repro import fixar_seed, registrar
    fixar_seed(42)
    ...
    registrar("E01_facenet", config={...}, resultados={...}, seed=42)

PRIVACIDADE: não registre nomes, caminhos nem resultados individuais de
participantes privados; só métricas agregadas.
"""
import json
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

PACOTES = [
    "torch", "torchvision", "numpy", "scikit-learn", "facenet-pytorch",
    "insightface", "onnxruntime", "opencv-python", "matplotlib", "pillow",
]


def fixar_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _versao(pacote):
    try:
        return version(pacote)
    except PackageNotFoundError:
        return None


def info_ambiente(seed=None):
    info = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "so": platform.platform(),
        "cpu": platform.processor(),
        "seed": seed,
        "pacotes": {p: _versao(p) for p in PACOTES},
    }
    try:
        import torch

        info["cuda_disponivel"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


def registrar(nome, config, resultados=None, seed=42, pasta="experiments/runs"):
    """Salva um JSON com config + ambiente + resultados. Retorna o caminho."""
    destino = Path(pasta)
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"{nome}_{datetime.now():%Y%m%d_%H%M%S}.json"
    arquivo.write_text(
        json.dumps(
            {"experimento": nome, "config": config, "ambiente": info_ambiente(seed),
             "resultados": resultados},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    return arquivo


if __name__ == "__main__":
    print(json.dumps(info_ambiente(42), indent=2, ensure_ascii=False))