"""Teste de fumaça: extrai um embedding com FaceNet e outro com ArcFace.

Uso (na raiz do projeto, com o ambiente ativado):
    python -m src.models.embedder                   # imagem sintética (ruído)
    python -m src.models.embedder --image foto.jpg  # imagem pública, se quiser

PRIVACIDADE: por padrão nenhuma foto de pessoa é usada. Não passe fotos
privadas por este script enquanto usar agentes de IA ou serviços externos.
"""
import argparse
import time
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _facenet_model():
    import torch
    from facenet_pytorch import InceptionResnetV1

    torch.set_grad_enabled(False)
    return InceptionResnetV1(pretrained="vggface2").eval()  # baixa os pesos na 1a vez


@lru_cache(maxsize=1)
def _arcface_model():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app.models["recognition"]  # w600k_r50 (ArcFace), entrada 112x112


def facenet_embedding(img_rgb: np.ndarray) -> np.ndarray:
    """img_rgb: uint8 HxWx3 em RGB. Retorna vetor de 512 dimensões."""
    import cv2
    import torch

    img = cv2.resize(img_rgb, (160, 160))
    x = torch.from_numpy(img).permute(2, 0, 1).float()
    x = ((x - 127.5) / 128.0).unsqueeze(0)  # padronização do FaceNet
    emb = _facenet_model()(x)
    return emb[0].detach().cpu().numpy()


def arcface_embedding(img_bgr: np.ndarray) -> np.ndarray:
    """img_bgr: uint8 HxWx3 em BGR. Retorna vetor de 512 dimensões, normalizado."""
    import cv2

    img = cv2.resize(img_bgr, (112, 112))
    feat = np.asarray(_arcface_model().get_feat(img)).reshape(-1)
    return feat / (np.linalg.norm(feat) + 1e-12)


def _load_images(path):
    import cv2

    if path is None:
        rng = np.random.default_rng(0)
        bgr = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Não consegui ler a imagem: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr


def _report(name, emb, seconds):
    print(f"{name:8s} dim={emb.shape[0]}  norma={np.linalg.norm(emb):.3f}  "
          f"primeiros={np.round(emb[:4], 4)}  ({seconds:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=None, help="imagem pública (opcional)")
    args = parser.parse_args()

    rgb, bgr = _load_images(args.image)

    t = time.time()
    _report("FaceNet", facenet_embedding(rgb), time.time() - t)

    t = time.time()
    _report("ArcFace", arcface_embedding(bgr), time.time() - t)
    print("Teste de fumaça concluído.")


if __name__ == "__main__":
    main()