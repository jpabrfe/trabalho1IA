"""Funções compartilhadas pelos serviços."""
import logging

import numpy as np

from src.face_app.contracts import Face
from src.face_app.errors import FaceAppError, InvalidImageError
from src.face_app.storage.repository import EMBEDDING_DIM

log = logging.getLogger("face_app.services")


def validar_imagem(imagem) -> np.ndarray:
    """Imagem colorida BGR: array uint8 de forma (altura, largura, 3)."""
    if (
        not isinstance(imagem, np.ndarray)
        or imagem.dtype != np.uint8
        or imagem.ndim != 3
        or imagem.shape[2] != 3
        or imagem.shape[0] < 1
        or imagem.shape[1] < 1
    ):
        raise InvalidImageError("Imagem ilegível ou em formato não suportado.")
    return imagem


def adequada(face: Face, min_confidence: float, min_face_px: int) -> bool:
    _, _, largura, altura = face.caixa
    return face.confianca >= min_confidence and min(largura, altura) >= min_face_px


def detectar(detector, imagem, min_confidence, min_face_px):
    """Retorna (todas as faces, faces adequadas)."""
    try:
        todas = list(detector.detect(imagem))
    except FaceAppError:
        raise
    except Exception as exc:
        log.exception("falha inesperada no detector")
        raise FaceAppError("Falha ao detectar faces na imagem.") from exc
    return todas, [f for f in todas if adequada(f, min_confidence, min_face_px)]


def extrair(embedder, face: Face) -> np.ndarray:
    """Embedding validado e normalizado (norma 1) de uma face."""
    try:
        bruto = embedder.embed(face.recorte)
    except FaceAppError:
        raise
    except Exception as exc:
        log.exception("falha inesperada no extrator")
        raise FaceAppError("Falha ao extrair o embedding da face.") from exc
    try:
        emb = np.asarray(bruto, dtype=np.float32)
    except (TypeError, ValueError):
        raise FaceAppError("O extrator devolveu um embedding inválido.") from None
    if emb.shape != (EMBEDDING_DIM,) or not np.isfinite(emb).all():
        raise FaceAppError("O extrator devolveu um embedding inválido.")
    norma = float(np.linalg.norm(emb))
    if norma < 1e-6:
        raise FaceAppError("O extrator devolveu um embedding inválido.")
    return (emb / norma).astype(np.float32)