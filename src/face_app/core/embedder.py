"""Extrator de embedding (spec 001): ArcFace buffalo_l, vetor (512,) float32 de norma 1.

Reaproveita arcface_embedding de src/models/embedder.py (Parte 1). O modelo so
e carregado na primeira chamada, nao ao importar este modulo.
"""
import numpy as np


def _extrair(recorte_bgr: np.ndarray) -> np.ndarray:
    from src.models.embedder import arcface_embedding

    return arcface_embedding(recorte_bgr)


class ArcFaceEmbedder:
    def embed(self, recorte_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(recorte_bgr, np.ndarray) or recorte_bgr.size == 0:
            raise ValueError("Recorte de face invalido para extracao de embedding.")

        vec = np.asarray(_extrair(recorte_bgr), dtype=np.float32).reshape(-1)
        if vec.shape != (512,):
            raise ValueError(f"Embedding esperado com forma (512,), obtido {vec.shape}.")

        norma = float(np.linalg.norm(vec))
        if not np.isfinite(norma) or norma == 0.0:
            raise ValueError("Norma do embedding invalida.")

        return (vec / norma).astype(np.float32)