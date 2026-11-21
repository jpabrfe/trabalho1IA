"""Testes do ArcFaceEmbedder (spec 001) sem carregar o modelo nem usar fotos."""
import numpy as np
import pytest

from src.face_app.core import embedder as modulo
from src.face_app.core.embedder import ArcFaceEmbedder


def _recorte():
    return np.zeros((160, 160, 3), dtype=np.uint8)


def test_embedding_tem_forma_dtype_e_norma_unitaria(monkeypatch):
    monkeypatch.setattr(modulo, "_extrair", lambda r: np.arange(1, 513, dtype=np.float64))
    emb = ArcFaceEmbedder().embed(_recorte())
    assert emb.shape == (512,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


@pytest.mark.parametrize("entrada", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_recorte_invalido_levanta_erro(entrada):
    with pytest.raises(ValueError):
        ArcFaceEmbedder().embed(entrada)


def test_forma_errada_levanta_erro(monkeypatch):
    monkeypatch.setattr(modulo, "_extrair", lambda r: np.ones(10))
    with pytest.raises(ValueError):
        ArcFaceEmbedder().embed(_recorte())


def test_norma_zero_levanta_erro(monkeypatch):
    monkeypatch.setattr(modulo, "_extrair", lambda r: np.zeros(512))
    with pytest.raises(ValueError):
        ArcFaceEmbedder().embed(_recorte())