"""Testes do detector (spec 001) sem fotos reais no repositorio.

Teste opcional com imagem real: defina FACE_TEST_IMAGE com o caminho de uma
imagem PUBLICA (ou sua), guardada FORA da pasta do projeto. Nunca commitar.
"""
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("facenet_pytorch")

from src.face_app.contracts import Face  # noqa: E402
from src.face_app.core.detector import MtcnnDetector  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    return MtcnnDetector()


def test_imagem_sem_face_devolve_lista_vazia(detector):
    cinza = np.full((240, 320, 3), 127, dtype=np.uint8)
    assert detector.detect(cinza) == []


@pytest.mark.parametrize(
    "entrada",
    [None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((50, 50), dtype=np.uint8),
     np.zeros((50, 50, 4), dtype=np.uint8), np.zeros((50, 50, 3), dtype=np.float32)],
)
def test_entradas_invalidas_nao_lancam_excecao(detector, entrada):
    assert isinstance(detector.detect(entrada), list)


@pytest.mark.skipif(not os.environ.get("FACE_TEST_IMAGE"), reason="defina FACE_TEST_IMAGE")
def test_contrato_com_imagem_real(detector):
    imagem = cv2.imread(os.environ["FACE_TEST_IMAGE"])
    assert imagem is not None, "nao consegui ler FACE_TEST_IMAGE"
    faces = detector.detect(imagem)
    assert len(faces) >= 1
    altura, largura = imagem.shape[:2]
    for f in faces:
        assert isinstance(f, Face)
        assert f.recorte.shape == (160, 160, 3) and f.recorte.dtype == np.uint8
        x, y, w, h = f.caixa
        assert 0 <= x < largura and 0 <= y < altura and w > 0 and h > 0
        assert 0.0 <= f.confianca <= 1.0