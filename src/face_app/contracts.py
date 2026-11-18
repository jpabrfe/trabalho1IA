"""Contrato entre as camadas (specs 001 a 003). Mudanças só depois de combinadas."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Face:
    caixa: tuple[int, int, int, int]   # x, y, largura, altura
    recorte: np.ndarray                # BGR 160x160 uint8
    confianca: float


@dataclass(frozen=True)
class Reconhecimento:
    caixa: tuple[int, int, int, int]
    nome: str | None                   # None = desconhecida
    similaridade: float


@dataclass(frozen=True)
class EnrollResult:
    id: str
    n_embeddings: int


class FaceDetector(Protocol):
    def detect(self, imagem_bgr: np.ndarray) -> list[Face]: ...


class FaceEmbedder(Protocol):
    def embed(self, recorte_bgr: np.ndarray) -> np.ndarray:
        """Vetor float32 de forma (512,) com norma 1."""
        ...