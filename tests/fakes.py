"""Detector e extrator falsos para os testes: sem fotos reais, sem modelos."""
import numpy as np

from src.face_app.contracts import Face


def imagem(codigo=0):
    """Imagem sintética; o detector falso identifica a imagem pelo valor do pixel."""
    return np.full((200, 200, 3), codigo, dtype=np.uint8)


def face(identidade=1, variante=0, conf=0.99, lado=100):
    """Face sintética. O recorte carrega a identidade e a variante (outra 'foto')."""
    recorte = np.zeros((160, 160, 3), dtype=np.uint8)
    recorte[0, 0, 0] = identidade
    recorte[0, 1, 0] = variante
    return Face(caixa=(10, 10, lado, lado), recorte=recorte, confianca=conf)


class FakeDetector:
    def __init__(self, por_codigo=None):
        self.por_codigo = por_codigo if por_codigo is not None else {}

    def detect(self, imagem_bgr):
        return list(self.por_codigo.get(int(imagem_bgr[0, 0, 0]), []))


class FakeEmbedder:
    """Mesma identidade, embeddings parecidos (cos ~0,99); identidades distintas, quase ortogonais."""

    def __init__(self):
        self.chamadas = 0

    def embed(self, recorte_bgr):
        self.chamadas += 1
        ident, var = int(recorte_bgr[0, 0, 0]), int(recorte_bgr[0, 1, 0])
        base = np.random.default_rng(1000 + ident).normal(size=512)
        base /= np.linalg.norm(base)
        if var:
            ruido = np.random.default_rng(5000 + var).normal(size=512)
            base = base + 0.15 * ruido / np.linalg.norm(ruido)
        return (base / np.linalg.norm(base)).astype(np.float32)