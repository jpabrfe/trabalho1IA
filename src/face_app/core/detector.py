"""Detector MTCNN (spec 001): recortes BGR 160x160 uint8, uma Face por rosto.

ATENCAO: o threshold so vale se estes parametros forem IDENTICOS aos de
src/data/align.py (Parte 1): margin=0, image_size=160, post_process=False.
"""
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

from src.face_app.contracts import Face


class MtcnnDetector:
    def __init__(self, device: str = "cpu", image_size: int = 160, margin: int = 0):
        self.image_size = image_size
        self.margin = margin
        self._mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=True,
            post_process=False,  # pixels em 0..255, sem normalizacao
            device=device,
        )

    @staticmethod
    def _normalizar_entrada(imagem) -> np.ndarray | None:
        """Devolve uma imagem BGR uint8 HxWx3, ou None se a entrada for inutilizavel."""
        if not isinstance(imagem, np.ndarray) or imagem.size == 0:
            return None
        if imagem.dtype != np.uint8:
            return None
        if imagem.ndim == 2:
            return cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        if imagem.ndim == 3 and imagem.shape[2] == 3:
            return imagem
        if imagem.ndim == 3 and imagem.shape[2] == 4:
            return cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
        return None

    def detect(self, imagem_bgr: np.ndarray) -> list[Face]:
        imagem = self._normalizar_entrada(imagem_bgr)
        if imagem is None:
            return []

        altura_img, largura_img = imagem.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))

        # Uma unica deteccao; os recortes saem das MESMAS caixas.
        boxes, probs = self._mtcnn.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            return []
        recortes = self._mtcnn.extract(pil_img, boxes, None)
        if recortes is None:
            return []
        if recortes.ndim == 3:  # um unico recorte, sem eixo de lote
            recortes = recortes.unsqueeze(0)

        faces: list[Face] = []
        for idx, box in enumerate(boxes):
            if idx >= recortes.shape[0]:
                break
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1 = max(0, min(x1, largura_img - 1))
            y1 = max(0, min(y1, altura_img - 1))
            x2 = max(x1 + 1, min(x2, largura_img))
            y2 = max(y1 + 1, min(y2, altura_img))

            rgb = recortes[idx].detach().cpu().permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.shape[:2] != (self.image_size, self.image_size):
                continue  # sem fallback: pre-processamento diferente invalida o threshold

            confianca = float(probs[idx]) if probs is not None and probs[idx] is not None else 0.0
            faces.append(
                Face(
                    caixa=(x1, y1, x2 - x1, y2 - y1),
                    recorte=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    confianca=confianca,
                )
            )
        return faces