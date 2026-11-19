"""Serviço de reconhecimento (specs 003 e 004). Não grava nada em disco."""
import logging

from src.face_app.contracts import Reconhecimento
from src.face_app.services._common import detectar, extrair, validar_imagem
from src.face_app.storage.repository import PersonRepository

log = logging.getLogger("face_app.services.recognition")


class RecognitionService:
    def __init__(self, detector, embedder, repo: PersonRepository,
                 threshold: float, min_confidence: float, min_face_px: int):
        self._detector, self._embedder, self._repo = detector, embedder, repo
        self._threshold = threshold
        self._min_conf, self._min_px = min_confidence, min_face_px

    def n_pessoas(self) -> int:
        """Para a interface mostrar 'Nenhuma pessoa cadastrada'."""
        return self._repo.count()

    def reconhecer(self, imagem) -> list[Reconhecimento]:
        validar_imagem(imagem)
        _, faces = detectar(self._detector, imagem, self._min_conf, self._min_px)
        if not faces:
            return []

        _, nomes, protos = self._repo.prototipos()
        resultados = []
        for face in faces:
            emb = extrair(self._embedder, face)
            if not nomes:  # ninguém cadastrado: toda face é desconhecida
                resultados.append(Reconhecimento(face.caixa, None, 0.0))
                continue
            sims = protos @ emb
            i = int(sims.argmax())
            sim = float(sims[i])
            nome = nomes[i] if sim >= self._threshold else None  # igual ao threshold: aceita
            resultados.append(Reconhecimento(face.caixa, nome, sim))

        log.info("reconhecimento faces=%d reconhecidas=%d",
                 len(resultados), sum(r.nome is not None for r in resultados))
        return resultados