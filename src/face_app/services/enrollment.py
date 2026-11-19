"""Serviço de cadastro (spec 002). O cadastro é atômico: qualquer erro, nada é gravado."""
import logging

import numpy as np

from src.face_app.contracts import EnrollResult
from src.face_app.errors import (
    DuplicatePersonError,
    FaceAppError,
    InvalidImageError,
    MultipleFacesError,
    NoFaceError,
)
from src.face_app.services._common import detectar, extrair, validar_imagem
from src.face_app.storage.repository import PersonRepository, normalizar_nome

log = logging.getLogger("face_app.services.enrollment")


class EnrollmentService:
    def __init__(self, detector, embedder, repo: PersonRepository,
                 threshold: float, min_confidence: float, min_face_px: int):
        self._detector, self._embedder, self._repo = detector, embedder, repo
        self._threshold = threshold
        self._min_conf, self._min_px = min_confidence, min_face_px

    def enroll(self, nome, imagens) -> EnrollResult:
        nome = normalizar_nome(nome)
        if self._repo.nome_existe(nome):
            raise DuplicatePersonError("Já existe uma pessoa cadastrada com este nome.")
        imagens = list(imagens) if imagens is not None else []
        if not imagens:
            raise InvalidImageError("Envie ao menos uma imagem.")

        embeddings = [
            self._embedding_da_imagem(img, i, len(imagens))
            for i, img in enumerate(imagens, start=1)
        ]
        matriz = np.stack(embeddings)

        _, _, protos = self._repo.prototipos()
        if len(protos):
            media = matriz.mean(axis=0)
            norma = float(np.linalg.norm(media))
            if norma < 1e-6:
                raise FaceAppError("Embeddings inconsistentes entre as imagens.")
            if float((protos @ (media / norma)).max()) >= self._threshold:
                raise DuplicatePersonError("Este rosto já está cadastrado.")

        id_ = self._repo.add_person(nome, matriz)
        log.info("cadastro concluido id=%s imagens=%d", id_, len(imagens))
        return EnrollResult(id=id_, n_embeddings=len(embeddings))

    def _embedding_da_imagem(self, imagem, i, total):
        try:
            validar_imagem(imagem)
            todas, adequadas = detectar(self._detector, imagem, self._min_conf, self._min_px)
            if not todas:
                raise NoFaceError()
            if not adequadas:
                raise InvalidImageError()
            if len(adequadas) > 1:
                raise MultipleFacesError()
            return extrair(self._embedder, adequadas[0])
        except FaceAppError as erro:
            if total > 1:  # diz qual das imagens falhou
                raise type(erro)(f"Imagem {i}: {erro.mensagem}") from erro
            raise