"""Repositório de pessoas (specs 002 e 006, ADR-002).

Layout em disco, dentro de FACE_PRIVATE_DIR/people/:
    <uuid>/meta.json         nome, data de criação e número de embeddings
    <uuid>/embeddings.npy    matriz float32 (n, 512), cada linha com norma 1
Fotos NÃO são guardadas (minimização de dados). Excluir uma pessoa é remover a pasta.
PRIVACIDADE: os logs usam só o uuid; nunca nomes nem embeddings.
"""
import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.face_app.errors import DuplicatePersonError, FaceAppError, InvalidNameError

EMBEDDING_DIM = 512
_NOME_RE = re.compile(r"^[\w .'-]+$")
_PREFIXO_TMP = ".tmp-"
log = logging.getLogger("face_app.storage")


@dataclass(frozen=True)
class Person:
    id: str
    nome: str
    n_embeddings: int
    criado_em: str


def normalizar_nome(nome) -> str:
    """Remove espaços repetidos e valida o nome (1 a 60 caracteres)."""
    if not isinstance(nome, str):
        raise InvalidNameError()
    limpo = " ".join(nome.split())
    if not limpo or len(limpo) > 60 or not _NOME_RE.match(limpo):
        raise InvalidNameError(
            "Nome inválido: use de 1 a 60 caracteres "
            "(letras, números, espaço, ponto, apóstrofo e hífen)."
        )
    return limpo


class PersonRepository:
    def __init__(self, base_dir):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._limpar_temporarios()

    # ---- cadastro -------------------------------------------------------
    def add_person(self, nome, embeddings) -> str:
        """Grava uma pessoa de forma atômica e devolve o identificador (uuid)."""
        nome = normalizar_nome(nome)
        emb = self._validar(embeddings)
        if self.nome_existe(nome):
            raise DuplicatePersonError("Já existe uma pessoa cadastrada com este nome.")

        id_ = str(uuid.uuid4())
        tmp, final = self._dir / f"{_PREFIXO_TMP}{id_}", self._dir / id_
        try:
            tmp.mkdir()
            np.save(tmp / "embeddings.npy", emb)
            meta = {
                "nome": nome,
                "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_embeddings": int(emb.shape[0]),
            }
            (tmp / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
            tmp.rename(final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        log.info("pessoa cadastrada id=%s n_embeddings=%d", id_, emb.shape[0])
        return id_

    # ---- consulta -------------------------------------------------------
    def list_people(self) -> list[Person]:
        pessoas = []
        for pasta in self._dir.iterdir():
            if not pasta.is_dir() or pasta.name.startswith(_PREFIXO_TMP):
                continue
            p = self._ler_pessoa(pasta)
            if p is not None:
                pessoas.append(p)
        return sorted(pessoas, key=lambda p: p.nome.casefold())

    def count(self) -> int:
        return len(self.list_people())

    def nome_existe(self, nome) -> bool:
        chave = normalizar_nome(nome).casefold()
        return any(p.nome.casefold() == chave for p in self.list_people())

    def prototipos(self):
        """Retorna (ids, nomes, matriz (K, 512)); cada linha é a média normalizada
        dos embeddings da pessoa (mesmo cálculo da Parte 1)."""
        ids, nomes, protos = [], [], []
        for p in self.list_people():
            try:
                emb = np.load(self._dir / p.id / "embeddings.npy", allow_pickle=False)
            except (OSError, ValueError):
                log.warning("embeddings ilegíveis, cadastro ignorado id=%s", p.id)
                continue
            if emb.ndim != 2 or emb.shape[0] < 1 or emb.shape[1] != EMBEDDING_DIM:
                log.warning("embeddings com forma inválida, cadastro ignorado id=%s", p.id)
                continue
            media = emb.mean(axis=0)
            norma = np.linalg.norm(media)
            if norma < 1e-6:
                continue
            ids.append(p.id)
            nomes.append(p.nome)
            protos.append(media / norma)
        if protos:
            matriz = np.stack(protos).astype(np.float32)
        else:
            matriz = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return ids, nomes, matriz

    # ---- exclusão -------------------------------------------------------
    def excluir(self, id_) -> bool:
        """Remove a pessoa por inteiro. Devolve False se o id não existir."""
        pasta = self._pasta(id_)
        if pasta is None or not pasta.is_dir():
            return False
        shutil.rmtree(pasta)
        log.info("pessoa excluida id=%s", pasta.name)
        return True

    # ---- internos -------------------------------------------------------
    def _pasta(self, id_):
        """Caminho da pasta, ou None se o id não for um uuid (evita path traversal)."""
        try:
            return self._dir / str(uuid.UUID(str(id_)))
        except ValueError:
            return None

    def _ler_pessoa(self, pasta: Path):
        try:
            uuid.UUID(pasta.name)
            meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
            return Person(
                id=pasta.name,
                nome=str(meta["nome"]),
                n_embeddings=int(meta["n_embeddings"]),
                criado_em=str(meta["criado_em"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("cadastro ilegível ignorado id=%s", pasta.name)
            return None

    @staticmethod
    def _validar(embeddings) -> np.ndarray:
        try:
            emb = np.asarray(embeddings, dtype=np.float32)
        except (ValueError, TypeError):
            raise FaceAppError("Embeddings inválidos.") from None
        if emb.ndim != 2 or emb.shape[0] < 1 or emb.shape[1] != EMBEDDING_DIM:
            raise FaceAppError(f"Embeddings inválidos: esperado (n, {EMBEDDING_DIM}).")
        if not np.isfinite(emb).all():
            raise FaceAppError("Embeddings inválidos: valores não finitos.")
        normas = np.linalg.norm(emb, axis=1, keepdims=True)
        if (normas < 1e-6).any():
            raise FaceAppError("Embeddings inválidos: vetor nulo.")
        return (emb / normas).astype(np.float32)

    def _limpar_temporarios(self):
        """Remove restos de cadastros interrompidos."""
        for pasta in self._dir.glob(f"{_PREFIXO_TMP}*"):
            shutil.rmtree(pasta, ignore_errors=True)