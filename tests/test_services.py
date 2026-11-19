import logging

import numpy as np
import pytest

from src.face_app.config import Settings
from src.face_app.errors import (
    DuplicatePersonError, FaceAppError, InvalidImageError,
    InvalidNameError, MultipleFacesError, NoFaceError,
)
from src.face_app.services import build_services
from tests.fakes import FakeDetector, FakeEmbedder, face

import tests.fakes as fakes

THR = 0.5
IMAGENS_INVALIDAS = [
    None,
    "texto",
    np.zeros((10, 10), dtype=np.uint8),
    np.zeros((10, 10, 3), dtype=np.float32),
    np.zeros((0, 10, 3), dtype=np.uint8),
]


def settings(tmp_path, threshold=THR):
    return Settings(threshold=threshold, metric="cosseno", min_confidence=0.9,
                    min_face_px=60, log_level="INFO", data_dir=tmp_path)


def img(detector, codigo, *faces):
    detector.por_codigo[codigo] = list(faces)
    return fakes.imagem(codigo)


@pytest.fixture
def detector():
    return FakeDetector()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def servicos(tmp_path, detector, embedder):
    return build_services(settings(tmp_path), detector, embedder)


def cadastrar_ana(servicos, detector):
    return servicos.cadastro.enroll(
        "Ana", [img(detector, 1, face(1, 0)), img(detector, 2, face(1, 1))]
    )


# ---------------------------- cadastro ----------------------------
def test_cadastro_valido(servicos, detector):
    r = cadastrar_ana(servicos, detector)
    (p,) = servicos.pessoas.listar()
    assert r.n_embeddings == 2 and p.id == r.id and p.nome == "Ana"


def test_sem_face(servicos, detector, tmp_path):
    with pytest.raises(NoFaceError):
        servicos.cadastro.enroll("Ana", [img(detector, 1)])
    assert list((tmp_path / "people").iterdir()) == []


def test_varias_faces(servicos, detector):
    with pytest.raises(MultipleFacesError):
        servicos.cadastro.enroll("Ana", [img(detector, 1, face(1), face(2))])
    assert servicos.pessoas.contar() == 0


def test_face_com_baixa_confianca_e_inadequada(servicos, detector):
    with pytest.raises(InvalidImageError):
        servicos.cadastro.enroll("Ana", [img(detector, 1, face(1, conf=0.5))])


def test_face_pequena_e_inadequada(servicos, detector):
    with pytest.raises(InvalidImageError):
        servicos.cadastro.enroll("Ana", [img(detector, 1, face(1, lado=20))])


def test_face_pequena_ao_fundo_e_ignorada(servicos, detector):
    r = servicos.cadastro.enroll("Ana", [img(detector, 1, face(1), face(2, lado=20))])
    assert r.n_embeddings == 1


@pytest.mark.parametrize("ruim", IMAGENS_INVALIDAS)
def test_cadastro_com_imagem_invalida(servicos, ruim):
    with pytest.raises(InvalidImageError):
        servicos.cadastro.enroll("Ana", [ruim])
    assert servicos.pessoas.contar() == 0


def test_lista_de_imagens_vazia(servicos):
    with pytest.raises(InvalidImageError):
        servicos.cadastro.enroll("Ana", [])


def test_imagem_ruim_no_meio_nao_grava_nada(servicos, detector, tmp_path):
    boa = img(detector, 1, face(1, 0))
    sem_face = img(detector, 2)
    with pytest.raises(NoFaceError, match="Imagem 2"):
        servicos.cadastro.enroll("Ana", [boa, sem_face])
    assert servicos.pessoas.contar() == 0
    assert list((tmp_path / "people").iterdir()) == []


def test_nome_duplicado_ignora_caixa(servicos, detector):
    cadastrar_ana(servicos, detector)
    with pytest.raises(DuplicatePersonError):
        servicos.cadastro.enroll("ana", [img(detector, 3, face(2, 0))])
    assert servicos.pessoas.contar() == 1


def test_mesmo_rosto_com_outro_nome_e_recusado(servicos, detector):
    cadastrar_ana(servicos, detector)
    with pytest.raises(DuplicatePersonError):
        servicos.cadastro.enroll("Bia", [img(detector, 3, face(1, 5))])
    assert servicos.pessoas.contar() == 1


def test_rosto_diferente_e_aceito(servicos, detector):
    cadastrar_ana(servicos, detector)
    servicos.cadastro.enroll("Carla", [img(detector, 4, face(2, 0))])
    assert servicos.pessoas.contar() == 2


class EmbedderRuim:
    def __init__(self, valor):
        self.valor = valor

    def embed(self, recorte):
        return self.valor


@pytest.mark.parametrize(
    "valor", [np.full(512, np.nan), np.ones(10), np.zeros(512), "texto"]
)
def test_embedding_invalido_nao_grava(tmp_path, detector, valor):
    s = build_services(settings(tmp_path), detector, EmbedderRuim(valor))
    with pytest.raises(FaceAppError):
        s.cadastro.enroll("Ana", [img(detector, 1, face(1))])
    assert list((tmp_path / "people").iterdir()) == []


def test_nome_invalido_nao_chama_o_extrator(servicos, detector, embedder):
    with pytest.raises(InvalidNameError):
        servicos.cadastro.enroll("../x", [img(detector, 1, face(1))])
    assert embedder.chamadas == 0


def test_falha_inesperada_do_detector_vira_erro_claro(tmp_path, embedder):
    class Quebrado:
        def detect(self, imagem):
            raise RuntimeError("boom")

    s = build_services(settings(tmp_path), Quebrado(), embedder)
    with pytest.raises(FaceAppError, match="Falha ao detectar"):
        s.cadastro.enroll("Ana", [fakes.imagem(1)])


# -------------------------- reconhecimento --------------------------
def test_reconhece_pessoa_cadastrada(servicos, detector):
    cadastrar_ana(servicos, detector)
    (r,) = servicos.reconhecimento.reconhecer(img(detector, 9, face(1, 3)))
    assert r.nome == "Ana" and r.similaridade >= THR


def test_pessoa_nao_cadastrada_e_desconhecida(servicos, detector):
    cadastrar_ana(servicos, detector)
    (r,) = servicos.reconhecimento.reconhecer(img(detector, 9, face(2, 0)))
    assert r.nome is None and r.similaridade < THR


def test_imagem_sem_face_devolve_lista_vazia(servicos, detector):
    cadastrar_ana(servicos, detector)
    assert servicos.reconhecimento.reconhecer(img(detector, 9)) == []


def test_duas_faces_sao_independentes(servicos, detector):
    cadastrar_ana(servicos, detector)
    res = servicos.reconhecimento.reconhecer(img(detector, 9, face(1, 3), face(2, 0)))
    assert [r.nome for r in res] == ["Ana", None]


def test_sem_pessoas_cadastradas_tudo_e_desconhecido(servicos, detector):
    assert servicos.reconhecimento.n_pessoas() == 0
    (r,) = servicos.reconhecimento.reconhecer(img(detector, 9, face(1)))
    assert r.nome is None


def test_threshold_igual_aceita_e_acima_rejeita(tmp_path, detector, embedder):
    ref = build_services(settings(tmp_path, threshold=-1.0), detector, embedder)
    cadastrar_ana(ref, detector)
    consulta = img(detector, 9, face(1, 2))
    sim = ref.reconhecimento.reconhecer(consulta)[0].similaridade
    aceita = build_services(settings(tmp_path, threshold=sim), detector, embedder)
    rejeita = build_services(settings(tmp_path, threshold=sim + 1e-6), detector, embedder)
    assert aceita.reconhecimento.reconhecer(consulta)[0].nome == "Ana"
    assert rejeita.reconhecimento.reconhecer(consulta)[0].nome is None


def test_faces_inadequadas_sao_ignoradas(servicos, detector):
    cadastrar_ana(servicos, detector)
    consulta = img(detector, 9, face(1, 3, conf=0.5), face(1, 3, lado=20))
    assert servicos.reconhecimento.reconhecer(consulta) == []


@pytest.mark.parametrize("ruim", IMAGENS_INVALIDAS)
def test_reconhecer_imagem_invalida(servicos, ruim):
    with pytest.raises(InvalidImageError):
        servicos.reconhecimento.reconhecer(ruim)


def test_reconhecer_nao_escreve_em_disco(servicos, detector, tmp_path):
    cadastrar_ana(servicos, detector)
    antes = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    servicos.reconhecimento.reconhecer(img(detector, 9, face(1, 3)))
    depois = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert antes == depois


def test_pessoa_excluida_nao_e_mais_reconhecida(servicos, detector):
    r = cadastrar_ana(servicos, detector)
    consulta = img(detector, 9, face(1, 3))
    assert servicos.reconhecimento.reconhecer(consulta)[0].nome == "Ana"
    assert servicos.pessoas.excluir(r.id) is True
    assert servicos.reconhecimento.reconhecer(consulta)[0].nome is None
    assert servicos.reconhecimento.n_pessoas() == 0


# ------------------------ pessoas e privacidade ------------------------
def test_listar_contar_e_excluir_inexistente(servicos, detector):
    cadastrar_ana(servicos, detector)
    servicos.cadastro.enroll("Bia", [img(detector, 4, face(2, 0))])
    assert [p.nome for p in servicos.pessoas.listar()] == ["Ana", "Bia"]
    assert servicos.pessoas.contar() == 2
    assert servicos.pessoas.excluir("nao-e-um-uuid") is False


def test_logs_nao_contem_nomes(servicos, detector, caplog):
    caplog.set_level(logging.DEBUG, logger="face_app")
    r = servicos.cadastro.enroll("Ana Teste", [img(detector, 1, face(1, 0))])
    servicos.reconhecimento.reconhecer(img(detector, 2, face(1, 1)))
    servicos.pessoas.excluir(r.id)
    assert caplog.text
    assert "Ana" not in caplog.text and "Teste" not in caplog.text