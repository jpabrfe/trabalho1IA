import json
import uuid

import numpy as np
import pytest

from src.face_app.errors import DuplicatePersonError, FaceAppError, InvalidNameError
from src.face_app.storage.repository import EMBEDDING_DIM, PersonRepository


def emb(seed, n=3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, EMBEDDING_DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def pasta(tmp_path):
    return tmp_path / "people"


@pytest.fixture
def repo(pasta):
    return PersonRepository(pasta)


def test_repositorio_novo_esta_vazio(repo):
    ids, nomes, protos = repo.prototipos()
    assert repo.list_people() == [] and repo.count() == 0
    assert ids == [] and nomes == [] and protos.shape == (0, EMBEDDING_DIM)


def test_cadastro_aparece_na_listagem(repo):
    id_ = repo.add_person("Ana", emb(1))
    (p,) = repo.list_people()
    assert p.id == id_ and p.nome == "Ana" and p.n_embeddings == 3


def test_prototipo_e_a_media_normalizada(repo):
    e = emb(2)
    repo.add_person("Ana", e)
    _, nomes, protos = repo.prototipos()
    esperado = e.mean(axis=0)
    esperado /= np.linalg.norm(esperado)
    assert nomes == ["Ana"]
    np.testing.assert_allclose(protos[0], esperado, atol=1e-5)
    assert np.linalg.norm(protos[0]) == pytest.approx(1.0, abs=1e-5)


def test_nome_duplicado_ignora_caixa_e_espacos(repo, pasta):
    repo.add_person("Ana  Souza", emb(1))
    with pytest.raises(DuplicatePersonError):
        repo.add_person("ana souza", emb(2))
    assert repo.count() == 1
    assert len(list(pasta.iterdir())) == 1  # nenhum resto do cadastro recusado


@pytest.mark.parametrize("nome", ["", "   ", "a" * 61, "x/../y", "..\\x", "nome\x00", None])
def test_nomes_invalidos(repo, nome):
    with pytest.raises(InvalidNameError):
        repo.add_person(nome, emb(1))
    assert repo.count() == 0


@pytest.mark.parametrize(
    "ruim",
    [
        np.ones((3, 10)),
        np.empty((0, EMBEDDING_DIM)),
        np.full((2, EMBEDDING_DIM), np.nan),
        np.zeros((2, EMBEDDING_DIM)),
        np.ones(EMBEDDING_DIM),
    ],
)
def test_embeddings_invalidos_nao_gravam_nada(repo, pasta, ruim):
    with pytest.raises(FaceAppError):
        repo.add_person("Ana", ruim)
    assert list(pasta.iterdir()) == []


def test_cadastro_e_atomico_quando_a_gravacao_falha(repo, pasta, monkeypatch):
    def falha(*args, **kwargs):
        raise OSError("disco cheio")

    monkeypatch.setattr(np, "save", falha)
    with pytest.raises(OSError):
        repo.add_person("Ana", emb(1))
    assert list(pasta.iterdir()) == []
    assert repo.count() == 0


def test_excluir_remove_tudo_da_pessoa(repo, pasta):
    id_ = repo.add_person("Ana", emb(1))
    assert repo.excluir(id_) is True
    assert repo.list_people() == []
    assert repo.prototipos()[2].shape == (0, EMBEDDING_DIM)
    assert list(pasta.iterdir()) == []  # nenhum arquivo dela permanece


def test_excluir_id_inexistente_ou_malicioso(repo, tmp_path):
    fora = tmp_path / "fora.txt"
    fora.write_text("não apagar")
    assert repo.excluir(str(uuid.uuid4())) is False
    for ruim in ("../fora.txt", "..", "", "../../x", None):
        assert repo.excluir(ruim) is False
    assert fora.exists()


def test_excluir_preserva_as_outras_pessoas(repo):
    a = repo.add_person("Ana", emb(1))
    repo.add_person("Bia", emb(2))
    repo.excluir(a)
    assert [p.nome for p in repo.list_people()] == ["Bia"]


def test_cadastro_ilegivel_e_ignorado(repo, pasta):
    ruim = repo.add_person("Ana", emb(1))
    repo.add_person("Bia", emb(2))
    (pasta / ruim / "meta.json").write_text("{", encoding="utf-8")
    assert [p.nome for p in repo.list_people()] == ["Bia"]
    assert repo.prototipos()[1] == ["Bia"]


def test_dados_persistem_entre_instancias(repo, pasta):
    repo.add_person("Ana", emb(1))
    assert [p.nome for p in PersonRepository(pasta).list_people()] == ["Ana"]


def test_temporarios_orfaos_sao_limpos_ao_iniciar(pasta):
    pasta.mkdir()
    orfao = pasta / ".tmp-abc"
    orfao.mkdir()
    (orfao / "x.npy").write_bytes(b"0")
    PersonRepository(pasta)
    assert not orfao.exists()


def test_nome_com_acentos_e_gravado_corretamente(repo, pasta):
    id_ = repo.add_person("José D'Ávila-Silva", emb(1))
    meta = json.loads((pasta / id_ / "meta.json").read_text(encoding="utf-8"))
    assert meta["nome"] == "José D'Ávila-Silva"