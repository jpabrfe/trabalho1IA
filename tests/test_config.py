import pytest

from src.face_app.config import ROOT, load_settings
from src.face_app.errors import ConfigError

BASE = (
    "threshold: 0.2679\nmetric: cosseno\nmin_confidence: 0.9\n"
    "min_face_px: 60\nlog_level: INFO\n"
)


def yaml_em(tmp_path, texto=BASE):
    caminho = tmp_path / "config.yaml"
    caminho.write_text(texto, encoding="utf-8")
    return caminho


def ambiente(tmp_path, **extra):
    return {"FACE_PRIVATE_DIR": str(tmp_path / "dados"), **extra}


def test_carrega_valores_do_arquivo(tmp_path):
    s = load_settings(yaml_em(tmp_path), ambiente(tmp_path))
    assert s.threshold == pytest.approx(0.2679)
    assert s.metric == "cosseno"
    assert s.min_face_px == 60
    assert s.data_dir == (tmp_path / "dados").resolve()


def test_ambiente_sobrescreve_threshold(tmp_path):
    s = load_settings(yaml_em(tmp_path), ambiente(tmp_path, FACE_THRESHOLD="0.5"))
    assert s.threshold == pytest.approx(0.5)


def test_sem_face_private_dir_recusa(tmp_path):
    with pytest.raises(ConfigError, match="FACE_PRIVATE_DIR"):
        load_settings(yaml_em(tmp_path), {})


def test_data_dir_dentro_do_repositorio_recusa(tmp_path):
    with pytest.raises(ConfigError, match="repositório"):
        load_settings(yaml_em(tmp_path), {"FACE_PRIVATE_DIR": str(ROOT / "dados")})


def test_threshold_fora_do_intervalo(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(yaml_em(tmp_path, BASE.replace("0.2679", "3")), ambiente(tmp_path))


def test_threshold_ausente_nao_tem_padrao_no_codigo(tmp_path):
    texto = BASE.replace("threshold: 0.2679\n", "")
    with pytest.raises(ConfigError, match="threshold"):
        load_settings(yaml_em(tmp_path, texto), ambiente(tmp_path))


def test_valor_de_ambiente_nao_numerico(tmp_path):
    with pytest.raises(ConfigError, match="FACE_THRESHOLD"):
        load_settings(yaml_em(tmp_path), ambiente(tmp_path, FACE_THRESHOLD="abc"))


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nao_existe.yaml", ambiente(tmp_path))