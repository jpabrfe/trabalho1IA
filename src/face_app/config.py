"""Configuração externa (ADR-005): config.yaml + variáveis de ambiente.

O threshold não tem valor padrão no código: precisa estar no arquivo ou no ambiente.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.face_app.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"

# chave -> (variável de ambiente, tipo)
_CAMPOS = {
    "threshold": ("FACE_THRESHOLD", float),
    "metric": ("FACE_METRIC", str),
    "min_confidence": ("FACE_MIN_CONFIDENCE", float),
    "min_face_px": ("FACE_MIN_FACE_PX", int),
}
_NIVEIS_LOG = {"DEBUG", "INFO", "WARNING", "ERROR"}


@dataclass(frozen=True)
class Settings:
    threshold: float
    metric: str
    min_confidence: float
    min_face_px: int
    log_level: str
    data_dir: Path

    @property
    def people_dir(self) -> Path:
        return self.data_dir / "people"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "logs" / "app.log"


def _como(tipo, valor, nome):
    try:
        return tipo(valor)
    except (TypeError, ValueError):
        raise ConfigError(f"Valor inválido para {nome}: {valor!r}.") from None


def load_settings(config_path=None, env=None) -> Settings:
    env = os.environ if env is None else env
    caminho = Path(config_path) if config_path else DEFAULT_CONFIG
    try:
        bruto = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"Arquivo de configuração não encontrado: {caminho.name}.") from None
    except yaml.YAMLError:
        raise ConfigError("Arquivo de configuração inválido (YAML).") from None
    if not isinstance(bruto, dict):
        raise ConfigError("Arquivo de configuração inválido: esperado um mapa chave/valor.")

    valores = {}
    for chave, (var, tipo) in _CAMPOS.items():
        if env.get(var):
            valores[chave] = _como(tipo, env[var], var)
        elif chave in bruto:
            valores[chave] = _como(tipo, bruto[chave], chave)
        else:
            raise ConfigError(f"Configuração ausente: {chave}.")
    nivel = str(env.get("FACE_LOG_LEVEL") or bruto.get("log_level", "INFO")).upper()

    if valores["metric"] != "cosseno":
        raise ConfigError("Métrica não suportada: use 'cosseno'.")
    if not -1.0 <= valores["threshold"] <= 1.0:
        raise ConfigError("threshold deve estar entre -1 e 1.")
    if not 0.0 < valores["min_confidence"] <= 1.0:
        raise ConfigError("min_confidence deve estar entre 0 (exclusivo) e 1.")
    if valores["min_face_px"] < 1:
        raise ConfigError("min_face_px deve ser positivo.")
    if nivel not in _NIVEIS_LOG:
        raise ConfigError(f"log_level inválido: {nivel}.")

    base = env.get("FACE_PRIVATE_DIR", "")
    if not base:
        raise ConfigError(
            "Defina a variável de ambiente FACE_PRIVATE_DIR "
            "(pasta de dados privados, fora do repositório)."
        )
    data_dir = Path(base).expanduser().resolve()
    if data_dir == ROOT or ROOT in data_dir.parents:
        raise ConfigError("FACE_PRIVATE_DIR está dentro do repositório; use uma pasta fora dele.")

    return Settings(log_level=nivel, data_dir=data_dir, **valores)