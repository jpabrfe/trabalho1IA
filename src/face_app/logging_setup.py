"""Logs da aplicação. PRIVACIDADE: nunca registrar nomes, embeddings nem imagens;
usar apenas identificadores internos (uuid) e tipos de evento."""
import logging
import sys
from pathlib import Path


def setup_logging(level="INFO", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("face_app")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:          # evita handlers duplicados (Streamlit reexecuta o script)
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        arquivo = logging.FileHandler(log_file, encoding="utf-8")
        arquivo.setFormatter(fmt)
        logger.addHandler(arquivo)
    return logger