"""Baixa o LFW via scikit-learn para data/lfw/ (pasta ignorada pelo Git)."""
from pathlib import Path

from sklearn.datasets import fetch_lfw_people

DEST = Path(__file__).resolve().parents[2] / "data" / "lfw"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    # Dispara o download e a extração das imagens (versão funneled).
    # O retorno (arrays) é descartado: usamos os arquivos .jpg extraídos.
    fetch_lfw_people(data_home=str(DEST), download_if_missing=True)
    pasta = DEST / "lfw_home" / "lfw_funneled"
    n = sum(1 for _ in pasta.rglob("*.jpg"))
    print(f"{n} imagens em {pasta}")


if __name__ == "__main__":
    main()