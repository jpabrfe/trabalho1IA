"""Detecta e alinha as faces do LFW com MTCNN.

Saída: data/lfw/aligned/<pessoa>/<arquivo>.jpg (160x160), mesmo caminho relativo
do LFW original, então o lfw_split.json vale também para as imagens alinhadas.
"""
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "lfw" / "lfw_home" / "lfw_funneled"
DST = ROOT / "data" / "lfw" / "aligned"
FALHAS = ROOT / "data" / "lfw" / "aligned_falhas.txt"


def main():
    torch.manual_seed(42)
    mtcnn = MTCNN(
        image_size=160, margin=0, select_largest=True,
        post_process=False, device="cpu",
    )
    arquivos = sorted(SRC.rglob("*.jpg"))
    falhas = []
    for f in tqdm(arquivos):
        rel = f.relative_to(SRC)
        saida = DST / rel
        if saida.exists():          # permite retomar de onde parou
            continue
        saida.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(f).convert("RGB")
        if mtcnn(img, save_path=str(saida)) is None:
            falhas.append(rel.as_posix())

    FALHAS.write_text("\n".join(falhas), encoding="utf-8")
    total = sum(1 for _ in DST.rglob("*.jpg"))
    print(f"Alinhadas: {total} | sem face detectada: {len(falhas)}")
    print(f"Lista das falhas em {FALHAS}")


if __name__ == "__main__":
    main()