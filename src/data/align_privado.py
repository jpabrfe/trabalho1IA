"""Alinha as fotos PRIVADAS com MTCNN. Lê e grava só dentro de FACE_PRIVATE_DIR.

PRIVACIDADE: não copie a saída para o repositório nem rode com agentes de IA
apontados para essa pasta. Uso: python -m src.data.align_privado
"""
import os
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
EXT = {".jpg", ".jpeg", ".png"}
LADO_MAX = 1280   # reduz fotos grandes de celular antes de detectar


def main():
    base = os.environ.get("FACE_PRIVATE_DIR")
    if not base:
        raise SystemExit("Defina FACE_PRIVATE_DIR (feche e reabra o terminal após o setx).")
    base = Path(base).resolve()
    if base == ROOT or ROOT in base.parents:
        raise SystemExit("FACE_PRIVATE_DIR está dentro do repositório: abortando.")

    src, dst = base / "photos", base / "aligned"
    fotos = sorted(f for f in src.rglob("*") if f.suffix.lower() in EXT) if src.exists() else []
    if not fotos:
        print("Nenhuma foto (.jpg/.jpeg/.png) em photos/.")
        return

    torch.manual_seed(42)
    mtcnn = MTCNN(image_size=160, margin=0, select_largest=True,
                  post_process=False, device="cpu")
    falhas = []
    for f in tqdm(fotos):
        rel = f.relative_to(src)
        saida = (dst / rel).with_suffix(".jpg")
        if saida.exists():
            continue
        saida.parent.mkdir(parents=True, exist_ok=True)
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")  # corrige rotação do celular
        img.thumbnail((LADO_MAX, LADO_MAX))
        if mtcnn(img, save_path=str(saida)) is None:
            falhas.append(rel.as_posix())

    (base / "aligned_falhas.txt").write_text("\n".join(falhas), encoding="utf-8")
    total = sum(1 for _ in dst.rglob("*.jpg"))
    print(f"Alinhadas: {total} | sem face detectada: {len(falhas)}")  # sem nomes no terminal


if __name__ == "__main__":
    main()