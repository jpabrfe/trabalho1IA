"""Verifica que o pipeline da aplicação (MtcnnDetector + ArcFaceEmbedder) reproduz o da Parte 1.

Para imagens originais do LFW, compara o embedding gerado pela aplicação com o
embedding da Parte 1 (calculado sobre os recortes salvos em data/lfw/aligned e
guardado em data/embeddings/lfw_arcface.npz). Só usa o LFW, que é público.
Uso: python -m src.eval.check_app_equivalence [N]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from src.face_app.core.detector import MtcnnDetector
from src.face_app.core.embedder import ArcFaceEmbedder

ROOT = Path(__file__).resolve().parents[2]
LFW = ROOT / "data" / "lfw" / "lfw_home" / "lfw_funneled"
CACHE = ROOT / "data" / "embeddings" / "lfw_arcface.npz"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    if not CACHE.exists():
        raise SystemExit("Cache ausente: rode antes `python -m src.eval.run_baseline --modelo arcface`.")
    dados = np.load(CACHE, allow_pickle=False)
    rels, ref = [str(r) for r in dados["rels"]], dados["emb"]

    idx = np.random.default_rng(42).choice(len(rels), size=min(n, len(rels)), replace=False)
    detector, embedder = MtcnnDetector(), ArcFaceEmbedder()

    sims, sem_face = [], 0
    for i in idx:
        faces = detector.detect(cv2.imread(str(LFW / rels[i])))
        if not faces:
            sem_face += 1
            continue
        maior = max(faces, key=lambda f: f.caixa[2] * f.caixa[3])  # como select_largest da Parte 1
        v = embedder.embed(maior.recorte)
        r = ref[i] / np.linalg.norm(ref[i])
        sims.append(float(v @ r))

    s = np.array(sims)
    print(f"imagens avaliadas: {len(s)} | sem face detectada: {sem_face}")
    print(f"similaridade app x Parte 1: min={s.min():.4f} p5={np.percentile(s, 5):.4f} "
          f"mediana={np.median(s):.4f} media={s.mean():.4f}")
    print(f"fracao com similaridade >= 0,95: {(s >= 0.95).mean():.2%}")


if __name__ == "__main__":
    main()