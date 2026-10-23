"""E01 (FaceNet, imagens alinhadas) e E02 (ArcFace, imagens originais).
Usa apenas known + impostors_val do split. Não toca em train nem impostors_test.
Uso (da raiz): python -m src.eval.extract_embeddings [limite]
"""
import sys
import json
from pathlib import Path
import numpy as np
import cv2
from src.models.embedder import facenet_embedding, arcface_embedding

SPLIT = Path("data/splits/lfw_split.json")
RAW_DIR = Path("data/lfw")
ALIGNED_DIR = Path("data/lfw/aligned")   # saída real do align.py
OUT = Path("data/embeddings")
EXTS = (".jpg", ".jpeg", ".png")

def collect(node):
    """Coleta recursivamente todas as strings que são caminhos de imagem."""
    if isinstance(node, str):
        return [node] if node.lower().endswith(EXTS) else []
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        out = []
        for x in node:
            out += collect(x)
        return out
    return []

def build_index(base):
    return {p.name: p for p in base.rglob("*") if p.suffix.lower() in EXTS}

def l2(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / np.linalg.norm(v)

def extract(name, fn, files, index, to_rgb):
    emb = np.full((len(files), 512), np.nan, dtype=np.float32)
    missing = unreadable = failed = 0
    for i, f in enumerate(files):
        p = index.get(Path(f).name)
        if p is None:
            missing += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            unreadable += 1
            continue
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            v = fn(img)
            if v is None:
                failed += 1
                continue
            emb[i] = l2(v)
        except Exception:
            failed += 1
            
    np.save(OUT / f"{name}_embeddings.npy", emb)
    (OUT / f"{name}_files.txt").write_text("\n".join(files), encoding="utf-8")
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    print(f"{name}: {emb.shape} | NaN={n_nan} "
          f"(não encontrada={missing}, ilegível={unreadable}, falha/sem face={failed})")

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    
    if not SPLIT.exists():
        print(f"[ERRO] Arquivo de split não encontrado: {SPLIT}")
        return

    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    files = list(dict.fromkeys(collect(s["known"]) + collect(s["impostors_val"])))
    if limit:
        files = files[:limit]
        
    print("Imagens do split (known + impostors_val):", len(files))

    # E02: ArcFace, imagens originais (o InsightFace detecta sozinho; recebe BGR)
    extract("E02_arcface", arcface_embedding, files, build_index(RAW_DIR), to_rgb=False)

    # E01: FaceNet, imagens alinhadas (RGB)
    if ALIGNED_DIR.exists():
        extract("E01_facenet", facenet_embedding, files, build_index(ALIGNED_DIR), to_rgb=True)
    else:
        print(f"[AVISO] {ALIGNED_DIR} não existe: E01 pulado até o align.py rodar.")

if __name__ == "__main__":
    main()