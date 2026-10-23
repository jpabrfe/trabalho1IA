"""E01 (FaceNet, imagens alinhadas) e E02 (ArcFace, imagens originais).
Suporta os splits 'val' ou 'test' via argumento --split.
Uso (da raiz): python -m src.eval.extract_embeddings --split test [limite]
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from src.models.embedder import facenet_embedding, arcface_embedding

SPLIT = Path("data/splits/lfw_split.json")
RAW_DIR = Path("data/lfw")
ALIGNED_DIR = Path("data/lfw/aligned")
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

def extract(name, fn, files, index, to_rgb, split_name):
    suffix = "_test" if split_name == "test" else ""
    emb_filename = OUT / f"{name}{suffix}_embeddings.npy"
    files_filename = OUT / f"{name}{suffix}_files.txt"

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
            
    np.save(emb_filename, emb)
    files_filename.write_text("\n".join(files), encoding="utf-8")
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    print(f"{name} ({split_name}): {emb.shape} | NaN={n_nan} "
          f"(não encontrada={missing}, ilegível={unreadable}, falha/sem face={failed})")

def main():
    parser = argparse.ArgumentParser(description="Extração de embeddings para LFW")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split a processar: val ou test")
    parser.add_argument("limit", type=int, nargs="?", default=None, help="Limite opcional de imagens")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    
    if not SPLIT.exists():
        print(f"[ERRO] Arquivo de split não encontrado: {SPLIT}")
        return

    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    
    # Seleciona impostors conforme o split escolhido
    impostors_key = "impostors_test" if args.split == "test" else "impostors_val"
    if impostors_key not in s:
        print(f"[ERRO] Chave '{impostors_key}' não encontrada em {SPLIT}")
        return

    files = list(dict.fromkeys(collect(s["known"]) + collect(s[impostors_key])))
    if args.limit:
        files = files[:args.limit]
        
    print(f"Imagens do split '{args.split}' (known + {impostors_key}): {len(files)}")

    # E02: ArcFace, imagens originais
    extract("E02_arcface", arcface_embedding, files, build_index(RAW_DIR), to_rgb=False, split_name=args.split)

    # E01: FaceNet, imagens alinhadas (RGB)
    if ALIGNED_DIR.exists():
        extract("E01_facenet", facenet_embedding, files, build_index(ALIGNED_DIR), to_rgb=True, split_name=args.split)
    else:
        print(f"[AVISO] {ALIGNED_DIR} não existe: E01 pulado até o align.py rodar.")

if __name__ == "__main__":
    main()