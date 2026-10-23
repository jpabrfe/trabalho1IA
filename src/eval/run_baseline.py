"""Baseline E01/E02: embeddings pré-treinados + protótipos + threshold na validação.

Uso (na raiz, com o ambiente ativo):
    python -m src.eval.run_baseline --modelo facenet
    python -m src.eval.run_baseline --modelo arcface --metrica euclidiana
    python -m src.eval.run_baseline --modelo facenet --teste   # SÓ no modelo final, uma vez

Sem --teste, o script avalia apenas na validação (onde o threshold é escolhido).
Dados: só LFW (público). Nenhuma foto privada passa por aqui.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.eval.metrics import (
    DESCONHECIDO, calcular_metricas, escolher_threshold,
    plotar_distribuicoes, prototipos, similaridades,
)
from src.utils.repro import fixar_seed, registrar

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "splits" / "lfw_split.json"
ALIGNED = ROOT / "data" / "lfw" / "aligned"
EMB = ROOT / "data" / "embeddings"


def extrator(modelo, ckpt=None):
    from src.models.embedder import _facenet_model, arcface_embedding, facenet_embedding

    if modelo == "facenet":
        if ckpt:
            import torch

            m = _facenet_model()  # instância em cache, a mesma que o facenet_embedding usa
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["modelo"])
            m.eval()
        return lambda bgr: facenet_embedding(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return arcface_embedding


def embeddings_de(modelo, rels, ckpt=None):
    """Embeddings das imagens alinhadas, com cache em data/embeddings (ignorada pelo Git)."""
    EMB.mkdir(parents=True, exist_ok=True)
    tag = f"_{Path(ckpt).stem}" if ckpt else ""
    cache = EMB / f"lfw_{modelo}{tag}.npz"
    if cache.exists():
        dados = np.load(cache, allow_pickle=False)
        idx = {r: i for i, r in enumerate(dados["rels"])}
        if all(r in idx for r in rels):
            return dados["emb"][[idx[r] for r in rels]]
    f = extrator(modelo, ckpt)
    embs = []
    for r in tqdm(rels, desc=f"embeddings {modelo}{tag}"):
        img = cv2.imread(str(ALIGNED / r))
        if img is None:
            raise FileNotFoundError(f"imagem alinhada não encontrada: {r}")
        embs.append(f(img))
    embs = np.stack(embs).astype(np.float32)
    np.savez(cache, rels=np.array(rels), emb=embs)
    return embs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=["facenet", "arcface"], required=True)
    p.add_argument("--metrica", choices=["cosseno", "euclidiana"], default="cosseno")
    p.add_argument("--ckpt", default=None, help="checkpoint do fine-tuning (só FaceNet)")
    p.add_argument("--teste", action="store_true",
                   help="avalia no TESTE (uma única vez, no modelo já escolhido)")
    a = p.parse_args()
    if a.ckpt and a.modelo != "facenet":
        raise SystemExit("--ckpt só vale para o FaceNet.")
    fixar_seed(42)

    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    known = s["known"]
    nomes_known = sorted(known)
    enroll = {n: known[n]["enroll"] for n in nomes_known}
    val = [(f, n) for n in nomes_known for f in known[n]["val"]]
    val += [(f, DESCONHECIDO) for f in s["impostors_val"]]
    test = [(f, n) for n in nomes_known for f in known[n]["test"]]
    test += [(f, DESCONHECIDO) for f in s["impostors_test"]]

    # Calcula embeddings de tudo (inclusive teste); o teste só é AVALIADO com --teste.
    todos = sorted({f for fs in enroll.values() for f in fs} | {f for f, _ in val + test})
    emb = dict(zip(todos, embeddings_de(a.modelo, todos, a.ckpt)))

    nomes, protos = prototipos(
        {n: np.stack([emb[f] for f in enroll[n]]) for n in nomes_known}
    )

    def avaliar(pares):
        X = np.stack([emb[f] for f, _ in pares])
        y = np.array([r for _, r in pares])
        return similaridades(X, protos, a.metrica), y

    exp = f"{a.modelo}_{a.metrica}" + (f"_ft_{Path(a.ckpt).stem}" if a.ckpt else "")
    figs = ROOT / "docs" / "figures"

    sv, yv = avaliar(val)
    thr, eer = escolher_threshold(sv, yv, nomes)  # threshold: SÓ na validação
    res_val, _, _ = calcular_metricas(sv, yv, nomes, thr)
    plotar_distribuicoes(sv, yv, nomes, thr, figs / f"{exp}_val.png", f"{exp} (validação)")
    resultados = {"eer_val": eer, "validacao": res_val}
    print(f"[{exp}] threshold={thr:.4f}  EER(val)={eer:.4f}")
    print("  validação:", {k: round(v, 4) for k, v in res_val.items()})

    if a.teste:
        st, yt = avaliar(test)
        res_t, _, _ = calcular_metricas(st, yt, nomes, thr)
        plotar_distribuicoes(st, yt, nomes, thr, figs / f"{exp}_teste.png", f"{exp} (teste)")
        resultados["teste"] = res_t
        print("  TESTE:", {k: round(v, 4) for k, v in res_t.items()})

    registrar(
        f"{exp}_{'TESTE' if a.teste else 'val'}",
        config={
            "modelo": a.modelo, "metrica": a.metrica, "k": len(nomes),
            "n_cadastro": len(enroll[nomes[0]]),
            "pesos": f"fine-tuning: {Path(a.ckpt).name}" if a.ckpt else "pré-treinados, sem fine-tuning",
        },
        resultados=resultados, seed=42,
    )


if __name__ == "__main__":
    main()