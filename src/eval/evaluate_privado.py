"""Avaliação PRIVADA, local: você entra como mais uma pessoa conhecida, junto das
50 conhecidas do LFW. Threshold fixado na VALIDAÇÃO do LFW (sem fotos privadas).

Usa o conjunto de TESTE do LFW, então só rode UMA vez, no modelo final, com --final.
PRIVACIDADE: lê fotos de FACE_PRIVATE_DIR/aligned/{cadastro,teste}; imprime e
grava (em FACE_PRIVATE_DIR/results) só números agregados, nunca nomes nem caminhos.
Uso: python -m src.eval.evaluate_privado --modelo facenet --final
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.eval.metrics import (
    DESCONHECIDO, calcular_metricas, decidir, escolher_threshold,
    prototipos, similaridades,
)
from src.eval.run_baseline import embeddings_de, extrator

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "splits" / "lfw_split.json"
EU = "participante"          # rótulo genérico, sem nome real
EXT = {".jpg", ".jpeg", ".png"}


def embeddings_privados(modelo, pasta):
    f = extrator(modelo)
    arquivos = sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXT)
    if not arquivos:
        raise SystemExit(f"Nenhuma foto alinhada em {pasta.name}/.")
    embs = []
    for p in arquivos:
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit("Falha ao ler uma foto alinhada.")
        embs.append(f(img))
    return np.stack(embs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["facenet", "arcface"], required=True)
    ap.add_argument("--metrica", choices=["cosseno", "euclidiana"], default="cosseno")
    ap.add_argument("--final", action="store_true",
                    help="confirma que é a avaliação final (usa o teste do LFW)")
    a = ap.parse_args()
    if not a.final:
        raise SystemExit("Use --final só quando o modelo já estiver escolhido pela validação.")

    base = os.environ.get("FACE_PRIVATE_DIR")
    if not base:
        raise SystemExit("Defina FACE_PRIVATE_DIR.")
    base = Path(base).resolve()
    if base == ROOT or ROOT in base.parents:
        raise SystemExit("FACE_PRIVATE_DIR está dentro do repositório: abortando.")

    # --- LFW: gallery de 50 conhecidas, threshold na validação ---
    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    known = s["known"]
    nomes_known = sorted(known)
    val = [(f, n) for n in nomes_known for f in known[n]["val"]]
    val += [(f, DESCONHECIDO) for f in s["impostors_val"]]
    test = [(f, n) for n in nomes_known for f in known[n]["test"]]
    test += [(f, DESCONHECIDO) for f in s["impostors_test"]]
    enroll = {n: known[n]["enroll"] for n in nomes_known}

    rels = sorted({f for fs in enroll.values() for f in fs} | {f for f, _ in val + test})
    emb = dict(zip(rels, embeddings_de(a.modelo, rels)))   # cache do run_baseline
    emb_enroll = {n: np.stack([emb[f] for f in enroll[n]]) for n in nomes_known}

    nomes, protos = prototipos(emb_enroll)
    sv = similaridades(np.stack([emb[f] for f, _ in val]), protos, a.metrica)
    thr, eer = escolher_threshold(sv, np.array([r for _, r in val]), nomes)

    # --- Gallery com você como mais uma pessoa (K = 51) ---
    cad = embeddings_privados(a.modelo, base / "aligned" / "cadastro")
    tst = embeddings_privados(a.modelo, base / "aligned" / "teste")
    nomes2, protos2 = prototipos({**emb_enroll, EU: cad})

    X = np.vstack([np.stack([emb[f] for f, _ in test]), tst])
    y = np.array([r for _, r in test] + [EU] * len(tst))
    sims = similaridades(X, protos2, a.metrica)
    res, _, _ = calcular_metricas(sims, y, nomes2, thr)
    prev, _, _ = decidir(sims, nomes2, thr)

    eu, imp = y == EU, y == DESCONHECIDO
    priv = {
        "n_cadastro": int(len(cad)), "n_teste": int(len(tst)),
        "reconhecida_corretamente": float((prev[eu] == EU).mean()),
        "rejeitada_como_desconhecida": float((prev[eu] == DESCONHECIDO).mean()),
        "confundida_com_outra_conhecida": float(((prev[eu] != EU) & (prev[eu] != DESCONHECIDO)).mean()),
        "impostoras_aceitas_como_participante": float((prev[imp] == EU).mean()),
    }
    saida = {"modelo": a.modelo, "metrica": a.metrica, "threshold_val": thr,
             "eer_val": eer, "geral_k51": res, "participante": priv}
    destino = base / "results"
    destino.mkdir(exist_ok=True)
    (destino / f"privado_{a.modelo}_{a.metrica}.json").write_text(
        json.dumps(saida, indent=2), encoding="utf-8")

    print(f"threshold (validação LFW) = {thr:.4f}")
    print("geral (K=51):", {k: round(v, 4) for k, v in res.items()})
    print("participante:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in priv.items()})


if __name__ == "__main__":
    main()