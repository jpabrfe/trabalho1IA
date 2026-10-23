"""Avaliação open-set do E01 (FaceNet) e E02 (ArcFace).

Uso (da raiz do projeto, com o .venv ativo):
  python -m src.eval.evaluate E02_arcface --split val
  python -m src.eval.evaluate E01_facenet --split val
  python -m src.eval.evaluate E02_arcface --split test   # só no fim, com threshold já fixado

Regras:
- val : protótipos (enroll) + probes known[val] + impostors_val.
        O threshold é escolhido AQUI (EER) e gravado em experiments/results/<exp>_val.json.
- test: protótipos (enroll) + known[test] + impostors_test.
        O threshold é LIDO do JSON de validação. Nunca é recalculado.
        Recusa rodar se o resultado de teste já existir (use --force só conscientemente).
- Probes sem face (NaN) são EXCLUÍDAS de todas as métricas e CONTADAS no resultado.
Depende de src/eval/metrics.py (prototipos, similaridades, escolher_threshold, calcular_metricas).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.eval import metrics as M

SPLIT = Path("data/splits/lfw_split.json")
EMB = Path("data/embeddings")
RES = Path("experiments/results")
DIM = 512


def load(name):
    ep, fp = EMB / f"{name}_embeddings.npy", EMB / f"{name}_files.txt"
    if not ep.exists() or not fp.exists():
        sys.exit(f"[ERRO] Faltam {ep} ou {fp}. Rode a extração antes.")
    emb = np.load(ep)
    files = fp.read_text(encoding="utf-8").split("\n")
    if len(files) != len(emb):
        sys.exit(f"[ERRO] {len(files)} nomes de arquivo, mas {len(emb)} embeddings.")
    return {f.replace("\\", "/"): emb[i] for i, f in enumerate(files)}


def get(by, lst, split, strict=True):
    """Matriz (len(lst), 512). Arquivo ausente na extração = erro (não vira NaN em silêncio)."""
    missing = [f for f in lst if f not in by]
    if missing and strict:
        sys.exit(f"[ERRO] {len(missing)} arquivos do split '{split}' não estão nos embeddings "
                 f"(ex.: {missing[0]}). Extraia o split '{split}' completo antes de avaliar.")
    return np.stack([by[f] for f in lst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="E01_facenet ou E02_arcface")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--force", action="store_true", help="permite regravar o resultado de teste")
    a = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    val_json = RES / f"{a.name}_val.json"
    out_json = RES / f"{a.name}_{a.split}.json"

    if a.split == "test":
        if not val_json.exists():
            sys.exit(f"[ERRO] {val_json} não existe. Rode --split val antes: o threshold vem de lá.")
        if out_json.exists() and not a.force:
            sys.exit(f"[ERRO] {out_json} já existe. O teste deve ser avaliado uma única vez.")

    by = load(a.name)
    s = json.loads(SPLIT.read_text(encoding="utf-8"))

    # --- protótipos (somente enroll) ---
    enroll = {i: get(by, d["enroll"], "enroll") for i, d in s["known"].items()}
    nomes, protos = M.prototipos(enroll)
    
    incompletas = {}
    for i, d in s["known"].items():
        arr = get(by, d["enroll"], "enroll", strict=False)
        nan_count = int(np.isnan(arr).any(axis=1).sum())
        if nan_count > 0:
            incompletas[i] = nan_count

    print(f"protótipos: {protos.shape} | ids com imagens de cadastro sem face: {incompletas}")

    # --- coletar probes do split escolhido (conhecidas + impostores) ---
    imp_key = "impostors_val" if a.split == "val" else "impostors_test"
    
    probes_list = []
    y_true_list = []

    for i, d in s["known"].items():
        if i not in nomes:
            continue
        for p in d[a.split]:
            probes_list.append(p)
            y_true_list.append(i)

    for p in s[imp_key]:
        probes_list.append(p)
        y_true_list.append(M.DESCONHECIDO)

    all_emb = get(by, probes_list, a.split)
    y_true_arr = np.array(y_true_list)

    # Filtrar NaNs nas probes
    ok_mask = ~np.isnan(all_emb).any(axis=1)
    conhecida_mask = y_true_arr != M.DESCONHECIDO
    nan_conhecidas = int((~ok_mask & conhecida_mask).sum())
    nan_impostores = int((~ok_mask & ~conhecida_mask).sum())

    print(f"[{a.split}] sem face -> conhecidas: {nan_conhecidas}/{conhecida_mask.sum()} | "
          f"impostores: {nan_impostores}/{(~conhecida_mask).sum()}")

    probes_validas = all_emb[ok_mask]
    y_true_validas = y_true_arr[ok_mask]

    if len(probes_validas) == 0:
        sys.exit("[ERRO] sem probes válidas para avaliar.")

    # --- similaridades ---
    sims = M.similaridades(probes_validas, protos, metrica="cosseno")

    # --- verificação de sanidade euclidiana ---
    sub = probes_validas[:500]
    sub_norm = M.normalizar(sub)
    protos_norm = M.normalizar(protos)
    cos_direct = sub_norm @ protos_norm.T
    euc_direct = np.linalg.norm(sub_norm[:, None, :] - protos_norm[None, :, :], axis=2)
    dev = float(np.abs(euc_direct ** 2 - (2 - 2 * cos_direct)).max())
    same_rank = bool((cos_direct.argmax(1) == euc_direct.argmin(1)).all())
    print(f"sanidade euclidiana: max|euc^2-(2-2cos)|={dev:.2e} | rankings iguais: {same_rank}")

    # --- threshold ---
    eer = None
    if a.split == "val":
        thr, eer = M.escolher_threshold(sims, y_true_validas, nomes)
        print(f"threshold (cosseno, EER na validação): {thr:.4f} | EER: {eer:.4f}")
    else:
        val_data = json.loads(val_json.read_text(encoding="utf-8"))
        thr = val_data["threshold"]
        print(f"threshold FIXADO da validação (cosseno): {thr:.4f}")
        _, eer = M.escolher_threshold(sims, y_true_validas, nomes)

    # --- calcular métricas ---
    res, cm, labels = M.calcular_metricas(sims, y_true_validas, nomes, thr)
    res.update(
        experiment=a.name,
        split=a.split,
        EER=eer,
        threshold_euclidean=float(np.sqrt(max(0.0, 2 - 2 * thr))),
        nan_known=nan_conhecidas,
        nan_impostor=nan_impostores,
        enroll_incomplete=incompletas,
        euclid_sanity_max_dev=dev,
        euclid_same_ranking=same_rank,
    )

    # Matriz de confusão resumida binária (conhecida/desconhecida)
    previstos, melhor, top1 = M.decidir(sims, nomes, thr)
    conhecida = y_true_validas != M.DESCONHECIDO
    tp = int(((previstos != M.DESCONHECIDO) & conhecida & (previstos == y_true_validas)).sum())
    wrong_id = int(((previstos != M.DESCONHECIDO) & conhecida & (previstos != y_true_validas)).sum())
    fr = int(((previstos == M.DESCONHECIDO) & conhecida).sum())
    fa = int(((previstos != M.DESCONHECIDO) & ~conhecida).sum())
    tn = int(((previstos == M.DESCONHECIDO) & ~conhecida).sum())
    
    res["confusion_known_vs_unknown"] = {
        "known_accepted_correct_id": tp,
        "known_accepted_wrong_id": wrong_id,
        "known_rejected": fr,
        "impostor_accepted": fa,
        "impostor_rejected": tn,
    }

    print(json.dumps(res, indent=2, ensure_ascii=False))
    
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("salvo em", out_json)
    if a.split == "val":
        print("Lembrete: estes são números de VALIDAÇÃO (threshold escolhido nestes mesmos dados).")


if __name__ == "__main__":
    main()