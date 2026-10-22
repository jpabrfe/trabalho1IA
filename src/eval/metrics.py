"""Métricas de identificação em conjunto aberto (1:N com rejeição).

Convenção: `sims` é uma matriz (N probes x K pessoas cadastradas); quanto maior,
mais parecido. y_true tem o nome da pessoa ou "desconhecido" (impostora).
O threshold é escolhido SOMENTE na validação; o teste só avalia.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

DESCONHECIDO = "desconhecido"


def normalizar(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def prototipos(embs_por_pessoa):
    """{nome: array (n, d)} -> (nomes, protótipos (K, d)); protótipo = média normalizada."""
    nomes = sorted(embs_por_pessoa)
    protos = np.stack(
        [normalizar(normalizar(embs_por_pessoa[n]).mean(axis=0)) for n in nomes]
    )
    return nomes, protos


def similaridades(probes, protos, metrica="cosseno"):
    """Cosseno ou (menos) distância euclidiana; maior = mais parecido."""
    probes = np.asarray(probes, dtype=np.float64)
    protos = np.asarray(protos, dtype=np.float64)
    if metrica == "cosseno":
        return normalizar(probes) @ normalizar(protos).T
    if metrica == "euclidiana":
        a2 = (probes ** 2).sum(1)[:, None]
        b2 = (protos ** 2).sum(1)[None, :]
        return -np.sqrt(np.maximum(a2 + b2 - 2 * probes @ protos.T, 0.0))
    raise ValueError(f"métrica desconhecida: {metrica}")


def decidir(sims, nomes, thr):
    """Retorna (previstos, melhor_score, top1). Abaixo do threshold = desconhecido."""
    nomes = np.asarray(nomes)
    top1 = nomes[sims.argmax(axis=1)]
    melhor = sims.max(axis=1)
    previstos = np.where(melhor >= thr, top1, DESCONHECIDO)
    return previstos, melhor, top1


def curva_far_frr(scores_conhecidas, scores_impostoras):
    """FAR e FRR para cada threshold candidato (os próprios scores)."""
    sk = np.sort(np.asarray(scores_conhecidas))
    si = np.sort(np.asarray(scores_impostoras))
    thrs = np.unique(np.concatenate([sk, si]))
    far = (len(si) - np.searchsorted(si, thrs, side="left")) / len(si)  # impostoras >= t
    frr = np.searchsorted(sk, thrs, side="left") / len(sk)              # conhecidas < t
    return thrs, far, frr


def escolher_threshold(sims_val, y_val, nomes):
    """Threshold no ponto EER, usando SÓ a validação. Retorna (threshold, eer)."""
    y_val = np.asarray(y_val)
    melhor = sims_val.max(axis=1)
    conhecida = y_val != DESCONHECIDO
    thrs, far, frr = curva_far_frr(melhor[conhecida], melhor[~conhecida])
    i = int(np.argmin(np.abs(far - frr)))
    return float(thrs[i]), float((far[i] + frr[i]) / 2)


def calcular_metricas(sims, y_true, nomes, thr):
    y_true = np.asarray(y_true)
    previstos, melhor, top1 = decidir(sims, nomes, thr)
    conhecida = y_true != DESCONHECIDO
    labels = list(nomes) + [DESCONHECIDO]

    res = {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, previstos)),
        "f1_macro": float(f1_score(y_true, previstos, labels=labels, average="macro", zero_division=0)),
        "f1_ponderado": float(f1_score(y_true, previstos, labels=labels, average="weighted", zero_division=0)),
        "far": float((previstos[~conhecida] != DESCONHECIDO).mean()),
        "frr": float((previstos[conhecida] == DESCONHECIDO).mean()),
        # conhecida aceita, mas confundida com outra conhecida
        "id_errada": float(((previstos != DESCONHECIDO) & conhecida & (previstos != y_true)).sum() / conhecida.sum()),
        "rank1": float((top1[conhecida] == y_true[conhecida]).mean()),  # sem threshold
        "roc_auc": float(roc_auc_score(conhecida, melhor)),
    }
    cm = confusion_matrix(y_true, previstos, labels=labels)
    return res, cm, labels


def plotar_distribuicoes(sims, y_true, nomes, thr, caminho, titulo=""):
    """Histogramas: conhecida-correta, conhecida-errada e desconhecida."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    top1 = np.asarray(nomes)[sims.argmax(axis=1)]
    melhor = sims.max(axis=1)
    conhecida = y_true != DESCONHECIDO
    grupos = [
        (conhecida & (top1 == y_true), "conhecida (id. correta)"),
        (conhecida & (top1 != y_true), "conhecida (id. errada)"),
        (~conhecida, "desconhecida"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for mask, rotulo in grupos:
        if mask.any():
            ax.hist(melhor[mask], bins=60, alpha=0.5, density=True, label=rotulo)
    ax.axvline(thr, color="k", ls="--", label="threshold")
    ax.set_xlabel("maior similaridade")
    ax.set_ylabel("densidade")
    ax.set_title(titulo)
    ax.legend()
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Teste com dados SINTÉTICOS (nenhuma foto): valida o protocolo val -> teste.
    rng = np.random.default_rng(42)
    d, K = 64, 5
    nomes_t = [f"pessoa_{i}" for i in range(K)]
    centros = normalizar(rng.normal(size=(K, d)))

    def amostras(n):
        probes, y = [], []
        for i in range(K):
            probes.append(centros[i] + 0.25 * rng.normal(size=(n, d)) / np.sqrt(d) * 3)
            y += [nomes_t[i]] * n
        imp = rng.normal(size=(n * K, d))
        probes.append(imp)
        y += [DESCONHECIDO] * len(imp)
        return np.vstack(probes), np.array(y)

    cad = {n: centros[i] + 0.25 * rng.normal(size=(5, d)) for i, n in enumerate(nomes_t)}
    nomes, protos = prototipos(cad)
    pv, yv = amostras(40)
    pt, yt = amostras(40)
    sv = similaridades(pv, protos)
    st = similaridades(pt, protos)
    thr, eer = escolher_threshold(sv, yv, nomes)   # validação
    res, cm, labels = calcular_metricas(st, yt, nomes, thr)  # teste (uma vez)
    print(f"threshold (val)={thr:.3f}  EER(val)={eer:.3f}")
    for k, v in res.items():
        print(f"  {k}: {v:.3f}")
    print("matriz de confusão:", cm.shape, "classes:", len(labels))