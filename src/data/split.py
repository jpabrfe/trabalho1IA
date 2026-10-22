"""Split por identidade do LFW: conhecidas, impostoras e treino de fine-tuning."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LFW = ROOT / "data" / "lfw" / "lfw_home" / "lfw_funneled"
OUT = ROOT / "data" / "splits" / "lfw_split.json"

SEED = 42
MIN_KNOWN = 15    # mínimo de imagens para uma identidade ser "conhecida"
K = 50            # nº de conhecidas (usa menos se não houver tantas)
N_ENROLL = 5      # imagens de cadastro por conhecida
FRAC_TRAIN, FRAC_VAL = 0.70, 0.15   # das demais identidades com 2+ imagens; o resto vai para teste


def listar():
    ids = {}
    for pasta in sorted(LFW.iterdir()):
        if pasta.is_dir():
            ids[pasta.name] = sorted(
                p.relative_to(LFW).as_posix() for p in pasta.glob("*.jpg")
            )
    return ids


def md5(rel):
    return hashlib.md5((LFW / rel).read_bytes()).hexdigest()


def main():
    rng = random.Random(SEED)
    ids = listar()

    # 1) identidades conhecidas: cadastro / validação / teste
    candidatas = sorted(n for n, f in ids.items() if len(f) >= MIN_KNOWN)
    rng.shuffle(candidatas)
    known = {}
    for nome in sorted(candidatas[:K]):
        fs = ids[nome][:]
        rng.shuffle(fs)
        resto = fs[N_ENROLL:]
        meio = len(resto) // 2
        known[nome] = {"enroll": fs[:N_ENROLL], "val": resto[:meio], "test": resto[meio:]}

    # 2) demais identidades: treino (fine-tuning) / impostoras val / impostoras teste
    demais = sorted(n for n in ids if n not in known)
    rng.shuffle(demais)
    multi = [n for n in demais if len(ids[n]) >= 2]
    single = [n for n in demais if len(ids[n]) < 2]
    n_tr = int(len(multi) * FRAC_TRAIN)
    n_va = int(len(multi) * FRAC_VAL)
    train_ids = multi[:n_tr]
    imp_val_ids = multi[n_tr:n_tr + n_va] + single[: len(single) // 2]
    imp_test_ids = multi[n_tr + n_va:] + single[len(single) // 2:]

    # 3) checagem: identidades disjuntas entre os grupos
    grupos_ids = [set(known), set(train_ids), set(imp_val_ids), set(imp_test_ids)]
    for i, a in enumerate(grupos_ids):
        for b in grupos_ids[i + 1:]:
            assert a.isdisjoint(b), "identidade repetida entre grupos"

    # 4) checagem: imagens duplicadas (mesmo hash) entre grupos.
    # Prioridade: cadastro > val > teste > impostoras > treino.
    hashes = {f: md5(f) for fs in ids.values() for f in fs}
    vistos = set()
    removidas = 0

    def filtra(fs):
        nonlocal removidas
        saida = []
        for f in fs:
            if hashes[f] in vistos:
                removidas += 1
            else:
                vistos.add(hashes[f])
                saida.append(f)
        return saida

    for parte in ("enroll", "val", "test"):
        for nome in known:
            known[nome][parte] = filtra(known[nome][parte])
    imp_val = filtra([f for n in imp_val_ids for f in ids[n]])
    imp_test = filtra([f for n in imp_test_ids for f in ids[n]])
    train = {n: filtra(ids[n]) for n in train_ids}
    train = {n: fs for n, fs in train.items() if len(fs) >= 2}

    # 5) salvar (só nomes de arquivos do LFW, sem imagens)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "seed": SEED,
                "params": {"min_known": MIN_KNOWN, "k": K, "n_enroll": N_ENROLL},
                "known": known,
                "impostors_val": imp_val,
                "impostors_test": imp_test,
                "train": train,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print(f"Conhecidas: {len(known)} identidades")
    print(f"  cadastro: {sum(len(v['enroll']) for v in known.values())} imgs")
    print(f"  probes val: {sum(len(v['val']) for v in known.values())} imgs")
    print(f"  probes teste: {sum(len(v['test']) for v in known.values())} imgs")
    print(f"Impostoras val: {len(imp_val_ids)} identidades, {len(imp_val)} imgs")
    print(f"Impostoras teste: {len(imp_test_ids)} identidades, {len(imp_test)} imgs")
    print(f"Treino fine-tuning: {len(train)} identidades, {sum(map(len, train.values()))} imgs")
    print(f"Duplicatas removidas entre grupos: {removidas}")
    print(f"Salvo em {OUT}")


if __name__ == "__main__":
    main()