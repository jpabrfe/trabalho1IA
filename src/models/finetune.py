"""Fine-tuning do FaceNet (InceptionResnetV1, pesos VGGFace2) com metric learning.

Dados: SÓ o grupo `train` do LFW (público). Nunca use fotos privadas aqui.
Uso:
    python finetune.py --dados DIR --estrategia parcial --loss arcface --epocas 15
DIR contém `lfw_split.json` (com a chave "train") e a pasta `aligned/`.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler

LIVRES_PARCIAL = ("repeat_3", "block8", "last_linear", "last_bn")


class FacesTrain(Dataset):
    def __init__(self, raiz, train, aug=True):
        self.raiz = Path(raiz) / "aligned"
        self.nomes = sorted(train)
        self.itens = [(f, i) for i, n in enumerate(self.nomes) for f in train[n]]
        self.por_id = {i: [] for i in range(len(self.nomes))}
        for k, (_, i) in enumerate(self.itens):
            self.por_id[i].append(k)
        self.aug = aug

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, k):
        f, y = self.itens[k]
        img = Image.open(self.raiz / f).convert("RGB")
        if self.aug and random.random() < 0.5:
            img = ImageOps.mirror(img)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()
        if self.aug:
            x = (x * random.uniform(0.8, 1.2)).clamp(0, 255)
        return (x - 127.5) / 128.0, y  # mesma padronização do embedder


class PKSampler(Sampler):
    """P identidades x K imagens por lote (necessário p/ triplet e útil p/ arcface)."""

    def __init__(self, por_id, P, K, seed):
        self.ids = [i for i, v in por_id.items() if len(v) >= 2]
        self.por_id, self.P, self.K = por_id, P, K
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ids) // self.P

    def __iter__(self):
        ids = self.ids[:]
        self.rng.shuffle(ids)
        for b in range(len(self)):
            lote = []
            for i in ids[b * self.P:(b + 1) * self.P]:
                pool = self.por_id[i]
                lote += (self.rng.sample(pool, self.K) if len(pool) >= self.K
                         else self.rng.choices(pool, k=self.K))
            yield lote


class ArcFaceHead(nn.Module):
    def __init__(self, d, n_classes, s=30.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, d))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m, self.n = s, m, n_classes

    def forward(self, emb, y):
        cos = (F.normalize(emb) @ F.normalize(self.W).T).clamp(-1 + 1e-7, 1 - 1e-7)
        alvo = torch.cos(torch.acos(cos) + self.m)
        onehot = F.one_hot(y, self.n).bool()
        return F.cross_entropy(self.s * torch.where(onehot, alvo, cos), y)


def triplet_batch_hard(emb, y, margem):
    emb = F.normalize(emb)
    d = (2 - 2 * emb @ emb.T).clamp(min=1e-8).sqrt()
    igual = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = (d * (igual & ~eye)).max(dim=1).values
    neg = d.masked_fill(igual, float("inf")).min(dim=1).values
    return F.relu(pos - neg + margem).mean()


def congelar(modelo, estrategia):
    for n, p in modelo.named_parameters():
        p.requires_grad = (estrategia == "completo") or n.startswith(LIVRES_PARCIAL)


def modo_treino(modelo, estrategia):
    modelo.train()
    if estrategia == "parcial":  # camadas congeladas ficam em eval (BatchNorm estável)
        for nome, m in modelo.named_children():
            if nome not in LIVRES_PARCIAL + ("dropout", "avgpool_1a"):
                m.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dados", required=True)
    p.add_argument("--estrategia", choices=["parcial", "completo"], default="parcial")
    p.add_argument("--loss", choices=["arcface", "triplet"], default="arcface")
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)         # use 1e-5 no "completo"
    p.add_argument("--lr_cabeca", type=float, default=1e-3)
    p.add_argument("--margem", type=float, default=None)     # arcface 0.3; triplet 0.2
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saida", default=None)
    a = p.parse_args()
    margem = a.margem if a.margem is not None else (0.3 if a.loss == "arcface" else 0.2)
    saida = Path(a.saida or f"ckpt_{a.estrategia}_{a.loss}.pt")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = json.loads((Path(a.dados) / "lfw_split.json").read_text(encoding="utf-8"))["train"]
    ds = FacesTrain(a.dados, train)
    dl = DataLoader(ds, batch_sampler=PKSampler(ds.por_id, a.P, a.K, a.seed),
                    num_workers=2, pin_memory=(dev == "cuda"))
    print(f"{len(ds.nomes)} identidades, {len(ds)} imagens, dispositivo: {dev}")

    modelo = InceptionResnetV1(pretrained="vggface2").to(dev)
    congelar(modelo, a.estrategia)
    cabeca = ArcFaceHead(512, len(ds.nomes), m=margem).to(dev) if a.loss == "arcface" else None
    grupos = [{"params": [q for n, q in modelo.named_parameters()
                          if q.requires_grad and not n.startswith("logits")], "lr": a.lr}]
    if cabeca is not None:
        grupos.append({"params": cabeca.parameters(), "lr": a.lr_cabeca})
    opt = torch.optim.AdamW(grupos, weight_decay=1e-4)

    historico = []
    for ep in range(1, a.epocas + 1):
        modo_treino(modelo, a.estrategia)
        t0, soma, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            emb = modelo(x)
            loss = cabeca(emb, y) if cabeca is not None else triplet_batch_hard(emb, y, margem)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            soma += loss.item(); n += 1
        historico.append(soma / n)
        print(f"época {ep:02d}/{a.epocas}  loss={soma / n:.4f}  ({time.time() - t0:.0f}s)")

    torch.save({"modelo": modelo.state_dict(), "config": vars(a), "margem": margem,
                "torch": torch.__version__}, saida)
    saida.with_suffix(".json").write_text(
        json.dumps({"config": vars(a), "margem": margem, "loss_por_epoca": historico}, indent=2),
        encoding="utf-8")
    print(f"Salvo: {saida}")


if __name__ == "__main__":
    main()