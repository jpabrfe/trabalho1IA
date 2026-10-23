"""Empacota SÓ as imagens do grupo `train` (LFW público) + split, para o Colab."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "splits" / "lfw_split.json"
ALIGNED = ROOT / "data" / "lfw" / "aligned"
OUT = ROOT / "data" / "lfw" / "colab_train.zip"   # pasta ignorada pelo Git


def main():
    train = json.loads(SPLIT.read_text(encoding="utf-8"))["train"]
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        z.writestr("lfw_split.json", json.dumps({"train": train}))
        for fs in train.values():
            for f in fs:
                z.write(ALIGNED / f, f"aligned/{f}")
                n += 1
    print(f"{n} imagens em {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()