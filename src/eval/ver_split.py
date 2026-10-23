"""Inspeciona a estrutura exata do lfw_split.json"""
import json
from pathlib import Path

SPLIT = Path("data/splits/lfw_split.json")

def main():
    if not SPLIT.exists():
        print(f"[ERRO] {SPLIT} não encontrado.")
        return
    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    print("Chaves principais do split:", list(s.keys()))
    
    for k in s.keys():
        val = s[k]
        print(f"\n--- Chave: '{k}' (tipo: {type(val)}) ---")
        if isinstance(val, dict):
            print(f"Total de identidades/itens: {len(val)}")
            sample_key = next(iter(val)) if val else None
            if sample_key:
                print(f"Exemplo de chave '{sample_key}': {type(val[sample_key])}")
                print(f"Conteúdo de exemplo: {val[sample_key]}")
        elif isinstance(val, list):
            print(f"Tamanho da lista: {len(val)}")
            if val:
                print(f"Exemplos (primeiros 3): {val[:3]}")

if __name__ == "__main__":
    main()