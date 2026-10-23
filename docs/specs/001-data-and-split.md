# 001: Dados e split

## Requisitos
- Usar o LFW (funneled) como dataset público; não versionar as imagens.
- Separar identidades em conhecidas, impostoras (validação e teste) e treino de fine-tuning.
- Impedir vazamento de dados (data leakage) entre os grupos.

## Especificação
- `src/data/download_lfw.py` baixa o dataset; `src/data/split.py` gera `data/splits/lfw_split.json` (seed 42).
- Conhecidas: até 50 identidades com 15 ou mais imagens; 5 imagens de cadastro, o resto dividido em probes de validação e de teste.
- Impostoras e treino: identidades diferentes das conhecidas.

## Critérios de aceitação
- Os conjuntos de identidades são disjuntos (verificado por `assert` no script).
- Nenhuma imagem com o mesmo hash aparece em dois grupos.
- Rodar o script de novo com a mesma seed gera o mesmo JSON.
- O teste não é usado para treino, escolha de modelo nem de threshold.