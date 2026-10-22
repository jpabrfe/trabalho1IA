# E03: cosseno vs. distância euclidiana

## Hipótese (escrita antes de rodar)
Com embeddings já normalizados (norma 1), cosseno e distância euclidiana
produzem a mesma ordenação (d² = 2 − 2·cos), então as métricas devem ser
praticamente idênticas, com diferenças apenas numéricas.

## Configuração
- Modelo: o melhor entre E01 (FaceNet) e E02 (ArcFace), pela validação
- Dataset: LFW alinhado, split `data/splits/lfw_split.json` (seed 42)
- Cadastro: 5 imagens por conhecida (50 conhecidas); protótipo = média normalizada
- Métricas: accuracy, F1, FAR, FRR, EER, ROC-AUC, rank-1
- Threshold: escolhido na validação (ponto EER)

## Resultado
(preencher depois de rodar)

## Interpretação
(preencher depois de rodar)