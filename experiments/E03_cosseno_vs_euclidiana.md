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

## Desvio em relação ao plano
A hipótese previa rodar o E03 só no melhor modelo. Como a checagem é barata
(mesmos embeddings, só muda a métrica), ela foi feita nos dois modelos (E01 e E02).

## Resultado (validação)
| Modelo  | Threshold (cosseno) | Threshold (euclidiana) | Mesma ordenação? |
|---------|---------------------|------------------------|------------------|
| FaceNet | 0,5887              | 0,9070                 | sim              |
| ArcFace | 0,3984              | 1,0969                 | sim              |

- A checagem de sanidade do script (`euclid_sanity_max_dev`) deu desvio máximo
  de 4,0e-12 nos dois modelos, ou seja, apenas ruído numérico.
- Os thresholds euclidianos correspondem aos do cosseno pela relação
  d = sqrt(2 - 2*cos): sqrt(2 - 2*0,5887) = 0,907 e sqrt(2 - 2*0,3984) = 1,097.

## Interpretação
A hipótese foi confirmada. Com embeddings normalizados (norma 1), cosseno e
distância euclidiana são equivalentes: produzem a mesma ordenação das pessoas
cadastradas e o mesmo ponto de operação (FAR, FRR e EER idênticos), com o
threshold apenas reescrito em outra escala. A escolha entre as duas métricas
não altera o desempenho. Foi adotado o cosseno por ser a convenção da
literatura. Para a métrica fazer diferença seria preciso usar embeddings sem
normalização, o que não foi investigado. Esta equivalência vale para os
embeddings pré-treinados normalizados; no fine-tuning, se os embeddings não
forem normalizados, a conclusão deve ser reavaliada.