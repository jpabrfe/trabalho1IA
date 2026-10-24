# E04: fine-tuning parcial do FaceNet com ArcFace loss

## Hipótese (escrita antes de rodar)
O fine-tuning parcial (últimos blocos descongelados) do FaceNet com ArcFace
loss, treinado só nas identidades do grupo `train` do LFW, reduz o EER na
validação em relação ao baseline pré-treinado (E01, EER 0,0402).

## Configuração
- Modelo: InceptionResnetV1 (VGGFace2); descongelados: repeat_3, block8, last_linear, last_bn
- Loss: ArcFace (s=30, m=0,3); lotes de 16 identidades x 4 imagens
- Dados: só LFW, grupo `train` do split (seed 42), sem fotos privadas
- Otimizador: AdamW, lr 1e-4 (cabeça 1e-3), 15 épocas
- Avaliação: mesma do E01 (validação; teste só no modelo final)

## Resultado (validação)
| Modelo                  | EER    | Accuracy | F1 macro | Rank-1 | ROC-AUC |
|-------------------------|--------|----------|----------|--------|---------|
| E01 FaceNet baseline    | 0,0402 | 0,960    | 0,887    | 0,969  | 0,984   |
| E04 fine-tuning parcial | 0,0432 | 0,957    | 0,867    | 0,969  | 0,979   |

Threshold escolhido na validação: 0,492 (cosseno).

## Interpretação
A hipótese não foi confirmada: o fine-tuning parcial não reduziu o EER; ele
ficou 0,3 ponto percentual acima do baseline. Essa diferença está dentro da
incerteza da validação (741 probes conhecidas), então a conclusão correta é que
não houve melhora detectável, e não que o fine-tuning piorou. O rank-1 não mudou.
Possíveis causas, não testadas: os pesos pré-treinados já foram treinados em
uma base muito maior que as 5.334 imagens de treino usadas aqui; a cabeça
ArcFace, inicializada do zero com 1.141 classes, pode ter empurrado as camadas
liberadas para separar essas identidades específicas; e epochs e learning rate
não foram ajustados.