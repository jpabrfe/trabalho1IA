# E05: fine-tuning completo do FaceNet com ArcFace loss

## Hipótese (escrita antes de rodar)
O fine-tuning completo reduz o EER na validação em relação ao parcial (E04),
mas com risco maior de sobreajuste às identidades de treino (1141), o que pode
piorar a generalização para identidades novas.

## Configuração
Igual ao E04, exceto: todas as camadas treináveis, lr 1e-5, loss ArcFace.

## Resultado (validação)
| Modelo                  | EER    | Accuracy | F1 macro | Rank-1 | ROC-AUC |
|-------------------------|--------|----------|----------|--------|---------|
| E01 FaceNet baseline    | 0,0402 | 0,960    | 0,887    | 0,969  | 0,984   |
| E04 fine-tuning parcial | 0,0432 | 0,957    | 0,867    | 0,969  | 0,979   |
| E05 fine-tuning completo| 0,0404 | 0,960    | 0,876    | 0,969  | 0,977   |

Threshold escolhido na validação: 0,4966 (cosseno).

## Interpretação
A hipótese foi parcialmente confirmada: o fine-tuning completo ficou melhor
que o parcial (EER 0,0404 contra 0,0432), mas não superou o baseline pré-treinado
(0,0402); a diferença de 0,0002 é desprezível frente à incerteza da validação.
ROC-AUC (0,977) e F1 macro (0,876) ficaram abaixo do baseline, o que é compatível
com sobreajuste às identidades de treino, hipótese não testada diretamente.
Conclusão: com 5.334 imagens de treino e este conjunto de hiperparâmetros,
o fine-tuning não trouxe ganho sobre o modelo pré-treinado.