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

## Resultado
(preencher depois de rodar)

## Interpretação
(preencher depois de rodar)