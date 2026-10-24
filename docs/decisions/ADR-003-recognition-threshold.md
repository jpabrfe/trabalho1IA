# ADR-003: Threshold de reconhecimento

- Status: aceito
- Data: 24/10/2026

## Contexto
O sistema precisa rejeitar pessoas fora do conjunto cadastrado (identificação em
conjunto aberto). Um threshold alto rejeita pessoas cadastradas (FRR); um baixo
aceita desconhecidas (FAR).

## Decisão
Threshold global de **0,2679** sobre a similaridade de cosseno, para o modelo
final (ArcFace pré-treinado, buffalo_l). Foi escolhido na validação do LFW, no
ponto em que FAR e FRR se igualam (EER = 0,0365), com impostoras do LFW, e
fixado antes de olhar o teste. Com distância euclidiana, o valor equivalente é
cerca de 1,21 (d = sqrt(2 - 2 cos)), sem diferença de desempenho (E03).

## Resultados com o threshold fixado
| Conjunto | FAR | FRR | Accuracy | F1 macro |
|----------|-----|-----|----------|----------|
| Validação (LFW) | 0,0365 | 0,0364 | 0,964 | 0,871 |
| Teste (LFW, uma única vez) | 0,0301 | 0,0301 | 0,970 | 0,905 |

Na avaliação com fotos privadas (agregada), nenhuma impostora do LFW foi aceita
e nenhuma foto foi confundida com outra pessoa; a taxa de rejeição indevida foi
maior que a do LFW (16% em 25 fotos de teste de um participante).

## Alternativas consideradas
- Threshold por pessoa: mais flexível, mas exige mais dados por