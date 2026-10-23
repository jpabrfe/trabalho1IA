# 002: Pessoa desconhecida e threshold

## Requisitos
- O sistema decide entre "pessoa conhecida X" e "desconhecida"; não classifica toda face como alguém cadastrado.

## Especificação
- Para cada probe: calcular a similaridade com o protótipo de cada pessoa cadastrada (média normalizada das imagens de cadastro).
- Se a maior similaridade for maior ou igual ao threshold, a identidade é a da maior; senão, "desconhecida".
- O threshold é escolhido na validação, no ponto em que FAR e FRR se igualam (EER), e fixado antes de olhar o teste.
- Métricas reportadas: accuracy, F1, FAR, FRR, EER, ROC-AUC, rank-1 e identificação errada (conhecida confundida com outra conhecida).

## Critérios de aceitação
- `escolher_threshold` recebe só dados de validação.
- `calcular_metricas` é chamada no teste uma única vez, para o modelo já escolhido.
- A matriz de confusão tem uma classe extra "desconhecida".