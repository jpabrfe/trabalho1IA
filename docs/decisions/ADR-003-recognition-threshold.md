# ADR-003: Threshold de reconhecimento

- Status: rascunho
- Data: 22/10/2026

## Contexto
O sistema precisa rejeitar pessoas fora do conjunto cadastrado (identificação em conjunto aberto). Um threshold alto rejeita conhecidas (FRR); um baixo aceita desconhecidas (FAR).

## Decisão
Threshold global, escolhido na validação no ponto EER (FAR = FRR), com impostoras do LFW. Valor escolhido: (preencher após os experimentos).

## Alternativas consideradas
- Threshold por pessoa: mais flexível, mas exige mais dados por pessoa; fora do escopo.
- Threshold por FAR alvo (por exemplo, 1%): útil em aplicações de segurança; pode ser reportado como análise extra.

## Consequências
- O teste nunca influencia a escolha.
- Se a distribuição das fotos privadas for diferente da do LFW, o threshold pode não generalizar; discutir na análise.