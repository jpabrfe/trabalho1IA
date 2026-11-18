# 004: Pessoa desconhecida e threshold

## Requisitos
- O sistema não deve classificar toda face como alguém cadastrado.

## Especificação
- Reconhecida se a maior similaridade for maior ou igual ao threshold; senão, desconhecida.
- Threshold padrão 0,2679 (cosseno), escolhido na validação do LFW no ponto EER (Parte 1, ADR-003). Fica em arquivo de configuração, nunca no código.
- Limitação registrada: foi calibrado com fotos do LFW; em câmera real pode exigir recalibração, e isso é aviso na documentação.

## Critérios de aceitação
- Similaridade igual ao threshold é aceita; imediatamente abaixo, rejeitada.
- Alterar o threshold na configuração muda a decisão sem alterar código.
- A pessoa de maior similaridade abaixo do threshold nunca é devolvida como reconhecida.