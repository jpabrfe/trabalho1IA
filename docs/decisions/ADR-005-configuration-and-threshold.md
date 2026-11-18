# ADR-005: Configuração externa e threshold

- Status: aceito
- Data: 18/11/2026

## Contexto
O threshold de 0,2679 foi calibrado na validação do LFW com o pipeline da Parte 1
(MTCNN, recorte 160x160, ArcFace buffalo_l, similaridade de cosseno), conforme o
ADR-003. Em câmera real pode ser necessário ajustá-lo, e o caminho dos dados
privados não pode estar no código nem no repositório.

## Decisão
- Parâmetros não sensíveis em `config.yaml`, versionado: `threshold` (0,2679),
  `min_confidence` (0,9), `min_face_px` (60) e a métrica (cosseno).
- Qualquer valor pode ser sobrescrito por variável de ambiente (por exemplo
  `FACE_THRESHOLD`), o que facilita testes e recalibração sem editar código.
- O caminho dos dados privados vem exclusivamente de `FACE_PRIVATE_DIR`.
- Na inicialização, a aplicação valida a configuração: se `FACE_PRIVATE_DIR`
  estiver ausente ou dentro do repositório, recusa iniciar com mensagem clara.
- Nenhum segredo e nenhum dado pessoal em arquivos versionados.

## Alternativas consideradas
- Constantes no código: simples, mas obrigam a editar o código para recalibrar.
- Banco de dados de configuração: exagero para poucos parâmetros.
- Controle deslizante na interface: arriscado, pois permite alterar a decisão
  sem registro; a interface apenas exibe o valor em uso.

## Consequências
- Recalibrar é editar `config.yaml` e registrar a justificativa; não exige novo código.
- O threshold só vale para o pipeline descrito acima; trocar detector, recorte ou
  modelo exige recalibrar com dados de validação.
- Os testes sobrescrevem a configuração com valores controlados.