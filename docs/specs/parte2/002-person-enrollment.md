# 002: Cadastro de pessoa

## Requisitos
- Cadastrar uma pessoa a partir de uma ou mais imagens, tratando casos de erro.

## Especificação
- `EnrollmentService.enroll(nome, imagens) -> EnrollResult(id, n_embeddings)`.
- Fluxo: validar nome e imagens, detectar, exigir exatamente uma face em cada imagem, extrair embeddings, checar duplicidade, gravar.
- Erros (com mensagem em português para a interface):
  - `InvalidImageError`: imagem ilegível ou face inadequada (spec 001).
  - `NoFaceError`: nenhuma face detectada.
  - `MultipleFacesError`: mais de uma face na imagem.
  - `DuplicatePersonError`: o nome já existe, ou o rosto já pertence a uma pessoa cadastrada (similaridade com o protótipo maior ou igual ao threshold).
- O cadastro é atômico: se qualquer imagem falhar, nada é gravado.
- Cada pessoa recebe um identificador interno (uuid); logs usam o identificador, nunca o nome.

## Critérios de aceitação
- Cada tipo de erro é produzido pelo seu caso de teste e não deixa dados gravados.
- Cadastrar a mesma pessoa duas vezes (mesmo nome ou mesmo rosto) falha.
- Um cadastro válido torna a pessoa visível em `listar()` e no reconhecimento.