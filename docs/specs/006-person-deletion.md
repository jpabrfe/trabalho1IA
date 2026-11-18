# 006: Exclusão de pessoa (Privacy by Design)

## Requisitos
- Remover uma pessoa e todos os dados associados a ela.

## Especificação
- `PersonRepository.excluir(id) -> bool` apaga, de uma vez, os metadados, os embeddings e quaisquer fotos guardadas dessa pessoa.
- Cada pessoa vive em uma pasta própria dentro de `FACE_PRIVATE_DIR`, de modo que excluir é remover a pasta.
- Registrar em log o evento com o identificador, sem nome.

## Critérios de aceitação
- Depois de excluir, a pessoa some da listagem e não é mais reconhecida.
- Nenhum arquivo dela permanece em disco (verificado no teste).
- Excluir um identificador inexistente devolve falso, sem erro.
- As demais pessoas permanecem intactas.