# 003: Reconhecimento

## Requisitos
- Dada uma imagem (ou quadro de câmera), dizer, para cada face, quem é ou se é desconhecida.

## Especificação
- `RecognitionService.reconhecer(imagem) -> list[Reconhecimento(caixa, nome|None, similaridade)]`.
- Para cada face: embedding, similaridade de cosseno com o protótipo de cada pessoa (média normalizada dos embeddings do cadastro); vale a de maior valor.
- A decisão segue a spec 004.
- Sem pessoas cadastradas, toda face é desconhecida e o serviço sinaliza "nenhuma pessoa cadastrada".

## Critérios de aceitação
- Uma face cadastrada é reconhecida com o nome certo (teste com embeddings sintéticos).
- Uma face não cadastrada é devolvida como desconhecida.
- Imagem sem face devolve lista vazia.
- Duas faces na imagem produzem dois resultados independentes.
- A imagem enviada para reconhecer não é gravada em disco.