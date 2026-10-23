# ADR-002: Armazenamento de embeddings

- Status: aceito
- Data: 22/10/2026

## Contexto
Embeddings faciais são dados biométricos privados (enunciado, seção 20). Os embeddings do LFW são públicos, mas os das fotos dos participantes não podem ir ao repositório.

## Decisão
- Embeddings do LFW: arquivos `.npy` em `data/embeddings/` (ignorada pelo Git), com um arquivo de nomes na mesma ordem. Regeneráveis pelos scripts.
- Embeddings das fotos privadas: apenas em `FACE_PRIVATE_DIR/embeddings`, no computador de cada participante, nunca no projeto.
- Protótipos (média normalizada) são recalculados em memória a cada avaliação; nada é persistido além dos `.npy`.

## Alternativas consideradas
- Banco vetorial: desnecessário para 50 pessoas conhecidas; fica para a Parte 2, se for preciso.
- Publicar embeddings sem as fotos: proibido pelo enunciado.

## Consequências
- Excluir os dados de uma pessoa é apagar os arquivos dela em `face_private`.
- Os resultados privados são reportados só de forma agregada.