# 005: Interface

## Requisitos
- Interface web local para cadastrar, reconhecer e gerenciar pessoas, com foto da câmera ou arquivo.

## Especificação
- Streamlit, com três abas: Cadastrar, Reconhecer e Pessoas (lista e exclusão).
- Na aba Reconhecer, mostrar a imagem com a caixa e o rótulo (nome ou "desconhecido") e a similaridade.
- Sem pessoas cadastradas, mostrar "Nenhuma pessoa cadastrada".
- Erros dos serviços aparecem como mensagens claras, sem traceback.
- Aviso de privacidade visível: os dados ficam só neste computador.
- A interface só chama os serviços; não contém lógica de detecção, embedding ou armazenamento.

## Critérios de aceitação
- Primeiro uso, com a pasta de dados vazia, mostra "Nenhuma pessoa cadastrada" e não falha.
- Cada erro da spec 002 é exibido com uma mensagem específica.
- Fluxo completo cadastrar, reconhecer e excluir feito pela interface, sem terminal.