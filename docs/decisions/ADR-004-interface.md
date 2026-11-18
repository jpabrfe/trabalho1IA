# ADR-004: Interface da aplicação

- Status: aceito
- Data: 18/11/2026

## Contexto
A Parte 2 exige uma interface (web, desktop, mobile ou câmera) para cadastrar,
reconhecer e excluir pessoas. O prazo é curto e a equipe tem experiência em
Python, não em desenvolvimento web.

## Decisão
Interface web local em Streamlit, com envio de arquivo e foto pela câmera do
navegador (`st.camera_input`), organizada em três abas: Cadastrar, Reconhecer e
Pessoas. A interface só chama os serviços (specs 002, 003 e 006) e não contém
lógica de detecção, embedding ou armazenamento.

## Alternativas consideradas
- Janela desktop com OpenCV (vídeo em tempo real): dá vídeo contínuo, mas exige
  mais código de interface e é mais difícil de testar e de gravar na demonstração.
- Flask ou FastAPI com página HTML: mais controle, mas muito mais trabalho para
  o mesmo resultado funcional.
- Gradio: equivalente ao Streamlit; escolhido o Streamlit pela documentação e
  pelo controle de estado entre abas.

## Consequências
- O reconhecimento é feito por foto (quadro capturado), e não por vídeo
  contínuo. Isso atende ao requisito de receber uma imagem; vídeo em tempo real
  fica como melhoria futura.
- Tudo roda em localhost: nenhuma imagem sai do computador do usuário.
- A câmera depende da permissão do navegador.
- Trocar de interface no futuro exige mexer apenas na camada de interface.