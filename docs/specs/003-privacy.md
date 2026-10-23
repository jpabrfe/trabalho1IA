# 003: Privacidade dos dados do projeto

## Requisitos
- Fotos, embeddings, termos de consentimento e checkpoints derivados de dados privados nunca vão ao repositório.
- Cada participante assina o próprio termo, e cada um guarda as próprias fotos no próprio computador.

## Especificação
- Dados privados ficam em `FACE_PRIVATE_DIR`, fora do projeto e fora de Drive/OneDrive.
- `.gitignore` bloqueia os diretórios e as extensões sensíveis; o histórico do Git é auditado antes da entrega.
- Treino e fine-tuning usam só o LFW. As fotos privadas servem apenas para avaliação, localmente.
- Impostores da avaliação privada vêm do LFW. Resultados privados são reportados só de forma agregada.

## Critérios de aceitação
- Checklist do Anexo B concluído.
- Nenhuma foto, embedding, termo ou checkpoint privado no histórico do Git.
- Nenhum dado privado enviado a serviços externos (inclusive agentes de IA e Colab).