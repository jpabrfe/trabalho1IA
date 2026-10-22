# Reconhecimento Facial - Parte 1

Trabalho de Inteligência Artificial (UFMS): identificação facial em conjunto aberto.

## Estrutura
- `src/`: código (dados, modelos, avaliação)
- `configs/` e `experiments/`: configurações e registros de experimentos
- `docs/specs/` e `docs/decisions/`: especificações e ADRs
- `data/README.md`: fonte e licença dos datasets

## Privacidade
Fotos, embeddings, termos de consentimento e checkpoints privados ficam fora do repositório (`FACE_PRIVATE_DIR`).

## Instalação

Python 3.13 em ambiente virtual, na raiz do projeto:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate       # Linux/macOS

# 1) dependências (versões fixadas em requirements.txt)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2) facenet-pytorch, instalado à parte e sem checar dependências
pip install facenet-pytorch==2.6.0 --no-deps
```

O `facenet-pytorch 2.6.0` declara `numpy<2.0`, mas funciona com o numpy 2.x do
`requirements.txt`. Por isso ele não está no `requirements.txt` e é instalado
com `--no-deps`.

Teste de fumaça (usa imagem de ruído, sem fotos):

```bash
python -m src.models.embedder
```