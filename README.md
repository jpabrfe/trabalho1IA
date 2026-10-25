# Reconhecimento Facial - Parte 1

Trabalho de Inteligência Artificial (UFMS): identificação facial em conjunto aberto.

## Modelo escolhido

**ArcFace pré-treinado (InsightFace buffalo_l), sem fine-tuning, com cosseno e
threshold 0,2679.** Foi o de menor EER na validação (0,0365), entre os
modelos comparados com a mesma entrada (faces recortadas pelo MTCNN). No teste do
LFW, avaliado uma única vez, obteve EER 0,030, accuracy 0,970 e F1 macro 0,905.
O fine-tuning do FaceNet (parcial e completo) não superou o modelo pré-treinado.
Detalhes e ressalvas em [ADR-001](docs/decisions/ADR-001-face-model.md).

## Experimentos

Métricas na validação (LFW: 50 conhecidas e impostoras disjuntas). F1 é macro.

| Exp. | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|------|--------|-----------|---------|----------|----|-----------|
| E01 | FaceNet (VGGFace2) | pré-treinado | LFW | 0,960 | 0,887 | EER 0,0402 |
| E02 | ArcFace (buffalo_l) | pré-treinado | LFW | 0,964 | 0,871 | EER 0,0365; **modelo final** |
| E02a | ArcFace | pré-treinado, imagem original | LFW | 0,951 | 0,859 | EER 0,0486; sem recorte do MTCNN |
| E03 | E01 e E02 | cosseno vs. euclidiana | LFW | idêntica | idêntica | mesma ordenação (embeddings normalizados) |
| E04 | FaceNet | fine-tuning parcial, ArcFace loss | LFW | 0,957 | 0,867 | EER 0,0432 |
| E05 | FaceNet | fine-tuning completo, ArcFace loss | LFW | 0,960 | 0,876 | EER 0,0404 |

Avaliação final (teste do LFW, uma única vez, E02): accuracy 0,970, F1 0,905,
FAR = FRR = 0,030.
Avaliação com fotos privadas (agregada, média dos participantes): Avaliação com fotos privadas (agregada, 2 participantes, 45 fotos de teste): 88,9% reconhecidas (40 de 45), 11,1% rejeitadas como desconhecidas, 0% confundidas com outra pessoa e 0% de impostoras do LFW aceitas como participante.

Análise completa em [docs/ANALISE.md](docs/ANALISE.md).

## Datasets

**LFW (Labeled Faces in the Wild), versão funneled, 13.233 imagens.**
- Fonte oficial: `vis-www.cs.umass.edu/lfw/` (página inacessível em 21/10/2026).
  Artigo: Huang, Ramesh, Berg e Learned-Miller, 2007.
- Obtenção: espelho usado pelo scikit-learn (`fetch_lfw_people`), por
  `python -m src.data.download_lfw`. O dataset não é versionado.
- Licença/termos: não localizados, pois a página oficial estava fora do ar;
  a verificar antes da entrega. Detalhes em [data/README.md](data/README.md).
- Finalidade e uso: pesquisa em reconhecimento facial. Aqui serve para treino do
  fine-tuning, validação, teste e como impostoras (desconhecidas), em grupos de
  identidades disjuntos (seed 42).

**Dataset privado:** fotos dos próprios participantes, com termo de
consentimento assinado, usadas só para avaliação local e reportadas de forma
agregada. Não faz parte do repositório.

## Estrutura
- `src/`: código (dados, modelos, avaliação)
- `configs/` e `experiments/`: configurações e registros de experimentos
- `docs/specs/` e `docs/decisions/`: especificações e ADRs
- `data/README.md`: fonte e licença dos datasets

## Privacidade
Fotos, embeddings, termos de consentimento e checkpoints privados ficam fora do repositório (`FACE_PRIVATE_DIR`).

Os dados privados (fotos, embeddings, termos de consentimento) ficam só no
computador de cada participante, fora do projeto, e são eliminados ao fim da
disciplina. Os checkpoints do fine-tuning foram treinados apenas com o LFW e
não são versionados. Resultados com fotos privadas são reportados só de forma
agregada.

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

## Reprodução

Na raiz do projeto, com o ambiente ativo:

```bash
python -m src.data.download_lfw        # baixa o LFW
python -m src.data.split               # gera data/splits/lfw_split.json
python -m src.data.align               # recorta as faces com MTCNN
python -m src.eval.run_baseline --modelo facenet     # E01 (validação)
python -m src.eval.run_baseline --modelo arcface     # E02 (validação)
python -m src.eval.run_baseline --modelo arcface --metrica euclidiana   # E03
```

Fine-tuning (E04 e E05), no Colab com GPU, só com dados do LFW:

```bash
python -m src.data.make_train_zip      # empacota o grupo de treino (local)
python finetune.py --dados dados --estrategia parcial --loss arcface --epocas 15
python finetune.py --dados dados --estrategia completo --loss arcface --lr 1e-5 --epocas 15
python -m src.eval.run_baseline --modelo facenet --ckpt checkpoints/<arquivo>.pt
```

Avaliação final no teste do LFW, **uma única vez**, no modelo escolhido:

```bash
python -m src.eval.run_baseline --modelo arcface --teste
```

Avaliação privada (local, cada participante no seu computador; requer
`FACE_PRIVATE_DIR` com fotos em `photos/cadastro` e `photos/teste`):

```bash
python -m src.data.align_privado
python -m src.eval.evaluate_privado --modelo arcface --final
```

Seeds, versões e hardware de cada execução ficam em `experiments/runs/`.