# Reconhecimento Facial (Inteligência Artificial, UFMS)

Sistema de reconhecimento facial em conjunto aberto: dada uma face, decide se a
pessoa é uma das cadastradas ou se é desconhecida.

- **Parte 1 (experimentação):** tag [`parte1-entrega`](../../tree/parte1-entrega) e a seção "Parte 1" abaixo. Vídeo: //drive.google.com/file/d/1wuz9tYekn-P8dkUVGbBB5BDIY9Cgrut_/view?usp=sharing
- **Parte 2 (aplicação):** este branch (`main`). Vídeo: <link do vídeo da Parte 2>

## Parte 2: aplicação

Aplicação web local, feita em Streamlit, com três funções:

- **Cadastrar** uma pessoa a partir de uma ou mais fotos (arquivo ou câmera).
- **Reconhecer** as faces de uma imagem: nome da pessoa ou "desconhecido".
- **Excluir** uma pessoa e todos os dados dela.

O pipeline é o escolhido na Parte 1: MTCNN recorta a face (160x160), o ArcFace
(InsightFace buffalo_l) gera um embedding de 512 números e a decisão é por
similaridade de cosseno com o protótipo de cada pessoa (média normalizada dos
embeddings do cadastro), com threshold 0,2679.

No primeiro uso, a aplicação mostra "Nenhuma pessoa cadastrada": o repositório
não contém nenhum dado de pessoas.

### Instalação

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
`requirements.txt`; por isso ele não está no `requirements.txt` e é instalado
com `--no-deps`. Na primeira execução, os pesos do FaceNet e do ArcFace são
baixados automaticamente (cerca de 400 MB) e não são versionados.

### Execução

Os dados das pessoas ficam numa pasta **fora do repositório**, indicada pela
variável de ambiente `FACE_PRIVATE_DIR`. A aplicação recusa iniciar se ela não
estiver definida ou se apontar para dentro do repositório.

```powershell
$env:FACE_PRIVATE_DIR = "C:\caminho\para\pasta_privada"    # Windows (PowerShell)
python -m streamlit run src/face_app/ui/app.py
```

```bash
export FACE_PRIVATE_DIR="$HOME/pasta_privada"               # Linux/macOS
python -m streamlit run src/face_app/ui/app.py
```

Execute sempre da raiz do projeto. A aplicação abre no navegador, em
`localhost`; nenhuma imagem sai do computador. A primeira abertura demora, porque
os modelos são carregados.

### Como usar
1. **Cadastrar:** informe o nome, confirme que a pessoa assinou o termo de
   consentimento e envie uma ou mais fotos (arquivo e/ou câmera) com **uma única
   face** cada. Várias fotos da mesma pessoa, com luz e ângulo variados, melhoram
   o resultado. O cadastro é atômico: se qualquer foto falhar, nada é gravado.
2. **Reconhecer:** envie uma imagem ou tire uma foto (se houver foto da câmera e
   arquivo, vale a da câmera); a aplicação desenha uma caixa em cada face, com o
   nome (ou "desconhecido") e a similaridade. A imagem enviada para reconhecer
   não é gravada.
3. **Pessoas:** lista as pessoas cadastradas e permite excluí-las, com
   confirmação que mostra o nome da pessoa.

Imagens com o lado maior acima de 1280 pixels são reduzidas antes do
processamento. Mensagens de erro (sem rosto, várias faces, imagem inadequada ou
ilegível, nome inválido, pessoa duplicada) aparecem na tela em texto claro, sem
traceback.

### Configuração

`config.yaml` (versionado, sem dados pessoais). Cada valor pode ser sobrescrito
por variável de ambiente.

| Parâmetro | Variável | Padrão | Descrição |
|---|---|---|---|
| `threshold` | `FACE_THRESHOLD` | 0,2679 | Similaridade mínima para reconhecer |
| `metric` | `FACE_METRIC` | cosseno | Única métrica suportada |
| `min_confidence` | `FACE_MIN_CONFIDENCE` | 0,9 | Confiança mínima do detector |
| `min_face_px` | `FACE_MIN_FACE_PX` | 60 | Lado mínimo da face, em pixels |
| `log_level` | `FACE_LOG_LEVEL` | INFO | Nível dos logs |
| (só ambiente) | `FACE_PRIVATE_DIR` | (obrigatória) | Pasta dos dados privados |

O threshold foi calibrado com o LFW (Parte 1, [ADR-003](docs/decisions/ADR-003-recognition-threshold.md))
e só vale para este pipeline. Se detector, recorte ou modelo mudarem, é preciso
recalibrar. Detalhes em [ADR-005](docs/decisions/ADR-005-configuration-and-threshold.md).
A equivalência do pipeline da aplicação com o da Parte 1 foi verificada em 50
imagens do LFW (similaridade mediana de 0,9945 com os embeddings da Parte 1).

### Arquitetura

```
Interface (Streamlit)  ──►  Serviços  ──►  Núcleo (detector MTCNN, extrator ArcFace)
                            cadastro       Repositório de pessoas (disco)
                            reconhecimento
                            pessoas
```

- `src/face_app/ui/`: interface. Só chama os serviços; não tem lógica de detecção,
  embedding nem armazenamento.
- `src/face_app/services/`: cadastro, reconhecimento e gerenciamento de pessoas.
  Validam as entradas, aplicam o threshold e traduzem falhas em erros claros.
- `src/face_app/core/`: detector de faces e extrator de embeddings.
- `src/face_app/storage/`: repositório de pessoas em disco.
- `src/face_app/contracts.py`: tipos e interfaces compartilhados entre as camadas.
- `src/face_app/errors.py`, `config.py`, `logging_setup.py`: erros, configuração
  externa e logs.

As camadas dependem de interfaces (`FaceDetector`, `FaceEmbedder`), então os
testes usam um detector e um extrator falsos, sem modelos e sem fotos reais.

### Dados e privacidade

- Cada pessoa fica em uma pasta própria (`FACE_PRIVATE_DIR/people/<uuid>/`) com o
  nome e os embeddings. **As fotos não são guardadas** (minimização de dados).
- Embeddings são dados biométricos: por isso ficam fora do repositório, e o
  `.gitignore` protege os diretórios sensíveis.
- Os logs registram só eventos e identificadores internos; nunca nomes,
  embeddings ou imagens.
- **Exclusão:** excluir uma pessoa na aba "Pessoas" remove a pasta inteira dela,
  e ela deixa de ser reconhecida imediatamente.
- Uso permitido: acadêmico e com consentimento de quem for cadastrado. Esta
  aplicação não deve ser usada para vigilância, monitoramento ou identificação
  de pessoas que não consentiram.

### Testes

```bash
python -m pytest -q
```

Cobrem a configuração, os erros, o repositório de pessoas (cadastro atômico,
exclusão completa, entradas inválidas), os serviços (cadastro, reconhecimento,
threshold, duplicidades e privacidade dos logs) e o núcleo. Usam dados
sintéticos, exceto um teste do detector real, que fica pulado a menos que a
variável `FACE_TEST_IMAGE` aponte para uma imagem pública (por exemplo, do LFW):

```powershell
$env:FACE_TEST_IMAGE = "data\lfw\lfw_home\lfw_funneled\George_W_Bush\George_W_Bush_0001.jpg"
python -m pytest tests\test_core.py -q
```

### Desenvolvimento orientado a especificações

Os requisitos e critérios de aceitação estão em
[`docs/specs/parte2/`](docs/specs/parte2/) (detecção, cadastro, reconhecimento,
pessoa desconhecida, interface e exclusão) e as decisões em
[`docs/decisions/`](docs/decisions/): ADR-004 (interface) e ADR-005 (configuração
e threshold), além dos ADRs da Parte 1.

### Uso de agentes de IA

Foram usados assistentes de IA (Claude e ChatGPT) para planejar, escrever
código, testes e documentação, a partir de especificações escritas por nós.
Nenhuma foto, embedding ou dado de pessoas foi enviado a eles: os testes usam
dados sintéticos e os dados reais ficam apenas nos computadores dos
integrantes. O código foi revisado e testado por nós, e cada integrante revisou
o do outro.

### Limitações
- O reconhecimento é feito sobre uma foto (arquivo ou captura da câmera), e não
  sobre vídeo contínuo.
- O threshold vem do LFW; em fotos de celular, com poses diferentes, a taxa de
  rejeição foi maior (Parte 1). Pode exigir recalibração.
- O ArcFace recebe o recorte do MTCNN, e não o alinhamento por cinco pontos de
  referência que ele espera.
- Sem autenticação: pensada para uso local e individual.
- Os modelos pré-treinados têm uso restrito a pesquisa não comercial.

## Parte 1: experimentação

**Modelo escolhido: ArcFace pré-treinado (InsightFace buffalo_l), sem
fine-tuning, com cosseno e threshold 0,2679.** Foi o de menor EER na validação
(0,0365), entre os modelos comparados com a mesma entrada (faces recortadas pelo
MTCNN). No teste do LFW, avaliado uma única vez, obteve EER 0,030, accuracy 0,970
e F1 macro 0,905. O fine-tuning do FaceNet (parcial e completo) não superou o
modelo pré-treinado. Detalhes em [ADR-001](docs/decisions/ADR-001-face-model.md)
e análise completa em [docs/ANALISE.md](docs/ANALISE.md).

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
FAR = FRR = 0,030. Avaliação com fotos privadas (agregada, 2 participantes, 45
fotos de teste): 88,9% reconhecidas, 11,1% rejeitadas, 0% confundidas e 0% de
impostoras do LFW aceitas como participante.

### Datasets

**LFW (Labeled Faces in the Wild), versão funneled, 13.233 imagens.**
- Fonte oficial: `vis-www.cs.umass.edu/lfw/` (página inacessível em 25/10/2026).
  Artigo: Huang, Ramesh, Berg e Learned-Miller, 2007.
- Obtenção: espelho usado pelo scikit-learn (`fetch_lfw_people`), por
  `python -m src.data.download_lfw`. O dataset não é versionado.
- Licença/termos: formais não localizados (página oficial inacessível); descrito
  como benchmark de pesquisa não comercial. Uso aqui: acadêmico, sem redistribuição.
  Detalhes em [data/README.md](data/README.md).
- Finalidade: treino do fine-tuning, validação, teste e impostoras (desconhecidas),
  em grupos de identidades disjuntos (seed 42).

**Dataset privado:** fotos dos próprios participantes, com termo de consentimento
assinado, usadas só para avaliação local e reportadas de forma agregada. Não faz
parte do repositório.

**Pesos pré-treinados (não versionados, baixados da fonte):**
- FaceNet (facenet-pytorch), pesos treinados no VGGFace2, licença CC BY-NC 4.0.
- ArcFace (InsightFace buffalo_l), treinado no WebFace600K (subconjunto do
  WebFace260M, de uso acadêmico e de pesquisa, sem uso comercial); modelos
  restritos a pesquisa não comercial.

### Reprodução

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

Equivalência entre a aplicação e a Parte 1 (só LFW):

```bash
python -m src.eval.check_app_equivalence 50
```

Os arquivos de `experiments/results/` (E01 e E02a) foram gerados por
`src/eval/extract_embeddings.py` e `src/eval/evaluate.py`; os de
`experiments/runs/` (E01 a E05), por `src/eval/run_baseline.py`. Seeds, versões
e hardware de cada execução ficam em `experiments/runs/`.

## Estrutura do repositório
- `src/face_app/`: aplicação da Parte 2 (interface, serviços, núcleo, armazenamento)
- `src/data/`, `src/eval/`, `src/models/`, `src/utils/`: código da Parte 1
- `tests/`: testes da aplicação
- `config.yaml`: configuração não sensível
- `configs/` e `experiments/`: registros dos experimentos
- `docs/specs/` e `docs/decisions/`: especificações e ADRs
- `docs/ANALISE.md` e `docs/figures/`: análise e gráficos da Parte 1
- `data/README.md`: fonte e licença dos datasets

## Privacidade
Fotos, embeddings, termos de consentimento e checkpoints privados ficam fora do
repositório (`FACE_PRIVATE_DIR`) e são eliminados ao fim da disciplina. Os
checkpoints do fine-tuning foram treinados apenas com o LFW e não são
versionados. Resultados com fotos privadas são reportados só de forma agregada.