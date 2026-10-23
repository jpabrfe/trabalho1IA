# ADR-001 - Modelos de reconhecimento facial

- Status: aceito
- Data: 23/10/2026
- Consulta às fontes oficiais (código e modelos): 21/10/2026

## Contexto
Identificação facial em conjunto aberto (K pessoas conhecidas + rejeição).
Comparar modelos pré-treinados, com e sem fine-tuning.

## Candidatos

### FaceNet - facenet-pytorch (InceptionResnetV1)
- Fonte oficial: https://github.com/timesler/facenet-pytorch
- Licença do código: MIT
- Pesos usados: vggface2 (portados de https://github.com/davidsandberg/facenet)
- Termos do dataset de treino (VGGFace2): a verificar na página oficial do
  projeto, www.robots.ox.ac.uk/~vgg/data/vgg_face2/
- Uso: apenas referência; pesos baixados automaticamente, não versionados no Git
- Versão usada: facenet-pytorch 2.6.0

### ArcFace - InsightFace, pacote buffalo_l (ResNet50 @ WebFace600K)
- Fonte oficial: https://github.com/deepinsight/insightface
- Licença do código: MIT
- Licença dos modelos pré-treinados: somente pesquisa não comercial
- Termos do dataset de treino (WebFace600K, subconjunto do WebFace260M): a
  verificar na página do benchmark, www.face-benchmark.org
- Uso: apenas referência; baixado automaticamente em ~/.insightface, não versionado
- Versão usada: insightface 2.1

## Critério de escolha
Menor EER na validação (LFW, 50 conhecidas e impostoras disjuntas), por ser a
métrica própria do problema de conjunto aberto e não depender de um threshold
fixo. Desempate: F1 macro. O conjunto de teste não participou da escolha.

## Resultados na validação
| Modelo  | EER    | Accuracy | F1 macro | Rank-1 | ROC-AUC |
|---------|--------|----------|----------|--------|---------|
| FaceNet | 0,0402 | 0,960    | 0,887    | 0,969  | 0,984   |
| ArcFace | 0,0486 | 0,951    | 0,859    | 0,977  | 0,986   |

## Decisão
Adotar o FaceNet (InceptionResnetV1, pesos pré-treinados no VGGFace2) como
modelo base e ponto de partida do fine-tuning, pelas razões:
1. Menor EER, maior accuracy e maior F1 macro na validação.
2. Está em PyTorch, o que permite fine-tuning parcial e completo. O ArcFace
   usado (buffalo_l) é distribuído em ONNX e não pode ser ajustado diretamente
   neste projeto.

## Implicações
- Trabalho acadêmico e não comercial: compatível com as licenças declaradas.
- Modelos derivados de fine-tuning do buffalo_l herdariam a restrição não comercial.
- Checkpoints treinados com dados privados são privados (regra do enunciado):
  nunca vão ao GitHub, independentemente da licença.
- Nenhum peso pré-treinado é copiado para o repositório.

## Ressalvas
- A diferença de EER entre os modelos é pequena (0,8 ponto percentual) frente à
  incerteza de uma validação com 741 probes conhecidas; o ArcFace é melhor em
  rank-1 e AUC.
- O ArcFace recebeu o recorte do MTCNN redimensionado, e não o alinhamento por
  cinco pontos de referência que ele espera, o que pode ter prejudicado o resultado.
- Os pesos pré-treinados vêm de bases de rostos coletados na web; pode haver
  identidades em comum com o LFW. Isso não foi verificado.
- A decisão final do projeto compara também as versões com fine-tuning (E04 e E05).

## Pendências
- Ler os termos de uso do VGGFace2 e do WebFace260M (WebFace600K) nos endereços
  acima e preencher os campos "a verificar".