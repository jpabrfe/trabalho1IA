# ADR-001 - Modelos de reconhecimento facial

- Status: aceito
- Data: 24/10/2026
- Consulta às fontes oficiais (código e modelos): 21/10/2026

## Contexto
Identificação facial em conjunto aberto (K pessoas conhecidas + rejeição).
Comparar modelos pré-treinados, com e sem fine-tuning.

## Candidatos

### FaceNet - facenet-pytorch (InceptionResnetV1)
- Fonte oficial: https://github.com/timesler/facenet-pytorch
- Licença do código: MIT
- Pesos usados: vggface2 (portados de https://github.com/davidsandberg/facenet)
- Termos do dataset de treino (VGGFace2): licença Creative Commons
  Atribuição-NãoComercial 4.0 (CC BY-NC 4.0), conforme a página oficial do
  dataset, lida em 25/10/2026. Uso compatível: trabalho acadêmico, não
  comercial, sem redistribuição do dataset. Não foi determinado se a licença se
  estende aos pesos treinados com ele. Referência: Cao, Shen, Xie, Parkhi e
  Zisserman, "VGGFace2: A dataset for recognising faces across pose and age"
  (arXiv:1710.08092).
- Uso: apenas referência; pesos baixados automaticamente, não versionados no Git
- Versão usada: facenet-pytorch 2.6.0

### ArcFace - InsightFace, pacote buffalo_l (ResNet50 @ WebFace600K)
- Fonte oficial: https://github.com/deepinsight/insightface
- Licença do código: MIT
- Licença dos modelos pré-treinados: somente pesquisa não comercial
- Termos do dataset de treino (WebFace600K, subconjunto do WebFace260M): - Termos do dataset de treino (WebFace600K, subconjunto do WebFace260M): os
  termos de uso do WebFace260M, lidos em 25/10/2026, destinam o dataset
  exclusivamente a fins acadêmicos, de pesquisa e informativos, e proíbem uso
  comercial. Uso compatível: trabalho acadêmico, sem redistribuição. Não foi
  determinado se os termos se estendem aos pesos treinados com ele., www.face-benchmark.org
- Uso: apenas referência; baixado automaticamente em ~/.insightface, não versionado
- Versão usada: insightface 2.1

## Critério de escolha
Menor EER na validação (LFW, 50 conhecidas e impostoras disjuntas), por ser a
métrica própria do problema de conjunto aberto e não depender de um threshold
fixo. Desempate: F1 macro. Todos os modelos recebem a mesma entrada (faces
recortadas pelo MTCNN, em data/lfw/aligned). O conjunto de teste não participou
da escolha.

## Resultados na validação
| Modelo                       | EER    | Accuracy | F1 macro | Rank-1 | ROC-AUC |
|------------------------------|--------|----------|----------|--------|---------|
| E01 FaceNet pré-treinado     | 0,0402 | 0,960    | 0,887    | 0,969  | 0,984   |
| E02 ArcFace pré-treinado     | 0,0365 | 0,964    | 0,871    | 0,966  | 0,984   |
| E04 FaceNet, ajuste parcial  | 0,0432 | 0,957    | 0,867    | 0,969  | 0,979   |
| E05 FaceNet, ajuste completo | 0,0404 | 0,960    | 0,876    | 0,969  | 0,977   |

Variante do E02 com as imagens originais do LFW, sem recorte (E02a): EER 0,0486,
accuracy 0,951, F1 macro 0,859, rank-1 0,977, ROC-AUC 0,986. Não entra na
comparação entre modelos, porque a entrada é diferente; serve para mostrar o
efeito do pré-processamento.

## Decisão
Adotar o ArcFace pré-treinado (InsightFace buffalo_l), sem fine-tuning, como
modelo final, por ter o menor EER na validação.

O FaceNet foi o modelo-base dos experimentos de fine-tuning porque está em
PyTorch; o buffalo_l é distribuído em ONNX e não pode ser ajustado diretamente
neste projeto. O fine-tuning parcial (E04) e o completo (E05), com ArcFace loss
e só com dados do LFW, não superaram o FaceNet pré-treinado (E01) na validação.

## Implicações
- Trabalho acadêmico e não comercial: compatível com as licenças declaradas.
- Modelos derivados de fine-tuning do buffalo_l herdariam a restrição não comercial.
- Checkpoints treinados com dados privados são privados (regra do enunciado):
  nunca vão ao GitHub, independentemente da licença. Os checkpoints deste
  projeto foram treinados só com o LFW e, mesmo assim, não são versionados.
- Nenhum peso pré-treinado é copiado para o repositório.

## Ressalvas
- A diferença de EER entre ArcFace e FaceNet (0,37 ponto percentual) é pequena
  frente à incerteza de uma validação com 741 probes conhecidas; o FaceNet é
  melhor em F1 macro e em rank-1.
- O ArcFace espera alinhamento por cinco pontos de referência; aqui recebeu o
  recorte do MTCNN redimensionado para 112x112. Com as imagens originais, sem
  recorte, o EER foi 0,0486: o pré-processamento altera o resultado.
- Os pesos pré-treinados vêm de bases de rostos coletados na web; pode haver
  identidades em comum com o LFW. Isso não foi verificado.
- Os baselines E01 e E02a foram avaliados também no conjunto de teste antes
  da escolha final; a decisão se baseou apenas na validação.

## Pendências
- Ler os termos de uso do VGGFace2 e do WebFace260M (WebFace600K) nos endereços
  acima e preencher os campos "a verificar".