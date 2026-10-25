# Análise dos resultados (Parte 1)

Pergunta do trabalho: qual configuração de modelo, treinamento e/ou fine-tuning
tem o melhor desempenho em identificação facial em conjunto aberto?

## 1. Protocolo
- Dados: LFW (funneled), 13.233 imagens. Grupos de identidades disjuntos:
  50 conhecidas (5 imagens de cadastro cada; o resto dividido em probes de
  validação e de teste), impostoras de validação, impostoras de teste e
  identidades de treino do fine-tuning (1.141 identidades, 5.334 imagens).
- Prevenção de leakage: identidades disjuntas entre os grupos (verificado por
  `assert`) e nenhuma imagem com o mesmo hash em dois grupos (0 removidas).
  Quase-duplicatas além do hash idêntico não foram verificadas.
- Decisão: protótipo por pessoa (média normalizada das 5 imagens de cadastro);
  aceita se a maior similaridade de cosseno for maior ou igual ao threshold.
- Threshold e escolha do modelo: só na validação (EER). O teste foi avaliado
  uma única vez, no modelo final.
- Métricas: EER, FAR, FRR, accuracy, F1 macro, rank-1, ROC-AUC e identificação
  errada. A accuracy e o F1 são calculados sobre 51 classes (50 pessoas mais
  "desconhecido").

## 2. Resultados
Validação (mesma entrada para todos: faces recortadas pelo MTCNN):

| Exp. | Modelo | EER | Accuracy | F1 macro | Rank-1 | ROC-AUC |
|------|--------|-----|----------|----------|--------|---------|
| E01 | FaceNet pré-treinado | 0,0402 | 0,960 | 0,887 | 0,969 | 0,984 |
| E02 | ArcFace pré-treinado | 0,0365 | 0,964 | 0,871 | 0,966 | 0,984 |
| E04 | FaceNet, ajuste parcial | 0,0432 | 0,957 | 0,867 | 0,969 | 0,979 |
| E05 | FaceNet, ajuste completo | 0,0404 | 0,960 | 0,876 | 0,969 | 0,977 |

Modelo final (E02), teste do LFW, avaliado uma única vez: EER 0,030, accuracy
0,970, F1 macro 0,905, rank-1 0,971, ROC-AUC 0,988, identificação errada 0,13%.

## 3. Qual modelo foi melhor e por quê
O ArcFace pré-treinado teve o menor EER na validação (0,0365, contra 0,0402 do
FaceNet). A diferença, porém, é pequena: com 741 probes conhecidas o erro-padrão
do FRR é de cerca de 0,7 ponto percentual, e com 3.149 impostoras o do FAR é de
cerca de 0,35. Os dois modelos são, na prática, equivalentes na validação; o
FaceNet é melhor em F1 macro e em rank-1. O ArcFace foi escolhido pelo critério
definido antes (menor EER), que é o mais adequado para conjunto aberto.
Uma explicação possível, não testada: o ArcFace é treinado com uma loss
angular (margem no espaço do cosseno), o que combina com a decisão por
similaridade de cosseno com threshold; e foi treinado em uma base maior
(WebFace600K) que a do FaceNet (VGGFace2).

## 4. O que melhorou e o que piorou
- **Recorte do MTCNN (melhorou):** o ArcFace em imagens originais, sem recorte
  (E02a), teve EER 0,0486; com o recorte, 0,0365. O pré-processamento mudou o
  resultado em mais de 1 ponto percentual, mais que qualquer variação de modelo.
- **Cosseno vs. euclidiana (sem efeito):** com embeddings normalizados as duas
  métricas dão a mesma ordenação (E03). Só o valor do threshold muda de escala.
- **Fine-tuning parcial (piorou levemente):** EER de 0,0402 para 0,0432.
- **Fine-tuning completo (neutro):** EER 0,0404, empate com o baseline; ROC-AUC
  e F1 macro ficaram abaixo.

## 5. Impacto do fine-tuning
Com 5.334 imagens de treino, cabeça ArcFace inicializada do zero (1.141 classes),
15 épocas e hiperparâmetros não ajustados, o fine-tuning não superou o modelo
pré-treinado. Explicações possíveis, não testadas: os pesos pré-treinados já
foram aprendidos com milhões de imagens, e o ajuste com poucas identidades pode
sobreajustar a elas e piorar a generalização a identidades novas (compatível
com a queda de ROC-AUC e F1 macro no E05); e não houve busca de learning rate,
número de épocas ou da loss (Triplet não foi testada). O resultado não permite
concluir que o fine-tuning não ajuda em geral, só que não ajudou nesta configuração.

## 6. Impacto do threshold
O threshold de 0,2679 foi fixado na validação no ponto FAR = FRR. No teste,
FAR e FRR ficaram em 3,0% cada, ou seja, o ponto de operação generalizou.
Subir o threshold reduziria os desconhecidos aceitos ao custo de rejeitar mais
pessoas cadastradas, e baixá-lo faria o contrário; a escolha depende da
aplicação (ADR-003). As distribuições de similaridade estão em `docs/figures/`.
O histograma de similaridades na validação (`docs/figures/arcface_cosseno_val.png`)
mostra três grupos. As pessoas conhecidas identificadas corretamente ficam à
direita do threshold, concentradas entre 0,5 e 0,8, com pico perto de 0,7. As
desconhecidas e as conhecidas cuja maior similaridade foi com outra pessoa
ficam quase todas à esquerda do threshold, concentradas entre 0,1 e 0,2; o
pico das desconhecidas tem densidade em torno de 10, e o do grupo de conhecidas
com identificação errada chega a cerca de 50. A separação entre os grupos é
nítida, o que se reflete no ROC-AUC de 0,984. A sobreposição fica nas caudas
perto do threshold (0,2679): impostoras com similaridade alta passam como
conhecidas (FAR) e conhecidas com similaridade baixa são rejeitadas (FRR), o que
explica os erros de cerca de 3 a 4% em cada lado. Os histogramas são normalizados
por grupo (densidade), então a altura dos picos mostra o quanto cada grupo se
concentra, e não quantas amostras ele tem; o grupo de conhecidas com
identificação errada é pequeno, e por isso seu pico é estreito e alto.

## 7. Erros encontrados
- **LFW:** poucas conhecidas confundidas com outra (0,13%); a maior parte dos
  erros são rejeições de conhecidas e aceitações de impostoras, em proporções iguais.
- **Fotos privadas (agregado, 2 participantes, 45 fotos de teste):** 40
  reconhecidas (88,9%), 5 rejeitadas como desconhecidas (11,1%), nenhuma confundida
  com outra pessoa e nenhuma impostora do LFW aceita como participante. Na
  inspeção manual, as 4 rejeitadas de um dos participantes tinham a cabeça virada
  para o lado de forma mais acentuada que nas demais, e uma delas ângulo de baixo
  para cima. [COMPLETAR: o que o colega observou na foto rejeitada dele.] A causa
  provável, não testada, é a variação de pose frente a um protótipo formado por
  poucas fotos de cadastro (5 e 7). Como as rejeições (11%) superam as do LFW
  (cerca de 3%), o limiar calibrado só no LFW parece rigoroso para fotos de celular,
  mas com 45 fotos o intervalo é largo (cerca de 77% a 95% de reconhecimento).
- **F1 macro menor que o ponderado:** as classes são desbalanceadas (poucas
  identidades do LFW concentram muitas probes).

## 8. Limitações
- Validação e teste com poucas probes conhecidas (741 e 764); diferenças de
  menos de 1 ponto percentual não são conclusivas.
- O ArcFace recebeu o recorte do MTCNN redimensionado, e não o alinhamento por
  cinco pontos de referência que ele espera.
- O threshold foi calibrado só com o LFW; em fotos de celular com outra pose a
  rejeição foi maior.
- O dataset privado é pequeno (2 participantes, 45 fotos de teste) e foi
  avaliado localmente por cada participante, com 5 e 7 fotos de cadastro; os
  resultados são indicativos.
- Os pesos pré-treinados vêm de bases de rostos coletados na web e podem ter
  identidades em comum com o LFW (não verificado).
- Os baselines E01 e E02a foram avaliados também no teste antes da escolha final;
  a decisão se baseou apenas na validação.
- Não foi feita busca de hiperparâmetros, e protótipo vs. k-NN (E06) ficou fora.

## 9. Custo computacional e relação com o desempenho
- Hardware (participante A): 13th Gen Intel(R) Core(TM) i5-1335U, 8GB RAM.
- Hardware (participante B): notebook Acer Nitro V15 (ANV15-51), Intel Core
  i7-13620H (2,40 GHz), 16 GB de RAM DDR5 e NVIDIA GeForce RTX 4060 Laptop
  (8 GB de VRAM).
- Participante A: extração de embeddings do LFW alinhado em CPU (7.899 imagens)
  levou cerca de 20 min com o FaceNet e 35 min com o ArcFace na primeira
  execução; depois, o cache torna as avaliações quase instantâneas. Avaliar um
  checkpoint ajustado levou de 8 a 12 min.
- Participante B: detecção e alinhamento (MTCNN) de 27 fotos privadas em cerca
  de 5 s; extração de embeddings do ArcFace para o LFW completo (7.899 imagens)
  em 8 min 24 s (cerca de 15,7 it/s); avaliação em conjunto aberto e métricas
  instantâneas.
- Fine-tuning (E04 e E05): feito no Colab com GPU, alguns minutos por experimento; a loss por época está nos arquivos de histórico de cada execução.
- Parâmetros: 23.5 milhões (FaceNet) e 43.6 milhões (ArcFace buffalo_l).
- Os tempos dos dois computadores não são comparáveis diretamente (hardware
  diferente), mas nos dois casos a extração de embeddings do LFW completo leva
  de minutos a algumas dezenas de minutos. O modelo final dispensa treino. O
  fine-tuning custou GPU e tempo sem trazer ganho; nesta configuração, mais
  complexidade não melhorou o desempenho.
## 10. Conclusão
O ArcFace pré-treinado, com recorte MTCNN, protótipos e threshold de cosseno
escolhido na validação, é a melhor configuração encontrada, com EER 0,030 no
teste do LFW. O pré-processamento teve mais efeito que o fine-tuning, e os
principais pontos fracos são a variação de pose em fotos reais e o tamanho
pequeno dos conjuntos de avaliação.