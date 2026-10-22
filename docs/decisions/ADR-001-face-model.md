\# ADR-001 - Modelos de reconhecimento facial (RASCUNHO)



Status: rascunho (dia 1). Decisão final após os experimentos.

Consulta às fontes oficiais em: 21/10/2026.



\## Contexto

Identificação facial em conjunto aberto (K pessoas conhecidas + rejeição).

Comparar modelos pré-treinados, com e sem fine-tuning.



\## Candidatos



\### FaceNet - facenet-pytorch (InceptionResnetV1)

\- Fonte oficial: https://github.com/timesler/facenet-pytorch

\- Licença do código: MIT

\- Pesos: vggface2 e casia-webface (portados de https://github.com/davidsandberg/facenet)

\- Termos dos datasets de treino (VGGFace2, CASIA-Webface): A VERIFICAR

\- Uso: apenas referência; pesos baixados automaticamente, não versionados no Git

\- Versão usada: facenet-pytorch 2.6.0



\### ArcFace - InsightFace, pacote buffalo\_l (ResNet50 @ WebFace600K)

\- Fonte oficial: https://github.com/deepinsight/insightface

\- Licença do código: MIT

\- Licença dos modelos pré-treinados: somente pesquisa não comercial

\- Termos do dataset de treino (WebFace600K): A VERIFICAR

\- Uso: apenas referência; baixado automaticamente em \~/.insightface, não versionado

\- Versão usada: insightface 2.1



\## Implicações

\- Trabalho acadêmico e não comercial: compatível com ambas as licenças.

\- Modelos derivados de fine-tuning do buffalo\_l herdam a restrição não comercial.

\- Checkpoints treinados com dados privados são privados (regra do enunciado):

&#x20; nunca vão ao GitHub, independentemente da licença.

\- Nenhum peso pré-treinado é copiado para o repositório.



\## Decisão

(a preencher após os experimentos)



\## Pendências

\- Ler os termos de VGGFace2, CASIA-Webface e WebFace600K e preencher acima.

