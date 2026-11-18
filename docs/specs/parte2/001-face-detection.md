# 001: Detecção e extração de embedding

## Requisitos
- Dada uma imagem, localizar as faces e produzir, para cada uma, um recorte e um embedding comparável com o da Parte 1.

## Especificação
- Pipeline idêntico ao da Parte 1: MTCNN recorta a face em 160x160; o ArcFace (InsightFace buffalo_l) gera o embedding de 512 números, normalizado (norma 1). O threshold da spec 004 só vale nesse pipeline.
- Interface:
  - `Face`: caixa (x, y, largura, altura), recorte BGR 160x160 e confiança.
  - `FaceDetector.detect(imagem_bgr) -> list[Face]`
  - `FaceEmbedder.embed(recorte_bgr) -> ndarray(512,) float32, norma 1`
- Faces com confiança abaixo de `min_confidence` (padrão 0,9) ou menores que `min_face_px` (padrão 60) são descartadas como inadequadas.

## Critérios de aceitação
- Imagem sem face devolve lista vazia, sem exceção.
- Imagem com duas faces devolve duas `Face`.
- O embedding tem forma (512,) e norma 1 (tolerância 1e-5).
- Os testes usam um detector e um extrator falsos, sem fotos reais.