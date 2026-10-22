# Datasets

## LFW (Labeled Faces in the Wild)
- Nome: Labeled Faces in the Wild (versão funneled, 13.233 imagens)
- Fonte oficial: Página do LFW na UMass (vis-www.cs.umass.edu/lfw/) inacessível em 21/10/2026. Artigo de referência: https://people.cs.umass.edu/~elm/papers/lfw.pdf
- Obtenção: espelho no figshare usado pelo scikit-learn (fetch_lfw_people), versão funneled.
- Licença/termos: não localizados, pois a página oficial estava inacessível em 21/10/2026. A verificar antes da entrega.
- Finalidade: pesquisa em reconhecimento facial; neste trabalho é usado para treino/fine-tuning, validação, teste e como impostoras (desconhecidas).
- Citação: Huang, Ramesh, Berg e Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. 2007.
- Split: `src/data/split.py` gera `data/splits/lfw_split.json` (seed 42), com identidades disjuntas entre conhecidas, impostoras e treino, e checagem de imagens duplicadas por hash.