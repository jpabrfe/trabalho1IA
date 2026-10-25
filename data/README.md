# Datasets

## LFW (Labeled Faces in the Wild)
- Nome: Labeled Faces in the Wild (versão funneled, 13.233 imagens)
- Fonte oficial: Página do LFW na UMass (vis-www.cs.umass.edu/lfw/) inacessível em 21/10/2026. Artigo de referência: https://people.cs.umass.edu/~elm/papers/lfw.pdf
- Obtenção: espelho no figshare usado pelo scikit-learn (fetch_lfw_people), versão funneled.
- Licença/termos: termos formais não localizados, pois a página oficial
(vis-www.cs.umass.edu/lfw/) estava inacessível em 21/10 e 25/10/2026. O LFW
é descrito em fontes secundárias como benchmark público de pesquisa, usado
para fins não comerciais, mas isso não substitui os termos oficiais. Uso neste
trabalho: acadêmico, sem redistribuição; o repositório não contém imagens, só
um script de download.
- Finalidade: pesquisa em reconhecimento facial; neste trabalho é usado para treino/fine-tuning, validação, teste e como impostoras (desconhecidas).
- Citação: Huang, Ramesh, Berg e Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. 2007.
- Split: `src/data/split.py` gera `data/splits/lfw_split.json` (seed 42), com identidades disjuntas entre conhecidas, impostoras e treino, e checagem de imagens duplicadas por hash.