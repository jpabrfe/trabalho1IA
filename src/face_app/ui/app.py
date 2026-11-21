"""Interface Streamlit (spec 005, ADR-004). So chama os servicos: sem logica de
deteccao, embedding ou armazenamento.

Executar (da raiz do projeto, com FACE_PRIVATE_DIR definida):
    streamlit run src/face_app/ui/app.py
"""
import logging
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.face_app.config import load_settings
from src.face_app.logging_setup import setup_logging  # noqa: E402
from src.face_app.errors import ConfigError, FaceAppError  # noqa: E402
from src.face_app.services.factory import build_services  # noqa: E402

log = logging.getLogger("face_app.ui")
TIPOS = ["jpg", "jpeg", "png"]


@st.cache_resource(show_spinner="Carregando modelos (a primeira vez demora)...")
def _carregar():
    from src.face_app.core.detector import MtcnnDetector
    from src.face_app.core.embedder import ArcFaceEmbedder

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_file)
    return settings, build_services(settings, MtcnnDetector(), ArcFaceEmbedder())


LADO_MAX = 1280


def _ler_imagem(arquivo):
    """Decodifica um arquivo enviado pelo navegador para BGR; None se ilegível."""
    dados = np.frombuffer(arquivo.getvalue(), dtype=np.uint8)
    if dados.size == 0:
        return None
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        return None
    maior = max(imagem.shape[:2])
    if maior > LADO_MAX:
        fator = LADO_MAX / maior
        imagem = cv2.resize(imagem, None, fx=fator, fy=fator, interpolation=cv2.INTER_AREA)
    return imagem


def _sem_acentos(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _desenhar(imagem_bgr, resultados):
    """Caixa e rótulo sobre a imagem; devolve RGB para exibição."""
    saida = imagem_bgr.copy()
    escala = max(0.5, saida.shape[1] / 800)
    espessura = max(2, int(escala * 2))
    for i, r in enumerate(resultados, start=1):
        x, y, w, h = r.caixa
        cor = (0, 160, 0) if r.nome else (0, 0, 200)  # BGR
        rotulo = f"{i}: {_sem_acentos(r.nome) if r.nome else 'desconhecido'}"
        cv2.rectangle(saida, (x, y), (x + w, y + h), cor, espessura)
        cv2.putText(saida, rotulo, (x, max(int(20 * escala), y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * escala, cor, espessura)
    return cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)


def _chamar(funcao, *args):
    """Chama um serviço; mostra o erro (sem traceback) e devolve (ok, resultado)."""
    try:
        with st.spinner("Processando..."):
            return True, funcao(*args)
    except FaceAppError as erro:
        st.error(erro.mensagem)
    except Exception:
        log.exception("erro inesperado na interface")
        st.error("Erro inesperado. Tente novamente; os detalhes ficam no log da aplicação.")
    return False, None


def _aba_cadastrar(services):
    st.subheader("Cadastrar pessoa")
    nome = st.text_input("Nome")
    consentiu = st.checkbox("Esta pessoa leu e assinou o termo de consentimento.")
    arquivos = st.file_uploader("Fotos (uma ou mais)", type=TIPOS,
                                accept_multiple_files=True, key="cad_arquivos")
    foto = st.camera_input("Ou tire uma foto", key="cad_camera")

    if st.button("Cadastrar", type="primary"):
        if not consentiu:
            st.error("Só cadastre pessoas que assinaram o termo de consentimento.")
            return
        fontes = list(arquivos or []) + ([foto] if foto else [])
        imagens = [_ler_imagem(f) for f in fontes]
        ok, resultado = _chamar(services.cadastro.enroll, nome, imagens)
        if ok:
            st.success(f"Cadastro concluído com {resultado.n_embeddings} imagem(ns).")


def _aba_reconhecer(settings, services):
    st.subheader("Reconhecer")
    if services.reconhecimento.n_pessoas() == 0:
        st.info("Nenhuma pessoa cadastrada.")
    arquivo = st.file_uploader("Foto", type=TIPOS, key="rec_arquivo")
    foto = st.camera_input("Ou tire uma foto", key="rec_camera")
    fonte = foto or arquivo

    if st.button("Reconhecer", type="primary"):
        if fonte is None:
            st.warning("Envie ou tire uma foto.")
            return
        imagem = _ler_imagem(fonte)
        ok, resultados = _chamar(services.reconhecimento.reconhecer, imagem)
        if not ok:
            return
        if not resultados:
            st.warning("Nenhuma face adequada encontrada (ausente, pequena demais ou pouco nítida).")
            return
        st.image(_desenhar(imagem, resultados), caption="Resultado")
        for i, r in enumerate(resultados, start=1):
            rotulo = r.nome if r.nome else "desconhecido"
            st.write(f"**Face {i}:** {rotulo} (similaridade {r.similaridade:.3f})")
        st.caption(f"Limiar de decisão em uso: {settings.threshold} (cosseno).")


def _aba_pessoas(services):
    st.subheader("Pessoas cadastradas")

    pendente = st.session_state.get("excluir_id")
    if pendente:
        nome = st.session_state.get("excluir_nome", "esta pessoa")
        st.warning(f"Excluir **{nome}** remove o cadastro e todos os embeddings dela. "
                   "Não há como desfazer.")
        c1, c2 = st.columns(2)
        if c1.button("Confirmar exclusão", type="primary"):
            ok, removida = _chamar(services.pessoas.excluir, pendente)
            st.session_state.pop("excluir_id", None)
            st.session_state.pop("excluir_nome", None)
            if ok and removida:
                st.success("Pessoa excluída.")
            elif ok:
                st.warning("Pessoa não encontrada.")
        if c2.button("Cancelar"):
            st.session_state.pop("excluir_id", None)
            st.session_state.pop("excluir_nome", None)
            st.rerun()

    pessoas = services.pessoas.listar()
    if not pessoas:
        st.info("Nenhuma pessoa cadastrada.")
        return
    for p in pessoas:
        col_nome, col_botao = st.columns([4, 1])
        col_nome.write(p.nome)
        if col_botao.button("Excluir", key=f"excluir_{p.id}"):
            st.session_state["excluir_id"] = p.id
            st.session_state["excluir_nome"] = p.nome
            st.rerun()

def main():
    st.set_page_config(page_title="Reconhecimento facial")
    st.title("Reconhecimento facial")
    try:
        settings, services = _carregar()
    except ConfigError as erro:
        st.error(erro.mensagem)
        st.stop()
    except Exception:
        log.exception("falha ao iniciar a aplicação")
        st.error("Não foi possível iniciar a aplicação. Verifique a configuração e os modelos.")
        st.stop()

    st.info("Privacidade: os dados ficam somente neste computador. "
            "Nenhuma imagem é enviada pela internet.")
    aba_cad, aba_rec, aba_pes = st.tabs(["Cadastrar", "Reconhecer", "Pessoas"])
    with aba_cad:
        _aba_cadastrar(services)
    with aba_rec:
        _aba_reconhecer(settings, services)
    with aba_pes:
        _aba_pessoas(services)


if __name__ == "__main__":
    main()