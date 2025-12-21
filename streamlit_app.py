"""Streamlit entry point for the hybrid RAG prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit.runtime.state.session_state_proxy import SessionStateProxy

from hybrid_rag.config import load_config
from hybrid_rag.pipeline import HybridAnswer, HybridRAGPipeline, build_pipeline


def _load_pipeline(env_path: Optional[str]) -> HybridRAGPipeline:
    """Load (and cache) the hybrid pipeline."""

    @st.cache_resource(show_spinner=False)
    def _cached_pipeline(cache_key: Optional[str]) -> HybridRAGPipeline:
        path = Path(cache_key) if cache_key else None
        config = load_config(path)
        return build_pipeline(config)

    return _cached_pipeline(env_path)


def _render_sidebar(state: SessionStateProxy) -> tuple[Optional[str], bool]:
    """Render sidebar controls and return env path + debug flag."""

    st.sidebar.header("Settings")
    env_path = st.sidebar.text_input(
        "Custom .env path",
        value=state.get("env_path", ""),
        help="Leave empty to use the default .env lookup order.",
    )
    debug = st.sidebar.toggle("Show retrieval context", value=state.get("debug", False))
    if st.sidebar.button("Reload pipeline"):
        state["reload_key"] = env_path or "__default__"
        st.cache_resource.clear()
        st.rerun()
    state["env_path"] = env_path
    state["debug"] = debug
    return env_path or None, debug


def _display_answer(result: HybridAnswer, debug: bool) -> None:
    """Render the assistant answer and optional retrieval context."""

    st.markdown("### Réponse")
    st.write(result.answer)

    if not debug:
        return

    with st.expander("Vector context", expanded=False):
        for doc in result.vector_context:
            title = doc.metadata.get("document_name", doc.metadata.get("source", "doc"))
            chunk_index = doc.metadata.get("chunk_index", 0)
            label = f"{title} — chunk {chunk_index}"
            st.markdown(f"**{label}**")
            st.write(doc.page_content)
            st.markdown("---")

    with st.expander("Graph reasoning", expanded=False):
        st.write(result.graph_context or "(vide)")


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
    st.title("Hybrid RAG Chatbot")
    st.caption("FAISS + Neo4j + Ollama")

    env_path, debug = _render_sidebar(st.session_state)

    try:
        pipeline = _load_pipeline(env_path)
    except Exception as exc:  # pragma: no cover - interactive feedback
        st.error(f"Configuration/initialization error: {exc}")
        return

    st.markdown("### Question")
    default_prompt = st.session_state.get("last_question", "")
    question = st.text_area("Pose ta question", value=default_prompt, height=120) or ""
    col_submit, col_clear = st.columns([1, 1], gap="small")
    submit = col_submit.button("Envoyer", type="primary")
    clear = col_clear.button("Effacer")

    if clear:
        st.session_state["last_question"] = ""
        st.session_state.pop("last_answer", None)
        st.rerun()

    if submit:
        if not question.strip():
            st.warning("Merci d'entrer une question avant d'envoyer.")
        else:
            st.session_state["last_question"] = question
            with st.spinner("Génération en cours..."):
                try:
                    result = pipeline.answer(question)
                except Exception as exc:  # pragma: no cover - interactive feedback
                    st.error(f"Erreur pendant l'inférence: {exc}")
                    return
            st.session_state["last_answer"] = result
            _display_answer(result, debug)
    elif "last_answer" in st.session_state:
        _display_answer(st.session_state["last_answer"], debug)


if __name__ == "__main__":  # pragma: no cover
    main()
