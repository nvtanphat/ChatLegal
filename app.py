from __future__ import annotations

import math
import uuid

import streamlit as st

from src.core.model_factory import create_llm
from src.database.mongo_client import MongoService
from src.inference.inference_engine import InferenceEngine
from src.rag.embedder import EmbeddingService
from src.rag.reranker import VietnameseReranker
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import ChromaVectorStore


@st.cache_resource(show_spinner=False)
def build_engine() -> InferenceEngine:
    llm = create_llm()
    embedder = EmbeddingService()
    vector_store = ChromaVectorStore()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    reranker = VietnameseReranker()

    mongo_service = None
    try:
        mongo_service = MongoService()
        mongo_service.connect()
    except Exception:
        mongo_service = None

    return InferenceEngine(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        mongo_service=mongo_service,
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:      #212121;
    --surface: #2f2f2f;
    --border:  rgba(255,255,255,0.1);
    --accent:  #10a37f;
    --text:    #ececec;
    --muted:   #8e8ea0;
    --user-bg: #2f2f2f;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: "Inter", sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"],
#MainMenu, footer { display: none !important; }

.stApp { background: var(--bg) !important; }

.block-container {
    max-width: 680px !important;
    padding: 0 1.2rem 100px !important;
    margin: 0 auto !important;
}

/* ══ WELCOME ══ */
.welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 20vh;
    text-align: center;
    margin-bottom: 2rem;
}
.welcome h1 {
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem;
}
.welcome p {
    font-size: 0.88rem;
    color: var(--muted);
    margin: 0;
}

/* ══ CHIPS ══ */
.chips-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.6rem;
    text-align: center;
}
div.stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text) !important;
    font-family: "Inter", sans-serif !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    height: auto !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
    transition: background .15s !important;
}
div.stButton > button:hover {
    background: #383838 !important;
    border-color: rgba(255,255,255,0.22) !important;
    color: var(--text) !important;
}

/* ══ MESSAGES ══ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
    gap: 0.9rem !important;
    align-items: flex-start !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 30px !important; height: 30px !important;
    min-width: 30px !important; border-radius: 50% !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    flex-shrink: 0 !important;
}
[data-testid="stChatMessageAvatarUser"]      { background: #5b7fa6 !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent) !important; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
    background: var(--user-bg) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.7rem 1rem !important;
    max-width: 82% !important;
    margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    width: 100% !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-size: 0.925rem !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
}
[data-testid="stChatMessage"] code {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 5px !important; padding: 1px 5px !important;
    font-size: 0.85rem !important; color: #a8d8a8 !important;
}

/* ══ CITATIONS ══ */
.cite-label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted); margin: 0.9rem 0 0.4rem;
}
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; margin-bottom: 0.3rem !important;
}
div[data-testid="stExpander"] summary {
    font-size: 0.8rem !important; color: var(--muted) !important;
    padding: 0.55rem 0.9rem !important;
}
div[data-testid="stExpander"] * { color: var(--text) !important; }

/* ══ CHAT INPUT — compact pill style ═══════════════════════════

   st.chat_input renders inside [data-testid="stBottomBlockContainer"]
   which Streamlit fixes to the bottom automatically.
   We just restyle the visible elements to be compact + pill-shaped.
*/

/* Bottom container background */
[data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
    padding: 0.5rem 1.2rem 0.9rem !important;
    border-top: none !important;
    max-width: 100% !important;
}

/* Inner wrapper — center + constrain width */
[data-testid="stBottomBlockContainer"] > div {
    max-width: 680px !important;
    margin: 0 auto !important;
}

/* The actual chat input container — pill shape */
[data-testid="stChatInputContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 26px !important;
    padding: 4px 4px 4px 16px !important;
    box-shadow: none !important;
    align-items: center !important;
    transition: border-color .2s !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: rgba(255,255,255,0.28) !important;
}

/* Textarea inside chat input */
[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    color: var(--text) !important;
    font-family: "Inter", sans-serif !important;
    font-size: 0.92rem !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 6px 0 !important;
    min-height: unset !important;
    max-height: 120px !important;
    resize: none !important;
    caret-color: var(--accent) !important;
    line-height: 1.5 !important;
}
[data-testid="stChatInputTextArea"]::placeholder {
    color: var(--muted) !important;
}

/* Submit button — round circle */
[data-testid="stChatInputSubmitButton"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stChatInputSubmitButton"] > button {
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    padding: 0 !important;
    background: var(--accent) !important;
    border: none !important;
    border-radius: 50% !important;
    transition: opacity .15s !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stChatInputSubmitButton"] > button:hover {
    opacity: 0.82 !important;
}
[data-testid="stChatInputSubmitButton"] > button svg {
    fill: #fff !important;
    width: 16px !important;
    height: 16px !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--muted) !important; font-size: 0.82rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
</style>
"""

PRESETS = [
    "Điều kiện hợp đồng có hiệu lực?",
    "Điều 124 BLDS nói gì?",
    "Chia thừa kế khi không có di chúc?",
]


def _prepare_citations(citations: list[dict]) -> list[dict]:
    legal = [
        item
        for item in citations
        if str(item.get("source_kind", "")).strip().lower() == "legal"
    ]
    ordered = legal if legal else list(citations)

    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in ordered:
        source_id = str(item.get("source_id", "") or "").strip()
        title = str(item.get("title", "") or "").strip()
        article = str(item.get("article", "") or "").strip()
        key = (source_id.casefold(), title.casefold(), article.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:4]


def _format_score(score: object) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value) or value <= 0:
        return ""
    return f" · {value:.2f}"


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown('<p class="cite-label">Căn cứ pháp lý</p>', unsafe_allow_html=True)
    for i, c in enumerate(_prepare_citations(citations), 1):
        label = (
            f"{i}. {c.get('title') or 'Văn bản'} — {c.get('article') or '?'}"
            f"{_format_score(c.get('score'))}"
        )
        with st.expander(label):
            st.write(str(c.get("text", "")))


def main() -> None:
    st.set_page_config(page_title="VN Law Chatbot", page_icon="⚖️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "preset_query" not in st.session_state:
        st.session_state.preset_query = ""

    engine = build_engine()
    is_empty = len(st.session_state.messages) == 0

    # ── Welcome screen: title + chips ────────────────────────────
    if is_empty:
        st.markdown(
            """
            <div class="welcome">
                <h1>⚖️ VN Law Chatbot</h1>
                <p>Trợ lý pháp lý Việt Nam</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="chips-label">Gợi ý câu hỏi</p>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, txt in enumerate(PRESETS):
            with cols[i]:
                if st.button(txt, key=f"p{i}", use_container_width=True):
                    st.session_state.preset_query = txt
                    st.rerun()

    # ── Chat history ─────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # ── Chat input (Streamlit native — auto fixed to bottom) ──────
    user_input = st.chat_input("Nhập câu hỏi của bạn…")

    # ── Resolve query ─────────────────────────────────────────────
    user_query = ""
    if user_input:
        user_query = user_input.strip()
    elif st.session_state.preset_query:
        user_query = st.session_state.preset_query
        st.session_state.preset_query = ""

    if not user_query:
        return

    # ── User message ──────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ── Assistant stream ──────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu…"):
            stream = engine.ask_stream(user_query, session_id=st.session_state.session_id)
            ans_ph  = st.empty()
            cite_ph = st.container()
            full    = ""
            cites: list[dict] = []

            def _gen():
                nonlocal full, cites
                for item in stream:
                    if item["type"] == "citations":
                        cites = item["content"]
                        with cite_ph:
                            render_citations(cites)
                    elif item["type"] == "answer_chunk":
                        full += item["content"]
                        yield item["content"]
                    elif item["type"] == "answer":
                        full = item["content"]
                        ans_ph.markdown(full)

            ans_ph.write_stream(_gen())

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "citations": cites}
    )


if __name__ == "__main__":
    main()
