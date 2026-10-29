"""
Phase 6a: Chat interface for the UAE Labour Law assistant.

The interface talks to the FastAPI service over HTTP, so the two can run separately
(and in separate Docker containers later).

Run (with the API already running on port 8000):
    streamlit run ui/app.py
"""

import html
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 180  # seconds; the reranker on a laptop CPU can take a while

EXAMPLES = [
    "How many days of annual leave does an employee get?",
    "My employer is keeping my passport. Is that allowed?",
    "How long do I have to file a claim for unpaid wages?",
    "كم يوم إجازة سنوية يستحق العامل؟",
    "هل يجوز لصاحب العمل خصم مبلغ من راتبي بسبب التأخر عن العمل؟",
]

st.set_page_config(page_title="UAE Labour Law Assistant", page_icon="⚖️", layout="centered")

st.markdown(
    """
    <style>
      .rtl { direction: rtl; text-align: right; font-size: 1.05rem; line-height: 1.9; }
      .ltr { direction: ltr; text-align: left; }
      .source-text { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.7;
                     max-height: 320px; overflow-y: auto; padding: 0.5rem 0.75rem;
                     border-radius: 0.4rem; background: rgba(128,128,128,0.08); }
      .cite { font-size: 0.85rem; opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------

def is_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", text or ""))


def render_text(text: str, css_class: str = "") -> None:
    """Show text with the right direction: right-to-left for Arabic, left-to-right otherwise."""
    direction = "rtl" if is_arabic(text) else "ltr"
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="{direction} {css_class}">{safe}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


def ask(question: str) -> dict:
    """Call the API. Returns the response, or {'error': message} on failure."""
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=REQUEST_TIMEOUT)
    except requests.ConnectionError:
        return {"error": f"Can't reach the API at {API_URL}. Is it running?"}
    except requests.Timeout:
        return {"error": "The request took too long. Please try again."}
    if r.status_code == 200:
        return r.json()
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # validation errors come back as a list
        detail = "Please enter a question of at least 3 characters."
    return {"error": detail or f"The API returned an error ({r.status_code})."}


def timing(result: dict) -> str:
    if result.get("cached"):
        return "instant (asked before)"
    text = f'{result["latency_seconds"]}s'
    if result.get("search_seconds") is not None and result.get("llm_seconds") is not None:
        text += f' (search {result["search_seconds"]}s, answer {result["llm_seconds"]}s)'
    return text


def show_answer(result: dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return

    render_text(result["answer"])

    if result.get("found"):
        arts = ", ".join(str(a) for a in result["cited_articles"])
        label = "المواد" if result.get("language") == "ar" else "Articles"
        st.markdown(f'<div class="cite">📖 {label}: {arts} · {timing(result)}</div>',
                    unsafe_allow_html=True)

        with st.expander("Show sources / عرض المصادر"):
            seen = set()
            for s in result.get("sources", []):
                key = (s["article_number"], s["language"], s["part"])
                if key in seen:
                    continue
                seen.add(key)
                lang = "العربية" if s["language"] == "ar" else "English"
                part = f" (part {s['part']})" if s["part"] > 1 else ""
                st.markdown(f"**Article {s['article_number']}{part} · {lang}**")
                render_text(s["text"], "source-text")
    else:
        st.markdown(f'<div class="cite">{timing(result)}</div>', unsafe_allow_html=True)

    st.caption(result.get("disclaimer", "Not legal advice."))


# ---------- sidebar ----------

with st.sidebar:
    st.header("⚖️ About")
    st.write(
        "Ask about the **UAE Labour Law** (Federal Decree-Law No. 33 of 2021) in Arabic or English. "
        "Answers come only from the law and cite the articles they use."
    )

    health = get_health()
    if health is None:
        st.error("API: offline")
    elif health.get("status") == "ok":
        st.success(f"API: online · {health['chunks_indexed']} chunks · {health['retrieval_mode']} search")
    else:
        st.warning(f"API: degraded (database: {health.get('database')})")

    st.subheader("Try an example")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.pending = example

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Not legal advice. Check the official text or a qualified professional.")


# ---------- chat ----------

st.title("UAE Labour Law Assistant")
st.markdown('<div class="rtl">مساعد قانون العمل الإماراتي</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            render_text(message["content"])
        else:
            show_answer(message["result"])

typed = st.chat_input("Ask a question... / اكتب سؤالك...")
question = typed or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_text(question)

    with st.chat_message("assistant"):
        spinner_text = "جارٍ البحث في القانون..." if is_arabic(question) else "Searching the law..."
        with st.spinner(spinner_text):
            result = ask(question)
        show_answer(result)
    st.session_state.messages.append({"role": "assistant", "result": result})