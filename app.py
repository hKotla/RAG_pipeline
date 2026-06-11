"""Streamlit presentation layer for the RAG pipeline."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.generator import format_docs

# ---------------------------------------------------------------------------
# Pre-computed eval results (Step 7 — not re-run on every query)
# ---------------------------------------------------------------------------
EVAL_ROWS = [
    ("q1",  "What impacts on agriculture?",        0.80, 0.90, 0.90),
    ("q2",  "Renewable energy expansion?",          1.00, 1.00, 0.90),
    ("q3",  "Himalayan glaciers & rivers?",         0.60, 1.00, 0.75),
    ("q4",  "Reducing GHG emissions?",              0.60, 1.00, 1.00),
    ("q5",  "Causes of deforestation?",             0.80, 1.00, 0.90),
    ("q6",  "Industrial emissions?",                1.00, 1.00, 0.50),
    ("q7",  "Health effects of climate change?",    0.60, 1.00, 0.70),
    ("q8*", "Bangladesh GDP 2024? (OOC)",           0.80, 1.00, 0.00),
    ("q9*", "State flood risk budgets? (OOC)",      0.40, 1.00, 1.00),
    ("q10*","Modi COP28 coal? (halluc.)",           0.40, 1.00, 0.00),
]
NORMAL_AVG  = (0.77, 0.99, 0.81)
OVERALL_AVG = (0.70, 0.99, 0.67)

SUGGESTED_QUESTIONS = [
    "What are India's 2030 renewable energy targets?",
    "How does climate change affect Indian agriculture?",
    "What is causing deforestation in India?",
    "How are Himalayan glaciers responding to rising temperatures?",
    "What health risks does climate change pose in India?",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Climate RAG",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — eval scorecard
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Eval Scorecard")
    st.caption("Pre-computed on 10 golden questions. Judge: Claude Haiku 4.5.")

    st.markdown("**Normal queries (q1–q7)**")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Precision@5", f"{NORMAL_AVG[0]:.0%}", help="≥ 0.75 target ✅")
    col_b.metric("Faithfulness", f"{NORMAL_AVG[1]:.0%}", help="≥ 0.80 target ✅")
    col_c.metric("Relevance", f"{NORMAL_AVG[2]:.0%}", help="≥ 0.80 target ✅")

    with st.expander("Full breakdown (10 questions)"):
        import pandas as pd
        df = pd.DataFrame(
            EVAL_ROWS,
            columns=["ID", "Question (short)", "Precision@5", "Faithfulness", "Relevance"],
        )
        # Colour cells: green ≥ 0.75, yellow 0.50–0.74, red < 0.50
        def colour(val):
            if not isinstance(val, float):
                return ""
            if val >= 0.75:
                return "background-color: #d4edda; color: #155724"
            if val >= 0.50:
                return "background-color: #fff3cd; color: #856404"
            return "background-color: #f8d7da; color: #721c24"

        styled = df.style.applymap(colour, subset=["Precision@5", "Faithfulness", "Relevance"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            "\\* Adversarial (out-of-corpus / hallucination-bait) — "
            "low Relevance is expected and correct."
        )

    st.divider()
    st.markdown("**Pipeline settings**")
    st.markdown(
        "- Embeddings: `text-embedding-3-small`\n"
        "- LLM: `gpt-4o-mini`\n"
        "- Vector store: ChromaDB (local)\n"
        "- Retrieval: MMR, top-k = 5\n"
        "- Chunks: 1 000 chars / 150 overlap\n"
        "- Index: 348 chunks across 4 Wikipedia articles"
    )

    st.divider()
    st.markdown("**Corpus**")
    st.markdown(
        "- [Climate change in India](https://en.wikipedia.org/wiki/Climate_change_in_India)\n"
        "- [Climate of India](https://en.wikipedia.org/wiki/Climate_of_India)\n"
        "- [Renewable energy in India](https://en.wikipedia.org/wiki/Renewable_energy_in_India)\n"
        "- [Deforestation in India](https://en.wikipedia.org/wiki/Deforestation_in_India)"
    )

# ---------------------------------------------------------------------------
# Pipeline (cached — connects to existing chroma_db, no re-embedding)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to vector store…")
def load_pipeline() -> RAGPipeline:
    config = RAGConfig(llm_provider="openai", embedding_provider="openai")
    return RAGPipeline(config)

pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🌏 Climate Change in India — RAG Demo")
st.caption(
    "Ask anything about climate change, renewable energy, deforestation, or glaciers in India. "
    "Answers are grounded in the 4 Wikipedia articles listed in the sidebar."
)

def _set_question(q: str) -> None:
    st.session_state["question_input"] = q

# Suggested questions as quick-select pills
st.markdown("**Try a question:**")
pill_cols = st.columns(len(SUGGESTED_QUESTIONS))
for col, q in zip(pill_cols, SUGGESTED_QUESTIONS):
    col.button(
        q,
        use_container_width=True,
        key=f"pill_{q[:20]}",
        on_click=_set_question,
        args=(q,),
    )

question = st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. What are India's 2030 renewable energy targets?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary", disabled=not question.strip())

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if ask_clicked and question.strip():
    with st.spinner("Retrieving and generating…"):
        retrieved = pipeline.retriever.invoke(question)
        context   = format_docs(retrieved)
        answer    = pipeline.rag_chain.invoke({"question": question, "context": context})

    # Answer
    st.subheader("Answer")
    st.markdown(answer)

    # Sources panel
    st.subheader(f"Sources — {len(retrieved)} chunks retrieved")
    for i, doc in enumerate(retrieved, 1):
        source  = doc.metadata.get("source", "unknown")
        page    = doc.metadata.get("page", "")
        loc     = f" · p.{page}" if page != "" else ""
        preview = doc.page_content[:300].replace("\n", " ")
        label   = f"[{i}] {source}{loc}"

        with st.expander(label):
            st.markdown(f"[Open source]({source})" if source.startswith("http") else f"`{source}`")
            st.caption(preview + ("…" if len(doc.page_content) > 300 else ""))
            with st.expander("Full chunk text"):
                st.text(doc.page_content)
