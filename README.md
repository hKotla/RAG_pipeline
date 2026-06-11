# RAG Pipeline — Climate Change in India

A Retrieval-Augmented Generation (RAG) pipeline built from first principles in plain Python (no LangChain orchestration framework). It answers questions grounded in four Wikipedia articles about climate change in India, with citations back to the source chunks.

## Architecture

```
Sources (Wikipedia URLs)
    │
    ▼
[Ingestion] → chunk → embed → store
    │
    ▼
Vector Store  ←──── [Retriever] ←── User Query
                          │
                          ▼
                    [Generator / LLM]
                          │
                          ▼
                    Grounded Answer + Citations
```

Steps 1–4 (load, chunk, embed, store) run once at ingest time and persist to `./chroma_db`. Steps 5–6 (retrieve, generate) run per query.

## Tech stack

- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** ChromaDB (local, persisted to `./chroma_db`)
- **LLM:** OpenAI `gpt-4o-mini` (Anthropic / Google / Ollama also supported via `RAGConfig`)
- **Retrieval:** MMR search, top-k = 5
- **Chunking:** recursive character splitting, 1000 chars / 150 overlap
- **UI:** Streamlit

## Repository layout

```
RAG_Pipeline/
├── rag/
│   ├── config.py       # RAGConfig — providers, models, chunking, retrieval settings
│   ├── ingest.py        # load_documents, chunk_documents, build_embeddings, get_vectorstore
│   ├── retriever.py      # get_retriever (MMR, top_k=5)
│   ├── generator.py      # build_llm, get_rag_chain, format_docs
│   ├── pipeline.py        # RAGPipeline — ask(), ingest(), evaluate()
│   ├── eval.py             # Precision@K, Faithfulness, Answer Relevance metrics
│   └── query.py            # CLI: python -m rag.query "<question>"
├── evals/
│   └── golden_dataset.py    # 10 hand-crafted Q&A cases (7 normal + 3 adversarial)
├── tests/                     # pytest unit tests
├── app.py                      # Streamlit demo app
├── rag_pipeline.ipynb           # Step-by-step walkthrough notebook
└── chroma_db/                    # Persisted vector index (348 chunks, gitignored)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# then edit .env and add your OPENAI_API_KEY (and ANTHROPIC_API_KEY / GOOGLE_API_KEY if needed)
```

The `chroma_db/` index is checked in locally but gitignored — if it's missing, re-ingest the corpus via `RAGPipeline.ingest(web_urls=[...])` (see `rag_pipeline.ipynb` Step 4).

## Usage

**Streamlit demo:**

```bash
streamlit run app.py
```

Opens an interactive UI at `http://localhost:8501` — ask a question, view the grounded answer, inspect retrieved source chunks, and see pre-computed eval scores in the sidebar.

**CLI:**

```bash
python -m rag.query "What are India's renewable energy targets?"
```

**Tests:**

```bash
pytest tests/ -v
```

## Eval results

Judge model: `claude-haiku-4-5-20251001`

| ID | Question | Precision@5 | Faithfulness | Relevance |
|----|----------|------------|--------------|-----------|
| q1 | Agriculture & food security impacts | 0.80 | 0.90 | 0.90 |
| q2 | Renewable energy expansion | 1.00 | 1.00 | 0.90 |
| q3 | Himalayan glaciers & rivers | 0.60 | 1.00 | 0.75 |
| q4 | Reducing GHG emissions | 0.60 | 1.00 | 1.00 |
| q5 | Causes of deforestation | 0.80 | 1.00 | 0.90 |
| q6 | Industrial sector emissions | 1.00 | 1.00 | 0.50 |
| q7 | Health effects of climate change | 0.60 | 1.00 | 0.70 |
| q8* | Bangladesh GDP 2024 (out-of-corpus) | 0.80 | 1.00 | 0.00 |
| q9* | State flood risk budgets (unanswerable) | 0.40 | 1.00 | 1.00 |
| q10* | Modi COP28 coal pledge (hallucination-bait) | 0.40 | 1.00 | 0.00 |

**Normal queries (q1–q7) average:** Precision 0.77 ✅ / Faithfulness 0.99 ✅ / Relevance 0.81 ✅ — all above target (0.75–0.80).

\* Adversarial cases — low Relevance is expected and correct (the model declines to answer rather than hallucinate).

## Corpus

- [Climate change in India](https://en.wikipedia.org/wiki/Climate_change_in_India)
- [Climate of India](https://en.wikipedia.org/wiki/Climate_of_India)
- [Renewable energy in India](https://en.wikipedia.org/wiki/Renewable_energy_in_India)
- [Deforestation in India](https://en.wikipedia.org/wiki/Deforestation_in_India)
