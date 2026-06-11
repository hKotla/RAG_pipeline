from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from rag.config import RAGConfig
from rag.ingest import load_documents, chunk_documents, build_embeddings, get_vectorstore
from rag.retriever import get_retriever
from rag.generator import build_llm, get_rag_chain, format_docs


class RAGPipeline:
    """Orchestration class that coordinates config, ingestion, retrieval, and generation."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # Resolve the correct API key for the chosen embedding provider
        emb_api_key = ""
        if self.config.embedding_provider.lower() == "openai":
            emb_api_key = self.config.openai_api_key
        elif self.config.embedding_provider.lower() == "google":
            emb_api_key = self.config.google_api_key

        self.embeddings = build_embeddings(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model,
            api_key=emb_api_key,
        )

        self.vectorstore = get_vectorstore(
            persist_dir=self.config.chroma_persist_dir,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings,
        )

        self.retriever = get_retriever(
            vectorstore=self.vectorstore,
            search_type=self.config.search_type,
            k=self.config.top_k,
        )

        # Resolve the correct API key for the chosen LLM provider
        llm_api_key = ""
        if self.config.llm_provider.lower() == "openai":
            llm_api_key = self.config.openai_api_key
        elif self.config.llm_provider.lower() == "anthropic":
            llm_api_key = self.config.anthropic_api_key
        elif self.config.llm_provider.lower() == "google":
            llm_api_key = self.config.google_api_key

        self.llm = build_llm(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            api_key=llm_api_key,
        )

        # Chain takes {question, context} — retrieval happens in ask()/evaluate()
        # so the same retrieved docs are used for both source display and generation.
        self.rag_chain = get_rag_chain(llm=self.llm)

    def ingest(
        self,
        pdf_paths: Optional[List[str]] = None,
        docx_paths: Optional[List[str]] = None,
        text_paths: Optional[List[str]] = None,
        web_urls: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load and index documents, adding them to the vector store."""
        raw_docs = load_documents(
            pdf_paths=pdf_paths,
            docx_paths=docx_paths,
            text_paths=text_paths,
            web_urls=web_urls,
        )
        if not raw_docs:
            print("No documents loaded. Add paths or URLs to ingest.")
            return []

        chunks = chunk_documents(
            documents=raw_docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        print(f"Embedding and indexing {len(chunks)} chunks... (this may take a moment)")
        self.vectorstore.add_documents(chunks)
        print(f"Index updated. Total documents in collection: {self.vectorstore._collection.count()}")
        return chunks

    def ask(self, question: str, show_sources: bool = True) -> str:
        """Query the RAG pipeline."""
        retrieved = self.retriever.invoke(question)
        context   = format_docs(retrieved)
        answer    = self.rag_chain.invoke({"question": question, "context": context})

        print(f"Q: {question}")
        print(f"\nAnswer:\n{answer}")

        if show_sources:
            print(f"\nSources ({len(retrieved)} chunks retrieved):")
            for i, doc in enumerate(retrieved, 1):
                src     = doc.metadata.get("source", "unknown")
                page    = doc.metadata.get("page", "")
                loc     = f", p.{page}" if page != "" else ""
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"  [{i}] {src}{loc} -> {preview}...")

        return answer

    def evaluate(self, eval_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run standard keyword-based evaluation cases."""
        if not eval_cases:
            print("Add evaluation cases to test your pipeline.")
            return {"passed": 0, "total": 0}

        print("Running evaluation...\n")
        passed = 0
        for case in eval_cases:
            retrieved = self.retriever.invoke(case["question"])
            context   = format_docs(retrieved)
            answer    = self.rag_chain.invoke({"question": case["question"], "context": context})
            hits      = [kw for kw in case["must_contain"] if kw.lower() in answer.lower()]
            ok        = len(hits) == len(case["must_contain"])
            passed   += ok
            status    = "PASS" if ok else "FAIL"
            print(f"{status}  Q: {case['question']}")
            if not ok:
                missing = [kw for kw in case["must_contain"] if kw.lower() not in answer.lower()]
                print(f"         Missing keywords: {missing}")
                print(f"         Answer: {answer[:200]}")

        print(f"\nResult: {passed}/{len(eval_cases)} passed")
        return {"passed": passed, "total": len(eval_cases)}
