from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are a helpful assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information, say so clearly — don't make things up.

Context:
{context}

Question: {question}

Answer:"""
)


def build_llm(provider: str, model_name: str, api_key: str = ""):
    """Return an LLM object based on the provider config."""
    provider_lower = provider.lower()

    if provider_lower == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)  # type: ignore

    elif provider_lower == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=api_key)  # type: ignore

    elif provider_lower == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)

    elif provider_lower == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key or None)  # type: ignore

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved chunks with source metadata as headers."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        header = f"[{i}] Source: {source}" + (f", page {page}" if page != "" else "")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def get_rag_chain(llm, prompt=None):
    """Build the RAG chain. Expects input dict {question, context} where context is pre-formatted.

    The retriever is intentionally excluded from the chain so the caller retrieves once
    and reuses the same docs for both source display and generation.
    """
    if prompt is None:
        prompt = DEFAULT_RAG_PROMPT
    return prompt | llm | StrOutputParser()
