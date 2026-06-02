from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=api_key,  # type: ignore
        )

    elif provider_lower == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,  # type: ignore
            temperature=0,
            anthropic_api_key=api_key,  # type: ignore
        )

    elif provider_lower == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved chunks with source metadata as headers."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        header = f"[{i}] Source: {source}" + (f", page {page}" if page != "" else "")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)

def get_rag_chain(retriever, llm, prompt=None):
    """Build and return the LangChain RAG pipeline chain using LCEL."""
    if prompt is None:
        prompt = DEFAULT_RAG_PROMPT
    
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"])))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
