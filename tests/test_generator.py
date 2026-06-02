import pytest
from langchain_core.documents import Document
from rag.generator import format_docs, build_llm

def test_format_docs():
    docs = [
        Document(page_content="Content 1", metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content="Content 2", metadata={"source": "doc2.pdf"})
    ]
    formatted = format_docs(docs)
    
    expected = "[1] Source: doc1.pdf, page 1\nContent 1\n\n[2] Source: doc2.pdf\nContent 2"
    assert formatted == expected

def test_build_llm_invalid():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm(provider="invalid_provider", model_name="test")
