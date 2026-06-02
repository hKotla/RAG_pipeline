import pytest
from langchain_core.documents import Document
from rag.ingest import chunk_documents, load_documents, build_embeddings

def test_chunk_documents():
    docs = [
        Document(page_content="This is the first sentence. This is the second sentence. This is the third sentence.")
    ]
    # Small chunk size to force splitting
    chunks = chunk_documents(docs, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)

def test_load_documents_empty():
    docs = load_documents()
    assert docs == []

def test_build_embeddings_invalid():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        build_embeddings(provider="invalid_provider", model_name="test")
