from unittest.mock import MagicMock
from rag.retriever import get_retriever

def test_get_retriever():
    mock_vectorstore = MagicMock()
    get_retriever(mock_vectorstore, search_type="mmr", k=3)
    
    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 9}
    )
