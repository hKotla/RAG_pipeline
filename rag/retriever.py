def get_retriever(vectorstore, search_type: str = "mmr", k: int = 5):
    """Return a retriever object from the vectorstore.
    
    Uses MMR (Maximal Marginal Relevance) by default to balance relevance with diversity.
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
