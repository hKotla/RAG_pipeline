def get_retriever(vectorstore, search_type: str = "mmr", k: int = 5):
    """Return a retriever from the vectorstore.

    Uses MMR (Maximal Marginal Relevance) by default to balance relevance with diversity.
    fetch_k is only passed for MMR — it is not a valid kwarg for similarity search.
    """
    search_kwargs: dict = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = k * 3
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
