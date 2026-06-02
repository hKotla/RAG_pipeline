from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def load_documents(
    pdf_paths: Optional[List[str]] = None,
    docx_paths: Optional[List[str]] = None,
    text_paths: Optional[List[str]] = None,
    web_urls: Optional[List[str]] = None,
) -> List[Document]:
    """Load documents from all configured sources and return LangChain Documents."""
    docs: List[Document] = []

    # PDFs
    if pdf_paths:
        from langchain_community.document_loaders import PyPDFLoader
        for path in pdf_paths:
            print(f"  Loading PDF: {path}")
            pdf_loader = PyPDFLoader(path)
            docs.extend(pdf_loader.load())

    # DOCX
    if docx_paths:
        from langchain_community.document_loaders import Docx2txtLoader
        for path in docx_paths:
            print(f"  Loading DOCX: {path}")
            docx_loader = Docx2txtLoader(path)
            docs.extend(docx_loader.load())

    # Text / Markdown
    if text_paths:
        from langchain_community.document_loaders import TextLoader
        for path in text_paths:
            print(f"  Loading text: {path}")
            txt_loader = TextLoader(path, encoding="utf-8")
            docs.extend(txt_loader.load())

    # Web URLs
    if web_urls:
        from langchain_community.document_loaders import WebBaseLoader
        print(f"  Loading {len(web_urls)} web URL(s)...")
        web_loader = WebBaseLoader(web_urls)
        docs.extend(web_loader.load())

    return docs

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)

def build_embeddings(provider: str, model_name: str, api_key: str = ""):
    """Return an embeddings object based on the provider config."""
    provider_lower = provider.lower()
    if provider_lower == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,  # type: ignore
        )

    elif provider_lower == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

    elif provider_lower == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model_name)

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")

def get_vectorstore(
    persist_dir: str,
    collection_name: str,
    embeddings,
) -> Chroma:
    """Initialize or load the Chroma vectorstore."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
