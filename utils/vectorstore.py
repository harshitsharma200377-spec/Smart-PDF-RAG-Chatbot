from langchain_chroma import Chroma
from embeddings import get_embeddings

def create_vectorstore(chunks):
    return Chroma.from_documents(
        chunks,
        get_embeddings(),
        persist_directory="chroma_db"
    )

def load_vectorstore():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=get_embeddings()
    )
