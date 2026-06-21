from langchain_community.vectorstores import Chroma
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

if __name__ == "__main__":
    from loader import load_documents
    from splitter import split_documents
    docs = load_documents()
    chunks = split_documents(docs)
    db = create_vectorstore(chunks)
    print(f"✅ Vectorstore created!")
    print(f"Total vectors saved: {db._collection.count()}")