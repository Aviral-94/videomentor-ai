from langchain_qdrant import QdrantVectorStore

from app.vectorstore.embeddings import get_embeddings


QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "videomentor_videos"


def store_documents(documents):

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )

    return vector_store


def get_vector_store():

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )

    return vector_store