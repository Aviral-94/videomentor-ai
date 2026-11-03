import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

from app.vectorstore.embeddings import get_embeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = "videomentor_videos"


def _get_qdrant_params():
    url = os.getenv("QDRANT_URL", QDRANT_URL)
    api_key = os.getenv("QDRANT_API_KEY") or QDRANT_API_KEY
    if api_key and isinstance(api_key, str):
        api_key = api_key.strip() or None
    return url, api_key


def store_documents(documents):

    embeddings = get_embeddings()
    url, api_key = _get_qdrant_params()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=url,
        api_key=api_key,
        collection_name=COLLECTION_NAME
    )

    return vector_store


def get_vector_store():

    embeddings = get_embeddings()
    url, api_key = _get_qdrant_params()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=url,
        api_key=api_key,
        collection_name=COLLECTION_NAME
    )

    return vector_store
