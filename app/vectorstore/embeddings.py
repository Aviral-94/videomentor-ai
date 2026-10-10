from langchain_huggingface import HuggingFaceEmbeddings


_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print("Loading embedding model...")

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings