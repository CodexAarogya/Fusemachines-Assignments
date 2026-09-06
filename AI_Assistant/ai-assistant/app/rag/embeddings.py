"""
Embedding wrapper. Uses a local sentence-transformers model so the RAG
pipeline works fully offline / without incurring per-token embedding
API costs. Swap `model_name` for an OpenAI embedding model if preferred
(see the commented alternative below).
"""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name)


# --- Alternative: OpenAI embeddings ---
# from openai import OpenAI
# client = OpenAI()
# def embed_openai(texts):
#     resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
#     return [d.embedding for d in resp.data]
