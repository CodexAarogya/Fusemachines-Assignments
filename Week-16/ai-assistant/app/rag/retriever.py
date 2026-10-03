"""
High-level retriever: embed a query and fetch the most relevant chunks.
"""
import logging
from typing import Any, Dict, List

from app.rag.embeddings import EmbeddingModel
from app.rag.ingest import ingest_file
from app.rag.vector_store import VectorStore

logger = logging.getLogger("ai_assistant.retriever")


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore, chunk_size: int, chunk_overlap: int):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_path(self, path: str) -> int:
        records = ingest_file(path, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not records:
            return 0
        ids = [r[0] for r in records]
        texts = [r[1] for r in records]
        metas = [r[2] for r in records]
        embeddings = self.embedding_model.embed(texts)
        self.vector_store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
        logger.info("Ingested %d chunks from %s", len(records), path)
        return len(records)

    def query(self, text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.vector_store.count() == 0:
            return []
        embedding = self.embedding_model.embed_one(text)
        return self.vector_store.query(embedding, top_k=top_k)

    def build_context_block(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        parts = []
        for r in results:
            parts.append(f"[Source: {r['metadata'].get('source')} | chunk {r['chunk_id']}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)
