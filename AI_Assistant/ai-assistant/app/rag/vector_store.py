"""
Vector database wrapper around ChromaDB (embedded, persisted to disk).

Chroma is used because it requires no separate server for local/dev
deployment, yet persists to a volume so data survives container
restarts (see docker-compose.yml `chroma_data` volume).
"""
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        res = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    # Convert Chroma's L2 distance to a rough 0-1 similarity score.
                    "score": 1.0 / (1.0 + dists[i]),
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()
