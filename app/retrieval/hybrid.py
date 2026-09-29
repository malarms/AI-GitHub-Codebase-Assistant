from pathlib import Path

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB
from app.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        repo_path: Path,
        vector_db=None,
        embedding_model=None,
    ):
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

        self.vector_db = (
            vector_db
            if vector_db is not None
            else VectorDB()
        )

        self.bm25 = BM25Retriever(repo_path)

    def search(
        self,
        query: str,
        dense_k: int = 10,
        lexical_k: int = 10,
        final_k: int = 10,
        rrf_k: int = 60,
    ):

        query_vector = self.embedding_model.embed([query])[0]

        dense_results = self.vector_db.search(
            query_vector,
            limit=dense_k,
        )

        lexical_results = self.bm25.search(
            query,
            limit=lexical_k,
        )

        scores = {}
        chunks = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk = result.payload
            key = self._chunk_key(chunk)

            chunks[key] = chunk

            scores[key] = (
                scores.get(key, 0)
                + 1 / (rrf_k + rank)
            )

        for rank, result in enumerate(
            lexical_results,
            start=1,
        ):
            chunk = result["chunk"]
            key = self._chunk_key(chunk)

            chunks[key] = chunk

            scores[key] = (
                scores.get(key, 0)
                + 1 / (rrf_k + rank)
            )

        ranked_keys = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return [
            {
                "chunk": chunks[key],
                "rrf_score": scores[key],
            }
            for key in ranked_keys[:final_k]
        ]

    def _chunk_key(self, chunk):
        return (
            chunk["file"],
            chunk["start_line"],
            chunk["end_line"],
        )
