from pathlib import Path

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CodeReranker


class CodeRetriever:

    def __init__(self, repo_path: Path):

        self.hybrid = HybridRetriever(
            repo_path
        )

        self.reranker = CodeReranker()

    def search(
        self,
        query: str,
        candidate_k: int = 10,
        final_k: int = 5,
    ):

        candidates = self.hybrid.search(
            query,
            dense_k=candidate_k,
            lexical_k=candidate_k,
            final_k=candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=final_k,
        )