import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_code_file


class BM25Retriever:
    def __init__(self, repo_path: Path):
        self.chunks = []

        files = discover_files(repo_path)

        for file in files:
            self.chunks.extend(parse_code_file(file))

        print(f"BM25 chunks: {len(self.chunks)}")

        documents = [
            self._chunk_to_text(chunk)
            for chunk in self.chunks
        ]

        tokenized_documents = [
            self._tokenize(document)
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def _chunk_to_text(self, chunk: dict) -> str:
        return f"""
        file {chunk['file']}
        language {chunk['language']}
        symbol {chunk['symbol']}
        parent {chunk.get('parent', '')}
        type {chunk['symbol_type']}
        code {chunk['code']}
        """

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()

        return re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            text,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:limit]

        return [
            {
                "chunk": self.chunks[i],
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ]