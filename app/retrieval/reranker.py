from sentence_transformers import CrossEncoder


class CodeReranker:

    def __init__(self):

        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        pairs = []

        for candidate in candidates:

            chunk = candidate["chunk"]

            text = f"""
            File: {chunk['file']}
            Symbol: {chunk['symbol']}
            Parent: {chunk.get('parent', '')}
            Type: {chunk['symbol_type']}

            {chunk['code']}
            """

            pairs.append(
                [query, text]
            )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):

            reranked.append(
                {
                    "chunk": candidate["chunk"],
                    "rrf_score": candidate["rrf_score"],
                    "rerank_score": float(score),
                }
            )

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]