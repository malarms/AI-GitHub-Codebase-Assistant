from pathlib import Path

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB
from app.generation.llm import LLM


SYSTEM_PROMPT = """
You are a codebase assistant.

Answer the user's question using only the repository context provided.

Rules:
- Do not invent code or repository behavior.
- If the context is insufficient, say that you do not have enough information.
- Explain the code clearly and concisely.
- Cite the relevant sources using [1], [2], etc.
- Only use source numbers that actually appear in the provided context.
"""


class RAGPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDB()
        self.llm = LLM()

    def ask(self, question: str, top_k: int = 5) -> str:

        query_vector = self.embedding_model.embed([question])[0]

        results = self.vector_db.search(
            query_vector,
            limit=top_k,
        )

        context_parts = []

        for i, result in enumerate(results, start=1):
            chunk = result.payload

            context_parts.append(
                f"""
SOURCE [{i}]
File: {chunk['file']}
Symbol: {chunk['symbol']}
Type: {chunk['symbol_type']}
Lines: {chunk['start_line']}-{chunk['end_line']}

Code:
{chunk['code']}
"""
            )

        context = "\n".join(context_parts)

        user_prompt = f"""
Repository context:

{context}

User question:
{question}

Answer the question using the repository context above.
Include source citations such as [1] or [2].
"""

        return self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )