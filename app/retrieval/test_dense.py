from pathlib import Path

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB


embedding_model = EmbeddingModel()
vector_db = VectorDB()

query = "HTTPAdapter send"

query_vector = embedding_model.embed([query])[0]

results = vector_db.search(
    query_vector,
    limit=5,
)

print("\nDENSE RESULTS:\n")

for result in results:
    chunk = result.payload

    print(
        f"{chunk['symbol_type']} "
        f"{chunk['symbol']}"
    )

    print(f"File: {chunk['file']}")
    print(
        f"Lines: "
        f"{chunk['start_line']}-"
        f"{chunk['end_line']}"
    )
    print(f"Score: {result.score:.4f}")
    print()