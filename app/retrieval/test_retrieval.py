from pathlib import Path

from app.retrieval.indexer import build_index


repo_path = Path("data/repos/requests")

embedding_model, vector_db = build_index(repo_path)


query = "How does Requests make a GET request?"

query_vector = embedding_model.embed([query])[0]

results = vector_db.search(
    query_vector,
    limit=5
)

print("\nTop results:\n")

for result in results:

    payload = result.payload

    print(
        f"{payload['symbol_type']} "
        f"{payload['symbol']}"
    )

    print(
        f"File: {payload['file']}"
    )

    print(
        f"Lines: "
        f"{payload['start_line']}-"
        f"{payload['end_line']}"
    )

    print(
        f"Score: {result.score:.4f}"
    )

    print()