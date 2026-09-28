from pathlib import Path

from app.retrieval.bm25 import BM25Retriever


repo_path = Path("data/repos/requests")

retriever = BM25Retriever(repo_path)

query = "HTTPAdapter send"

results = retriever.search(query, limit=5)

print("\nBM25 RESULTS:\n")

for result in results:
    chunk = result["chunk"]

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
    print(f"Score: {result['score']:.4f}")
    print()