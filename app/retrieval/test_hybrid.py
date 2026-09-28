from pathlib import Path

from app.retrieval.hybrid import HybridRetriever


repo_path = Path("data/repos/requests")

retriever = HybridRetriever(repo_path)

query = "HTTPAdapter send"

results = retriever.search(
    query,
    dense_k=10,
    lexical_k=10,
    final_k=5,
)

print("\nHYBRID RESULTS:\n")

for rank, result in enumerate(results, start=1):
    chunk = result["chunk"]

    print(f"Rank: {rank}")
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
    print(f"RRF Score: {result['score']:.6f}")
    print()