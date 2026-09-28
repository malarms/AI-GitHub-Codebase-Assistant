from pathlib import Path

from app.retrieval.retriever import CodeRetriever


repo_path = Path(
    "data/repos/requests"
)

retriever = CodeRetriever(
    repo_path
)

query = "HTTPAdapter send"

results = retriever.search(
    query,
    candidate_k=10,
    final_k=5,
)

print("\nRERANKED RESULTS:\n")

for rank, result in enumerate(
    results,
    start=1,
):

    chunk = result["chunk"]

    print(
        f"Rank: {rank}"
    )

    print(
        f"{chunk['symbol_type']} "
        f"{chunk['symbol']}"
    )

    print(
        f"Parent: "
        f"{chunk.get('parent', '')}"
    )

    print(
        f"File: {chunk['file']}"
    )

    print(
        f"Lines: "
        f"{chunk['start_line']}-"
        f"{chunk['end_line']}"
    )

    print(
        f"RRF: "
        f"{result['rrf_score']:.6f}"
    )

    print(
        f"Rerank: "
        f"{result['rerank_score']:.4f}"
    )

    print()