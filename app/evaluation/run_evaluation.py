from pathlib import Path

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever


REPO_PATH = Path("data/repos/multi-language-test")
K = 5


BENCHMARK = [
    {
        "question": "How does the web app execute a sandboxed program?",
        "files": [
            "apps/web/lib/executions.ts",
            "apps/web/app/api/executions/route.ts",
            "apps/web/scripts/sandbox-runner.py",
        ],
    },
    {
        "question": "Where is the POST API endpoint for executions implemented?",
        "files": [
            "apps/web/app/api/executions/route.ts",
        ],
    },
    {
        "question": "How does the server check whether a request comes from the same origin?",
        "files": [
            "apps/web/lib/server.ts",
        ],
    },
    {
        "question": "How is a Java snapshot created?",
        "files": [
            "languages/java/assessments/03-advanced/solutions/Main.java",
            "languages/java/projects/03-advanced/solutions/Main.java",
            "languages/java/03-advanced/copy-and-move-semantics/example/Main.java",
        ],
    },
    {
        "question": "How does the Python sandbox runner start a process?",
        "files": [
            "apps/web/scripts/sandbox-runner.py",
        ],
    },
    {
        "question": "How does the application make an HTTP request?",
        "files": [
            "apps/web/app/api/executions/route.ts",
            "apps/web/app/auth/callback/route.ts",
            "apps/web/app/api/solution/route.ts",
        ],
    },
    {
        "question": "Where is the POST handler defined?",
        "files": [
            "apps/web/app/api/executions/route.ts",
        ],
    },
    {
        "question": "How does the application validate incoming execution requests?",
        "files": [
            "apps/web/app/api/executions/route.ts",
            "apps/web/app/api/executions/[id]/route.ts",
            "apps/web/components/practice.tsx",
        ],
    },
    {
        "question": "Where are execution results returned to the client?",
        "files": [
            "apps/web/lib/executions.ts",
            "apps/web/components/practice.tsx",
            "apps/web/app/api/executions/route.ts",
        ],
    },
]


def normalize(path):
    return str(path).replace("\\", "/")


def is_relevant(result, relevant_files):
    file_path = normalize(result["file"])

    return any(
        file_path.endswith(file)
        for file in relevant_files
    )


def evaluate_results(results, relevant_files):
    found = any(
        is_relevant(result, relevant_files)
        for result in results[:K]
    )

    recall = 1 if found else 0

    reciprocal_rank = 0

    for rank, result in enumerate(results[:K], start=1):
        if is_relevant(result, relevant_files):
            reciprocal_rank = 1 / rank
            break

    return recall, reciprocal_rank


def main():

    print("Loading models...")

    embedding_model = EmbeddingModel()
    vector_db = VectorDB()

    print("Building BM25...")
    bm25 = BM25Retriever(REPO_PATH)

    hybrid = HybridRetriever(
    REPO_PATH,
    vector_db=vector_db,
    embedding_model=embedding_model,
    )

    methods = {
        "Dense": [],
        "BM25": [],
        "Hybrid": [],
    }

    for i, item in enumerate(BENCHMARK, start=1):

        question = item["question"]
        relevant_files = item["files"]

        print(f"\n[{i}/{len(BENCHMARK)}] {question}")

        query_vector = embedding_model.embed(
            [question]
        )[0]

        dense_results = vector_db.search(
            query_vector,
            limit=K,
        )

        dense_results = [
            result.payload
            for result in dense_results
        ]

        recall, mrr = evaluate_results(
            dense_results,
            relevant_files,
        )

        methods["Dense"].append((recall, mrr))

        bm25_results = [
            result["chunk"]
            for result in bm25.search(
                question,
                limit=K,
            )
        ]

        recall, mrr = evaluate_results(
            bm25_results,
            relevant_files,
        )

        methods["BM25"].append((recall, mrr))

        hybrid_results = hybrid.search(
            question,
            dense_k=10,
            lexical_k=10,
            final_k=K,
        )

        hybrid_results = [
            result["chunk"]
            for result in hybrid_results
        ]

        recall, mrr = evaluate_results(
            hybrid_results,
            relevant_files,
        )

        methods["Hybrid"].append((recall, mrr))

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for method, scores in methods.items():

        recall = sum(
            score[0] for score in scores
        ) / len(scores)

        mrr = sum(
            score[1] for score in scores
        ) / len(scores)

        print(
            f"{method:<10}"
            f"Recall@5: {recall:.3f}   "
            f"MRR: {mrr:.3f}"
        )


if __name__ == "__main__":
    main()
