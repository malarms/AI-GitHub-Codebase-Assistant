from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB


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
    top_results = results[:K]

    recall = int(
        any(
            is_relevant(result, relevant_files)
            for result in top_results
        )
    )

    reciprocal_rank = 0

    for rank, result in enumerate(
        top_results,
        start=1,
    ):
        if is_relevant(result, relevant_files):
            reciprocal_rank = 1 / rank
            break

    return recall, reciprocal_rank


def main():

    print("Loading models...")

    embedding_model = EmbeddingModel()
    vector_db = VectorDB()

    scores = []

    for i, item in enumerate(
        BENCHMARK,
        start=1,
    ):

        question = item["question"]
        relevant_files = item["files"]

        print(
            f"\n[{i}/{len(BENCHMARK)}] "
            f"{question}"
        )

        query_vector = embedding_model.embed(
            [question]
        )[0]

        results = vector_db.search(
            query_vector,
            limit=K,
        )

        results = [
            result.payload
            for result in results
        ]

        scores.append(
            evaluate_results(
                results,
                relevant_files,
            )
        )

    recall = sum(
        score[0]
        for score in scores
    ) / len(scores)

    mrr = sum(
        score[1]
        for score in scores
    ) / len(scores)

    print("\n" + "=" * 60)
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        f"Recall@5: {recall:.3f}   "
        f"MRR: {mrr:.3f}"
    )


if __name__ == "__main__":
    main()
