from pathlib import Path

from qdrant_client.models import PointStruct

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_code_file
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB


def build_index(repo_path: Path):
    embedding_model = EmbeddingModel()
    vector_db = VectorDB()

    files = discover_files(repo_path)

    print(f"Files discovered: {len(files)}")

    all_chunks = []

    for file in files:
        chunks = parse_code_file(file)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError(
            "No code chunks were found in the repository."
        )

    texts = [
        f"""
        file {chunk['file']}
        language {chunk['language']}
        symbol {chunk['symbol']}
        parent {chunk.get('parent', '')}
        type {chunk['symbol_type']}
        code {chunk['code']}
        """
        for chunk in all_chunks
    ]

    print("Generating embeddings...")

    embeddings = embedding_model.embed(texts)

    vector_size = len(embeddings[0])

    vector_db.create_collection(vector_size)

    points = []

    for i, (chunk, embedding) in enumerate(
        zip(all_chunks, embeddings)
    ):
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload=chunk,
            )
        )

    print("Uploading vectors to Qdrant...")

    vector_db.insert(points)

    print("Index created successfully.")

    return embedding_model, vector_db
