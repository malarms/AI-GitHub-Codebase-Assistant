from qdrant_client.models import PointStruct

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_python_file
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB

from pathlib import Path


def build_index(repo_path: Path):

    embedding_model = EmbeddingModel()
    vector_db = VectorDB()

    all_chunks = []

    files = discover_files(repo_path)

    python_files = [
        file for file in files
        if file.suffix == ".py"
    ]

    for file in python_files:
        chunks = parse_python_file(file)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    texts = [
        chunk["code"]
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

    vector_db.insert(points)

    print("Index created successfully.")

    return embedding_model, vector_db