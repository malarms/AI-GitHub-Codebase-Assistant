from pathlib import Path

from qdrant_client.models import PointStruct

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_code_file
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB


def build_index(repo_path: Path):
    embedding_model = EmbeddingModel()
    vector_db = VectorDB()

    all_chunks = []

    files = discover_files(repo_path)

    print(f"Files discovered: {len(files)}")

    for file in files:
        try:
            chunks = parse_code_file(file)
            all_chunks.extend(chunks)

            print(
                f"{file} -> "
                f"{len(chunks)} chunks"
            )

        except Exception as e:
            print(
                f"Skipping {file}: {e}"
            )

    print(f"\nTotal code chunks: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("No code chunks were generated.")

    texts = [
        f"""
File: {chunk['file']}
Language: {chunk['language']}
Symbol: {chunk['symbol']}
Type: {chunk['symbol_type']}
Parent: {chunk.get('parent') or 'None'}
Lines: {chunk['start_line']}-{chunk['end_line']}

Code:
{chunk['code']}
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

    vector_db.insert(points)

    print("\nIndex created successfully.")

    return embedding_model, vector_db