from pathlib import Path

from qdrant_client.models import PointStruct

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_code_file
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_db import VectorDB


def build_index(repo_path: Path):
    embedding_model = EmbeddingModel()
    vector_db = VectorDB()

    # Discover all supported repository files
    files = discover_files(repo_path)

    print(f"Files discovered: {len(files)}")

    # Parse every file using the language-aware parser
    all_chunks = []

    for file in files:
        chunks = parse_code_file(file)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("No code chunks were found in the repository.")

    # Convert chunks into embedding text
    texts = [
        f"""
File: {chunk['file']}
Language: {chunk['language']}
Symbol: {chunk['symbol']}
Type: {chunk['symbol_type']}
Parent: {chunk.get('parent')}
Lines: {chunk['start_line']}-{chunk['end_line']}

Code:
{chunk['code']}
"""
        for chunk in all_chunks
    ]

    # Generate embeddings
    print("Generating embeddings...")

    embeddings = embedding_model.embed(texts)

    vector_size = len(embeddings[0])

    # Recreate Qdrant collection
    vector_db.create_collection(vector_size)

    # Create Qdrant points
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

    # Insert into Qdrant
    print("Uploading vectors to Qdrant...")

    vector_db.insert(points)

    print("Index created successfully.")

    return embedding_model, vector_db