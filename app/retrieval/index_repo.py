from pathlib import Path

from app.retrieval.indexer import build_index


repo_path = Path("data/repos/multi-language-test")

build_index(repo_path)

print("\nRepository indexed successfully.")