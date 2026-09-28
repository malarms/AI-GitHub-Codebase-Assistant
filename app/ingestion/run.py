from pathlib import Path

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_python_file


REPO_PATH = Path("data/repos/requests")

files = discover_files(REPO_PATH)

python_files = [
    file for file in files
    if file.suffix == ".py"
]

print(f"Python files: {len(python_files)}")

total_chunks = 0

for file in python_files:

    chunks = parse_python_file(file)

    total_chunks += len(chunks)

    print(f"\n{file}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks[:3]:

        print(
            f"  {chunk['symbol_type']}: "
            f"{chunk['symbol']} "
            f"({chunk['start_line']}-{chunk['end_line']})"
        )

print(f"\nTotal code chunks: {total_chunks}")