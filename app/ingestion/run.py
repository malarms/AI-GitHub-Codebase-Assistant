from pathlib import Path

from app.ingestion.github import discover_files
from app.ingestion.parser import parse_code_file


REPO_PATH = Path("data/repos/requests")


files = discover_files(REPO_PATH)

print(f"Files discovered: {len(files)}")

total_chunks = 0

for file in files:
    try:
        chunks = parse_code_file(file)

        total_chunks += len(chunks)

        print(
            f"{file} -> "
            f"{len(chunks)} chunks"
        )

        for chunk in chunks[:3]:
            print(
                f"  {chunk['language']} | "
                f"{chunk['symbol_type']} | "
                f"{chunk['symbol']} | "
                f"lines {chunk['start_line']}-{chunk['end_line']}"
            )

    except Exception as e:
        print(f"  ERROR: {e}")


print(f"\nTotal code chunks: {total_chunks}")