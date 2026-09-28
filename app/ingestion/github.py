import subprocess
from pathlib import Path
from urllib.parse import urlparse

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "tests",
    "test",
}

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".md",
    ".json",
    ".yaml",
    ".yml",
}


def clone_repository(repo_url: str, destination: str) -> Path:

    parsed_url = urlparse(repo_url)

    if parsed_url.hostname != "github.com":
        raise ValueError("Only GitHub repositories are supported.")

    destination_path = Path(destination)

    if destination_path.exists() and any(destination_path.iterdir()):
        raise ValueError("Destination directory is not empty.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(destination_path)],
        check=True,
    )

    return destination_path


def discover_files(repo_path: Path) -> list[Path]:

    files = []

    for path in repo_path.rglob("*"):

        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return files