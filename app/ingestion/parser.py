import ast
from pathlib import Path


def parse_python_file(file_path: Path) -> list[dict]:

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    chunks = []

    def create_chunk(node, symbol_type, parent=None):

        code = ast.get_source_segment(source, node)

        if code is None:
            return

        chunks.append(
            {
                "file": str(file_path),
                "language": "python",
                "symbol": node.name,
                "symbol_type": symbol_type,
                "parent": parent,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "code": code,
            }
        )

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            symbol_type = (
                "async_function"
                if isinstance(node, ast.AsyncFunctionDef)
                else "function"
            )

            create_chunk(node, symbol_type)

        elif isinstance(node, ast.ClassDef):

            create_chunk(node, "class")

            for child in node.body:

                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef)
                ):

                    symbol_type = (
                        "async_method"
                        if isinstance(child, ast.AsyncFunctionDef)
                        else "method"
                    )

                    create_chunk(
                        child,
                        symbol_type,
                        parent=node.name,
                    )

    return chunks