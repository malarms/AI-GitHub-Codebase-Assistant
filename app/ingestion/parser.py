from pathlib import Path

from tree_sitter import Language, Parser

import tree_sitter_python as ts_python
import tree_sitter_javascript as ts_javascript
import tree_sitter_typescript as ts_typescript
import tree_sitter_java as ts_java
import tree_sitter_c as ts_c
import tree_sitter_cpp as ts_cpp
import tree_sitter_go as ts_go
import tree_sitter_rust as ts_rust
import tree_sitter_ruby as ts_ruby


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG = {
    ".py": {
        "language": Language(ts_python.language()),
        "name": "python",
        "symbols": {
            "function_definition": "function",
            "class_definition": "class",
        },
    },
    ".js": {
        "language": Language(ts_javascript.language()),
        "name": "javascript",
        "symbols": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
        },
    },
    ".jsx": {
        "language": Language(ts_javascript.language()),
        "name": "javascript",
        "symbols": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
        },
    },
    ".ts": {
        "language": Language(ts_typescript.language_typescript()),
        "name": "typescript",
        "symbols": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
            "interface_declaration": "interface",
        },
    },
    ".tsx": {
        "language": Language(ts_typescript.language_tsx()),
        "name": "tsx",
        "symbols": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
            "interface_declaration": "interface",
        },
    },
    ".java": {
        "language": Language(ts_java.language()),
        "name": "java",
        "symbols": {
            "method_declaration": "method",
            "class_declaration": "class",
            "interface_declaration": "interface",
        },
    },
    ".c": {
        "language": Language(ts_c.language()),
        "name": "c",
        "symbols": {
            "function_definition": "function",
            "struct_specifier": "struct",
        },
    },
    ".h": {
        "language": Language(ts_c.language()),
        "name": "c",
        "symbols": {
            "function_definition": "function",
            "struct_specifier": "struct",
        },
    },
    ".cpp": {
        "language": Language(ts_cpp.language()),
        "name": "cpp",
        "symbols": {
            "function_definition": "function",
            "class_specifier": "class",
            "struct_specifier": "struct",
        },
    },
    ".hpp": {
        "language": Language(ts_cpp.language()),
        "name": "cpp",
        "symbols": {
            "function_definition": "function",
            "class_specifier": "class",
            "struct_specifier": "struct",
        },
    },
    ".go": {
        "language": Language(ts_go.language()),
        "name": "go",
        "symbols": {
            "function_declaration": "function",
            "method_declaration": "method",
            "type_declaration": "type",
        },
    },
    ".rs": {
        "language": Language(ts_rust.language()),
        "name": "rust",
        "symbols": {
            "function_item": "function",
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
            "impl_item": "impl",
        },
    },
    ".rb": {
        "language": Language(ts_ruby.language()),
        "name": "ruby",
        "symbols": {
            "method": "method",
            "singleton_method": "method",
            "class": "class",
            "module": "module",
        },
    },
}


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

# A semantic function/method/class is kept whole up to this size.
# Larger semantic chunks are split while preserving their identity.
MAX_SEMANTIC_LINES = 150

# Size used when splitting a large function/method/class.
SPLIT_LINES = 100

# Overlap between pieces of a large semantic chunk.
SPLIT_OVERLAP = 20

# Generic fallback chunks for unsupported files.
GENERIC_CHUNK_LINES = 80
GENERIC_OVERLAP = 10


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def get_language_config(file_path: Path):
    return LANGUAGE_CONFIG.get(file_path.suffix.lower())


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode(
        "utf-8",
        errors="replace",
    )


def _get_symbol_name(node, source: bytes) -> str:
    """
    Extract the name of a Tree-sitter symbol.

    Different languages use different field names, so we try several
    common possibilities.
    """

    for field_name in ("name", "declarator", "left"):
        child = node.child_by_field_name(field_name)

        if child is not None:
            text = _node_text(child, source).strip()

            if text:
                return text

    for child in node.children:
        if child.type in {
            "identifier",
            "property_identifier",
            "type_identifier",
            "field_identifier",
        }:
            return _node_text(child, source).strip()

    return node.type


def _collect_nodes(root, symbol_types):
    """
    Recursively collect all Tree-sitter nodes matching configured
    semantic symbol types.
    """

    nodes = []
    stack = [root]

    while stack:
        node = stack.pop()

        if node.type in symbol_types:
            nodes.append(node)

        stack.extend(reversed(node.children))

    return nodes


def _find_parent_symbol(node, symbol_nodes, source: bytes):
    """
    Find the closest enclosing semantic symbol.
    """

    current = node.parent

    while current is not None:
        if current in symbol_nodes:
            return _get_symbol_name(current, source)

        current = current.parent

    return None


# ---------------------------------------------------------------------------
# Semantic chunk creation
# ---------------------------------------------------------------------------

def _create_semantic_chunk(
    node,
    symbol_type,
    parent,
    language,
    file_path,
    source,
):
    return {
        "file": str(file_path),
        "language": language,
        "symbol": _get_symbol_name(node, source),
        "symbol_type": symbol_type,
        "parent": parent,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "is_stub": False,
        "code": _node_text(node, source),
    }


def _split_semantic_chunk(chunk: dict) -> list[dict]:
    """
    Split a large semantic chunk while preserving its metadata.

    Example:

        process_request()
        lines 100-350

    becomes:

        process_request [1/3] 100-199
        process_request [2/3] 180-279
        process_request [3/3] 260-350

    The chunks overlap so important logic near boundaries is not lost.
    """

    lines = chunk["code"].splitlines()

    if len(lines) <= MAX_SEMANTIC_LINES:
        return [chunk]

    chunks = []

    step = SPLIT_LINES - SPLIT_OVERLAP

    start = 0
    part_number = 1

    while start < len(lines):
        end = min(
            start + SPLIT_LINES,
            len(lines),
        )

        code = "\n".join(lines[start:end])

        start_line = chunk["start_line"] + start
        end_line = start_line + (end - start) - 1

        split_chunk = chunk.copy()

        split_chunk["symbol"] = (
            f"{chunk['symbol']} "
            f"[part {part_number}]"
        )

        split_chunk["start_line"] = start_line
        split_chunk["end_line"] = end_line
        split_chunk["code"] = code

        chunks.append(split_chunk)

        if end >= len(lines):
            break

        start += step
        part_number += 1

    return chunks


# ---------------------------------------------------------------------------
# Module-level chunks
# ---------------------------------------------------------------------------

def _create_module_chunk(
    root,
    symbol_nodes,
    file_path,
    language,
    source,
):
    """
    Capture small amounts of module-level code such as imports,
    constants and configuration.

    We intentionally do not create a huge module chunk.
    """

    symbol_types = {
        "function_definition",
        "class_definition",
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "struct_specifier",
        "class_specifier",
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "type_declaration",
        "method",
        "singleton_method",
        "class",
        "module",
    }

    top_level_nodes = []

    for node in root.children:

        if node.type in symbol_types:
            continue

        if node.type in {
            "comment",
            "line_comment",
            "block_comment",
        }:
            continue

        text = _node_text(node, source).strip()

        if text:
            top_level_nodes.append(node)

    if not top_level_nodes:
        return None

    start_line = (
        top_level_nodes[0].start_point[0] + 1
    )

    end_line = (
        top_level_nodes[-1].end_point[0] + 1
    )

    code = "\n".join(
        _node_text(node, source)
        for node in top_level_nodes
    ).strip()

    if not code:
        return None

    # Module-level code that is too large should not become one giant chunk.
    if end_line - start_line + 1 > GENERIC_CHUNK_LINES:
        return None

    return {
        "file": str(file_path),
        "language": language,
        "symbol": "__module__",
        "symbol_type": "module",
        "parent": None,
        "start_line": start_line,
        "end_line": end_line,
        "is_stub": False,
        "code": code,
    }


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

def _create_generic_chunks(
    source: bytes,
    file_path: Path,
    language: str,
    chunk_lines: int = GENERIC_CHUNK_LINES,
    overlap: int = GENERIC_OVERLAP,
):
    """
    Generic line-based fallback for files without a configured
    Tree-sitter grammar.
    """

    text = source.decode(
        "utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    if not lines:
        return []

    chunks = []

    step = max(
        1,
        chunk_lines - overlap,
    )

    start = 0
    part_number = 1

    while start < len(lines):

        end = min(
            start + chunk_lines,
            len(lines),
        )

        code = "\n".join(
            lines[start:end]
        )

        if code.strip():

            chunks.append(
                {
                    "file": str(file_path),
                    "language": language,
                    "symbol": f"chunk_{part_number}",
                    "symbol_type": "code_chunk",
                    "parent": None,
                    "start_line": start + 1,
                    "end_line": end,
                    "is_stub": False,
                    "code": code,
                }
            )

        if end >= len(lines):
            break

        start += step
        part_number += 1

    return chunks


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_code_file(file_path: Path) -> list[dict]:
    """
    Parse a source file into retrieval-friendly code chunks.

    Strategy:

    1. Use Tree-sitter for supported languages.
    2. Extract semantic symbols such as functions, methods and classes.
    3. Avoid giant class chunks when smaller semantic children exist.
    4. Split very large semantic symbols with overlap.
    5. Keep small module-level code.
    6. Fall back to generic chunks for unsupported files.
    """

    source = file_path.read_bytes()

    config = get_language_config(file_path)

    # ---------------------------------------------------------------
    # Unsupported language
    # ---------------------------------------------------------------

    if config is None:

        return _create_generic_chunks(
            source=source,
            file_path=file_path,
            language=(
                file_path.suffix.lower()
                .lstrip(".")
                or "unknown"
            ),
        )

    # ---------------------------------------------------------------
    # Tree-sitter parsing
    # ---------------------------------------------------------------

    parser = Parser(
        config["language"]
    )

    tree = parser.parse(source)

    symbol_types = set(
        config["symbols"].keys()
    )

    all_symbol_nodes = _collect_nodes(
        tree.root_node,
        symbol_types,
    )

    symbol_node_set = set(
        all_symbol_nodes
    )

    # ---------------------------------------------------------------
    # Identify useful semantic nodes
    # ---------------------------------------------------------------

    container_types = {
        "class_definition",
        "class_declaration",
        "class_specifier",
        "interface_declaration",
        "struct_specifier",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "type_declaration",
        "class",
        "module",
    }

    semantic_nodes = []

    for node in all_symbol_nodes:

        # A container containing smaller symbols should not itself
        # become a giant retrieval chunk.
        if node.type in container_types:

            has_child_symbol = False

            for other in all_symbol_nodes:

                if other == node:
                    continue

                if (
                    other.start_byte >= node.start_byte
                    and other.end_byte <= node.end_byte
                ):
                    has_child_symbol = True
                    break

            if has_child_symbol:
                continue

        semantic_nodes.append(node)

    # ---------------------------------------------------------------
    # Create semantic chunks
    # ---------------------------------------------------------------

    chunks = []

    for node in semantic_nodes:

        symbol_type = config["symbols"][
            node.type
        ]

        parent = _find_parent_symbol(
            node=node,
            symbol_nodes=symbol_node_set,
            source=source,
        )

        chunk = _create_semantic_chunk(
            node=node,
            symbol_type=symbol_type,
            parent=parent,
            language=config["name"],
            file_path=file_path,
            source=source,
        )

        # Large functions/classes/methods are split,
        # but they retain their semantic identity.
        split_chunks = _split_semantic_chunk(
            chunk
        )

        chunks.extend(
            split_chunks
        )

    # ---------------------------------------------------------------
    # Module-level code
    # ---------------------------------------------------------------

    module_chunk = _create_module_chunk(
        root=tree.root_node,
        symbol_nodes=symbol_node_set,
        file_path=file_path,
        language=config["name"],
        source=source,
    )

    if module_chunk is not None:
        chunks.insert(
            0,
            module_chunk,
        )

    # ---------------------------------------------------------------
    # Final fallback
    # ---------------------------------------------------------------

    if not chunks:

        return _create_generic_chunks(
            source=source,
            file_path=file_path,
            language=config["name"],
        )

    return chunks


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

def parse_python_file(
    file_path: Path,
) -> list[dict]:

    return parse_code_file(
        file_path
    )