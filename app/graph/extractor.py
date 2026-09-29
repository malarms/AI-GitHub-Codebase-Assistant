from pathlib import Path
import re
from collections import defaultdict

from tree_sitter import Parser

from app.ingestion.github import discover_files
from app.ingestion.parser import (
    get_language_config,
    _node_text,
    _get_symbol_name,
    _collect_nodes,
)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def _walk_tree(node):
    stack = [node]

    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


# ---------------------------------------------------------------------------
# Symbol names
# ---------------------------------------------------------------------------

def _get_graph_symbol_name(node, source):
    """Get a clean symbol name without modifying the existing parser."""

    for field_name in ("name", "declarator", "left"):
        child = node.child_by_field_name(field_name)

        if child is not None:
            text = _node_text(child, source).strip()

            if text:
                return text

    # Go:
    # type_declaration -> type_spec -> type_identifier
    if node.type == "type_declaration":
        for child in node.children:
            if child.type != "type_spec":
                continue

            for grandchild in child.children:
                if grandchild.type == "type_identifier":
                    return _node_text(
                        grandchild,
                        source,
                    ).strip()

    return _get_symbol_name(node, source)


def _symbol_id(file_path, node):
    return (
        f"{file_path}:"
        f"{node.start_point[0] + 1}:"
        f"{node.end_point[0] + 1}"
    )


def _find_graph_parent(node, symbol_nodes, source):
    """Find the closest enclosing semantic symbol."""

    current = node.parent

    while current is not None:
        if current in symbol_nodes:
            return _get_graph_symbol_name(
                current,
                source,
            )

        current = current.parent

    return None


def _extract_symbols(tree, source, config, file_path):
    symbol_types = set(config["symbols"].keys())

    nodes = _collect_nodes(
        tree.root_node,
        symbol_types,
    )

    symbol_node_set = set(nodes)
    symbols = []

    for node in nodes:
        symbol = _get_graph_symbol_name(
            node,
            source,
        )

        parent = _find_graph_parent(
            node,
            symbol_node_set,
            source,
        )

        qualified_name = (
            f"{parent}.{symbol}"
            if parent
            else symbol
        )

        symbols.append({
            "id": _symbol_id(file_path, node),
            "type": "symbol",
            "symbol": symbol,
            "qualified_name": qualified_name,
            "symbol_type": config["symbols"][node.type],
            "parent": parent,
            "file": str(file_path),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        })

    return symbols


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------

def _extract_contains_edges(symbols):
    return [
        {
            "source": symbol["file"],
            "target": symbol["id"],
            "type": "contains",
        }
        for symbol in symbols
    ]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

IMPORT_NODE_TYPES = {
    "import_statement",
    "import_declaration",
    "import_clause",
    "preproc_include",
    "use_declaration",
}


def _extract_import_names(text, language):
    """Turn an import statement into individual module names."""

    modules = []

    if language == "python":
        match = re.search(
            r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b",
            text,
        )

        if match:
            modules.append(match.group(1))
        else:
            match = re.search(
                r"^\s*import\s+(.+)",
                text,
            )

            if match:
                for item in match.group(1).split(","):
                    item = item.strip()
                    item = re.split(r"\s+as\s+", item)[0]
                    modules.append(item)

    elif language in {"javascript", "typescript", "tsx"}:
        modules.extend(
            re.findall(
                r'\bfrom\s+[\'"]([^\'"]+)[\'"]',
                text,
            )
        )

        modules.extend(
            re.findall(
                r'\bimport\s*[\'"]([^\'"]+)[\'"]',
                text,
            )
        )

        modules.extend(
            re.findall(
                r'\brequire\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
                text,
            )
        )

    elif language == "java":
        modules.extend(
            re.findall(
                r"^\s*import\s+(?:static\s+)?([\w.]+)",
                text,
                re.MULTILINE,
            )
        )

    elif language == "go":
        modules.extend(
            re.findall(
                r'"([^"]+)"',
                text,
            )
        )

    elif language in {"c", "cpp"}:
        modules.extend(
            re.findall(
                r"#include\s*[<\"]([^>\"]+)[>\"]",
                text,
            )
        )

    elif language == "rust":
        match = re.search(
            r"^\s*use\s+([^;]+)",
            text,
            re.MULTILINE,
        )

        if match:
            modules.append(match.group(1).strip())

    elif language == "ruby":
        modules.extend(
            re.findall(
                r"\brequire(?:_relative)?\s+[\'\"]([^\'\"]+)[\'\"]",
                text,
            )
        )

    else:
        modules.extend(
            re.findall(
                r'[\'"]([^\'"]+)[\'"]',
                text,
            )
        )

    # Remove duplicates while preserving order.
    result = []
    seen = set()

    for module in modules:
        module = module.strip()

        if module and module not in seen:
            seen.add(module)
            result.append(module)

    return result


def _extract_import_edges(
    tree,
    source,
    file_path,
    language,
):
    edges = []

    for node in _walk_tree(tree.root_node):
        if node.type not in IMPORT_NODE_TYPES:
            continue

        text = _node_text(
            node,
            source,
        )

        modules = _extract_import_names(
            text,
            language,
        )

        for module in modules:
            edges.append({
                "source": str(file_path),
                "target": module,
                "type": "imports",
                "language": language,
            })

    return edges


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

CALL_NODE_TYPES = {
    "call",
    "call_expression",
    "method_call_expression",
    "function_call_expression",
    "method_invocation",
}

CALL_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:\.|::|->)[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _get_call_target_node(node):
    """Get the AST node representing the called function/method."""

    for field_name in (
        "function",
        "name",
    ):
        child = node.child_by_field_name(
            field_name
        )

        if child is not None:
            return child

    return None


def _get_enclosing_caller(
    node,
    symbols,
    file_path,
):
    """Find the smallest named symbol containing this call."""

    line = node.start_point[0] + 1

    candidates = [
        symbol
        for symbol in symbols
        if symbol["file"] == str(file_path)
        and symbol["start_line"] <= line <= symbol["end_line"]
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda symbol: (
            symbol["end_line"]
            - symbol["start_line"]
        ),
    )


def _build_symbol_indexes(symbols):
    """Build lookup indexes for repository-level call resolution."""

    by_name = defaultdict(list)
    by_qualified_name = defaultdict(list)
    by_file = defaultdict(list)

    for symbol in symbols:
        by_name[symbol["symbol"]].append(symbol)
        by_qualified_name[
            symbol["qualified_name"]
        ].append(symbol)
        by_file[symbol["file"]].append(symbol)

    return {
        "by_name": by_name,
        "by_qualified_name": by_qualified_name,
        "by_file": by_file,
    }


def _resolve_call(
    called_name,
    caller,
    indexes,
):
    """
    Resolve a call conservatively.

    Resolution order:
    1. Exact qualified name in the same file
    2. Exact symbol name in the same file
    3. Exact qualified-name match across the repository

    We deliberately do NOT resolve by bare symbol name across
    unrelated files because that creates false positives such as
    resolving an external call like ``fmt.Print`` to an unrelated
    repository function named ``Print``.
    """

    same_file = indexes["by_file"].get(
        caller["file"],
        [],
    )

    # Exact qualified name in the same file.
    matches = [
        symbol
        for symbol in same_file
        if symbol["qualified_name"] == called_name
    ]

    if len(matches) == 1:
        return matches[0]

    # Exact symbol name in the same file.
    matches = [
        symbol
        for symbol in same_file
        if symbol["symbol"] == called_name
    ]

    if len(matches) == 1:
        return matches[0]

    # Exact qualified name across the repository.
    matches = indexes["by_qualified_name"].get(
        called_name,
        [],
    )

    if len(matches) == 1:
        return matches[0]

    return None

def _extract_call_edges(
    tree,
    source,
    file_path,
    symbols,
    indexes,
):
    edges = []

    for node in _walk_tree(tree.root_node):

        if node.type not in CALL_NODE_TYPES:
            continue

        target_node = _get_call_target_node(node)

        if target_node is None:
            continue

        called_name = _node_text(
            target_node,
            source,
        ).strip()

        # Reject anonymous functions, lambdas, nested expressions,
        # and anything that cannot be represented as a named call.
        if not CALL_TARGET_PATTERN.fullmatch(
            called_name
        ):
            continue

        caller = _get_enclosing_caller(
            node,
            symbols,
            file_path,
        )

        if caller is None:
            continue

        resolved_symbol = _resolve_call(
            called_name,
            caller,
            indexes,
        )

        edge = {
            "source": caller["id"],
            "target": (
                resolved_symbol["id"]
                if resolved_symbol is not None
                else called_name
            ),
            "target_name": called_name,
            "type": "calls",
            "file": str(file_path),
            "line": node.start_point[0] + 1,
            "resolved": resolved_symbol is not None,
        }

        edges.append(edge)

    return edges


# ---------------------------------------------------------------------------
# File graph
# ---------------------------------------------------------------------------

def extract_file_graph(
    file_path: Path,
):
    """Extract graph information from one source file."""

    config = get_language_config(
        file_path
    )

    if config is None:
        return {
            "nodes": [],
            "edges": [],
        }

    source = file_path.read_bytes()

    parser = Parser(
        config["language"]
    )

    tree = parser.parse(
        source
    )

    symbols = _extract_symbols(
        tree,
        source,
        config,
        file_path,
    )

    # File-local indexes are enough for extraction.
    indexes = _build_symbol_indexes(
        symbols
    )

    edges = []

    edges.extend(
        _extract_contains_edges(
            symbols
        )
    )

    edges.extend(
        _extract_import_edges(
            tree,
            source,
            file_path,
            config["name"],
        )
    )

    edges.extend(
        _extract_call_edges(
            tree,
            source,
            file_path,
            symbols,
            indexes,
        )
    )

    return {
        "nodes": symbols,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Repository graph
# ---------------------------------------------------------------------------

def build_code_graph(repo_path: Path):
    """
    Build the complete repository graph.

    Calls are resolved after all repository symbols are known.
    """

    files = discover_files(
        repo_path
    )

    all_nodes = []
    file_graphs = []

    # First pass: parse every file.
    for file_path in files:

        graph = extract_file_graph(
            file_path
        )

        all_nodes.extend(
            graph["nodes"]
        )

        file_graphs.append(
            (file_path, graph)
        )

    # Build repository-wide symbol indexes.
    indexes = _build_symbol_indexes(
        all_nodes
    )

    all_edges = []

    for file_path, graph in file_graphs:

        all_edges.extend(
            edge
            for edge in graph["edges"]
            if edge["type"] != "calls"
        )

        # Re-parse calls with repository-wide indexes.
        config = get_language_config(
            file_path
        )

        if config is None:
            continue

        source = file_path.read_bytes()

        parser = Parser(
            config["language"]
        )

        tree = parser.parse(
            source
        )

        file_symbols = [
            symbol
            for symbol in all_nodes
            if symbol["file"] == str(file_path)
        ]

        all_edges.extend(
            _extract_call_edges(
                tree,
                source,
                file_path,
                file_symbols,
                indexes,
            )
        )

    return {
        "nodes": all_nodes,
        "edges": all_edges,
    }
