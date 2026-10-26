"""
Static AST Code Analyzer to extract real rubric signals from student Python submissions.
"""

import ast


BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.BoolOp,
)


def extract_ast_metrics(source_code: str) -> dict[str, float]:
    """
    Parses raw Python source code and extracts cyclomatic complexity,
    docstring coverage ratio, and structural lint indicators.
    """
    tree = ast.parse(source_code)

    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    # 1. Docstring Coverage Ratio
    if not functions:
        docstring_coverage = 1.0 if ast.get_docstring(tree) else 0.0
    else:
        documented = sum(1 for fn in functions if ast.get_docstring(fn) is not None)
        docstring_coverage = round(documented / len(functions), 3)

    # 2. Average McCabe Cyclomatic Complexity per Function
    complexities = []
    for fn in functions:
        branch_count = sum(1 for node in ast.walk(fn) if isinstance(node, BRANCH_NODES))
        complexities.append(1 + branch_count)

    avg_complexity = round(float(sum(complexities) / max(len(complexities), 1)), 2)

    # 3. Structural Anti-Pattern Warnings (bare excepts, overly long functions)
    bare_excepts = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    )
    long_functions = sum(
        1 for fn in functions
        if getattr(fn, "end_lineno", fn.lineno) - fn.lineno >= 50
    )

    return {
        "code_complexity_score": avg_complexity,
        "docstring_coverage": docstring_coverage,
        "lint_warning_count": bare_excepts + long_functions,
    }