"""Extract structural info from Python code before sending to the LLM.

Reduces hallucinations by giving the specialist a concrete map of imports,
functions, classes, and undefined names — instead of the LLM having to guess.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

log = logging.getLogger("mlfix.ast")


@dataclass
class CodeStructure:
    imports: list[str] = field(default_factory=list)         # ["torch", "numpy as np"]
    functions: list[str] = field(default_factory=list)       # ["def train(model, data)"]
    classes: list[str] = field(default_factory=list)         # ["class MyModel(nn.Module)"]
    defined_names: set[str] = field(default_factory=set)     # all names assigned/defined
    used_names: set[str] = field(default_factory=set)        # all names referenced
    undefined_names: set[str] = field(default_factory=set)   # used but not defined/imported
    syntax_valid: bool = True
    syntax_error: str | None = None

    def to_prompt_summary(self) -> str:
        """Format as a compact block for injection into the specialist prompt."""
        if not self.syntax_valid:
            return f"[AST ANALYSIS] Syntax error prevents parsing: {self.syntax_error}"

        lines = ["[AST ANALYSIS]"]
        if self.imports:
            lines.append(f"Imports ({len(self.imports)}): {', '.join(self.imports[:10])}")
        if self.classes:
            lines.append(f"Classes: {', '.join(self.classes[:8])}")
        if self.functions:
            lines.append(f"Functions: {', '.join(self.functions[:10])}")
        if self.undefined_names:
            names = sorted(self.undefined_names)[:8]
            lines.append(f"Undefined references (possible bug source): {', '.join(names)}")
        return "\n".join(lines)


# Common Python builtins we shouldn't flag as undefined
_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
_COMMON_NAMES = {"self", "cls", "__name__", "__file__", "__doc__", "True", "False", "None"}


class _StructureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[str] = []
        self.functions: list[str] = []
        self.classes: list[str] = []
        self.defined_names: set[str] = set()
        self.used_names: set[str] = set()
        self.imported_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            self.imports.append(name)
            self.imported_names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.append(f"from {mod} import {alias.name}")
            self.imported_names.add(name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        args = [a.arg for a in node.args.args]
        self.functions.append(f"{node.name}({', '.join(args)})")
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        args = [a.arg for a in node.args.args]
        self.functions.append(f"async {node.name}({', '.join(args)})")
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(b) for b in node.bases] if node.bases else []
        signature = f"{node.name}({', '.join(bases)})" if bases else node.name
        self.classes.append(signature)
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                self.defined_names.add(tgt.id)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.defined_names.add(node.id)


def analyze(code: str) -> CodeStructure:
    """Parse code and extract structural info. Never raises."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        log.info("ast parse failed (syntax): %s", e.msg)
        return CodeStructure(syntax_valid=False, syntax_error=str(e))
    except Exception as e:
        log.warning("ast parse failed (other): %s", e)
        return CodeStructure(syntax_valid=False, syntax_error=str(e))

    v = _StructureVisitor()
    v.visit(tree)

    all_defined = v.defined_names | v.imported_names | _COMMON_NAMES
    try:
        builtins = set(dir(__builtins__)) if not isinstance(__builtins__, dict) else set(__builtins__.keys())
    except Exception:
        builtins = set()

    undefined = {n for n in v.used_names if n not in all_defined and n not in builtins}

    return CodeStructure(
        imports=v.imports,
        functions=v.functions,
        classes=v.classes,
        defined_names=v.defined_names,
        used_names=v.used_names,
        undefined_names=undefined,
        syntax_valid=True,
    )