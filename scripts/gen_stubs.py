#!/usr/bin/env python
"""Append minimal stubs for native symbols missing from _native.pyi.

pyo3 classes expose method *names* but not typed signatures, so generated
stubs are name-level: `class X:` with `def m(self, *a, **k) -> Any: ...` per
public method, and `def f(*a, **k) -> Any: ...` for functions. This is enough
for imports to type-check and for the coverage gate to pass; hand-written,
fully-typed stubs in the curated section above always take precedence and are
never overwritten (this only appends what's absent).

Run after adding native symbols: python scripts/gen_stubs.py
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

STUB = Path(__file__).resolve().parent.parent / "python" / "flowgentra_ai" / "_native.pyi"
MARKER = "# ─── Auto-generated stubs (regenerate with scripts/gen_stubs.py) ───"


def native_members():
    import flowgentra_ai._native as native

    classes: dict[str, list[str]] = {}
    funcs: set[str] = set()
    seen: set[int] = set()

    def visit(mod):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr)
            if inspect.isclass(obj):
                methods = [
                    m
                    for m in dir(obj)
                    if not m.startswith("_") and callable(getattr(obj, m, None))
                ]
                classes[attr] = methods
            elif inspect.isbuiltin(obj) or inspect.isfunction(obj):
                funcs.add(attr)
            elif inspect.ismodule(obj):
                visit(obj)

    visit(native)
    return classes, funcs


def curated_symbols(text: str) -> set[str]:
    body = text.split(MARKER)[0]
    classes = set(re.findall(r"^class (\w+)", body, re.M))
    funcs = set(re.findall(r"^def (\w+)", body, re.M))
    return classes | funcs


def main() -> None:
    text = STUB.read_text(encoding="utf-8")
    curated = text.split(MARKER)[0].rstrip() + "\n"
    have = curated_symbols(text)

    classes, funcs = native_members()

    lines: list[str] = [MARKER, ""]
    for name in sorted(classes):
        if name in have:
            continue
        methods = classes[name]
        lines.append(f"class {name}:")
        if not methods:
            lines.append("    ...")
        else:
            for m in methods:
                lines.append(f"    def {m}(self, *args: Any, **kwargs: Any) -> Any: ...")
        lines.append("")

    for name in sorted(funcs):
        if name in have:
            continue
        # Skip functions whose un-prefixed alias is already curated.
        if name.startswith("py_") and name[3:] in have:
            continue
        lines.append(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")

    STUB.write_text(curated + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    added = sum(1 for n in classes if n not in have) + sum(
        1 for n in funcs if n not in have and not (n.startswith("py_") and n[3:] in have)
    )
    print(f"Wrote {added} generated stubs to {STUB.name}")


if __name__ == "__main__":
    main()
