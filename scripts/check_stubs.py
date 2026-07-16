#!/usr/bin/env python
"""Stub coverage check: every native class/function must appear in _native.pyi.

The native module (_native) is the source of truth; the hand-written stub file
drifts as the Rust surface grows. This script imports the built module,
enumerates public classes and functions across every submodule, and reports
any symbol name absent from _native.pyi.

It is a name-level check (not a signature check) — enough to catch "a whole
class/function was added and never stubbed", which was the actual failure mode
(stubs covered ~40% of classes). Exit 1 if coverage is below the threshold.

Usage: python scripts/check_stubs.py [--min-coverage 0.9]
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

STUB = Path(__file__).resolve().parent.parent / "python" / "flowgentra_ai" / "_native.pyi"


def native_symbols() -> set[str]:
    """Public class and top-level function names across _native and its submodules."""
    import flowgentra_ai._native as native

    names: set[str] = set()
    seen_modules: set[int] = set()

    def visit(mod) -> None:
        if id(mod) in seen_modules:
            return
        seen_modules.add(id(mod))
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr)
            if inspect.isclass(obj):
                names.add(attr)
            elif inspect.isbuiltin(obj) or inspect.isfunction(obj):
                names.add(attr)
            elif inspect.ismodule(obj):
                visit(obj)

    visit(native)
    return names


def stub_symbols() -> set[str]:
    text = STUB.read_text(encoding="utf-8")
    classes = set(re.findall(r"^class (\w+)", text, re.M))
    funcs = set(re.findall(r"^def (\w+)", text, re.M))
    # Module-level aliases like `py_model_pricing` are declared as `def` too.
    return classes | funcs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-coverage", type=float, default=0.85)
    args = parser.parse_args()

    native = native_symbols()
    stubbed = stub_symbols()

    # Ignore native functions whose stub name is the un-prefixed alias
    # (e.g. native `py_model_pricing` is exported to users as `model_pricing`).
    missing = sorted(
        n
        for n in native - stubbed
        if not (n.startswith("py_") and n[3:] in stubbed)
    )

    covered = len(native) - len(missing)
    coverage = covered / max(len(native), 1)
    print(f"Stub coverage: {covered}/{len(native)} native symbols ({coverage:.0%})")
    if missing:
        print(f"Missing from _native.pyi ({len(missing)}):")
        for name in missing:
            print(f"  {name}")

    if coverage < args.min_coverage:
        print(
            f"FAIL: coverage {coverage:.0%} < required {args.min_coverage:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
