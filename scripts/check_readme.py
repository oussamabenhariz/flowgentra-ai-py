#!/usr/bin/env python
"""Execute every offline-runnable Python block in README.md.

README examples that don't run are a credibility killer; this script is the
CI gate that keeps them honest. Blocks are executed in order and share one
namespace, so an example may build on the block before it.

Skipped blocks (reported, not executed): anything needing network access,
API keys, external services, or files that don't exist in CI.

Exit code 0 = every executed block passed.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

# A block containing any of these markers needs external resources.
SKIP_MARKERS = (
    "api_key",
    "OPENAI",
    "client =",          # LLM client construction (network on use)
    "LLM(",
    "Agent.from_config",  # needs a config.yaml on disk
    "MCP",
    "chromadb",
    "Supervisor",
    "client)",
    "document.pdf",       # needs a real file
)


def main() -> int:
    src = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", src, re.S)
    if not blocks:
        print("ERROR: no python blocks found — README moved?", file=sys.stderr)
        return 1

    # Examples may write files (checkpoints); run in a throwaway directory.
    workdir = tempfile.mkdtemp(prefix="readme-check-")
    os.chdir(workdir)

    namespace: dict = {}
    ok = skipped = 0
    failures: list[tuple[int, str]] = []

    for i, block in enumerate(blocks):
        if any(marker in block for marker in SKIP_MARKERS):
            skipped += 1
            continue
        try:
            exec(compile(block, f"README.md block {i}", "exec"), namespace)
            ok += 1
        except Exception as e:  # noqa: BLE001 — report every failure kind
            failures.append((i, f"{type(e).__name__}: {e}"))

    print(f"README blocks: {ok} passed, {len(failures)} failed, {skipped} skipped")
    for i, msg in failures:
        print(f"  block {i}: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
