"""Select a frozen measurement profile consistently for CLI and MCP."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    profile = os.environ.get("PLOT_READER_PROFILE", "full")
    if args[:1] == ["--profile"]:
        if len(args) < 2:
            raise SystemExit("--profile requires fast or full")
        profile, args = args[1], args[2:]
    if profile == "experimental":
        sys.argv = [sys.argv[0], *args]
        importlib.import_module("plot_reader.complete_cli").main()
        return
    if profile not in {"fast", "full"}:
        raise SystemExit("profile must be fast, full or experimental")
    package = "plot_reader.fast" if profile == "fast" else "plot_reader"
    engine = importlib.import_module(package + ".core")
    # stdio MCP clients may filter TMPDIR. Use the same session directory in
    # both transports without changing the frozen numerical implementation.
    engine.SESSION_ROOT = Path(os.environ.get(
        "PLOT_READER_SESSION_ROOT",
        "/tmp/plot-reader-sessions" if os.name == "posix"
        else str(Path(tempfile.gettempdir()) / "plot-reader-sessions"),
    )).resolve()
    if args == ["version"]:
        print(json.dumps({"profile": profile, "implementation_id": engine.implementation_id(),
                          "engine_path": engine.__file__, "session_root": str(engine.SESSION_ROOT)}))
        return
    if args == ["mcp"]:
        importlib.import_module(package + ".server").main()
        return
    sys.argv = [sys.argv[0], *args]
    importlib.import_module(package + ".cli").main()


def mcp_main() -> None:
    sys.argv = [sys.argv[0], *sys.argv[1:], "mcp"]
    main()


if __name__ == "__main__":
    main()
