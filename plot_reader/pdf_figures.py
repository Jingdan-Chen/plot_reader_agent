"""PDF figure intake using the installed PaddleOCR skill's bundled scripts.

This adapter selects pages and indexes materialized pictures. The skill owns
submission, polling, download and materialization; no second API client lives here.
"""
from __future__ import annotations

import asyncio
import html
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from urllib.parse import unquote

from PIL import Image


class FigureIntakeError(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def page_range(page_count: int, start=None, end=None) -> tuple[int, int]:
    if start is None and end is None and page_count <= 10:
        return 1, page_count
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= page_count:
        raise FigureIntakeError("page_selection", f"Choose a page range from 1 to {page_count}.")
    return start, end


def selected_backend() -> str:
    backend = os.environ.get("PADDLEOCR_BACKEND", "api").strip().lower()
    if backend not in {"api", "mcp"}:
        raise FigureIntakeError("preflight", "PADDLEOCR_BACKEND must be api or mcp; no fallback was used.")
    return backend


def skill_directory() -> Path:
    installed=Path.home()/".codex/skills/paddleocr-ocr"
    bundled=Path(__file__).with_name('vendor')/'paddleocr-ocr'
    root = Path(os.environ.get("PADDLEOCR_SKILL_DIR", installed if installed.is_dir() else bundled)).expanduser().resolve()
    for filename in ("extract_api.py", "materialize_mcp_result.py"):
        if not (root/"scripts"/filename).is_file():
            raise FigureIntakeError("preflight", "The installed paddleocr-ocr skill scripts could not be found.")
    return root


def api_python() -> str:
    # Reuse an installed interpreter; never install packages during an upload.
    configured = os.environ.get("PADDLEOCR_PYTHON")
    options = [configured] if configured else [sys.executable, shutil.which("python3")]
    for executable in dict.fromkeys(p for p in options if p):
        try:
            probe = subprocess.run([executable, "-c", "import requests"], capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return executable
    raise FigureIntakeError("preflight", "PaddleOCR needs a Python installation with requests. Set PADDLEOCR_PYTHON to an existing interpreter, or approve installing requests separately.")


def sanitize(message: str) -> str:
    for key, value in os.environ.items():
        if value and re.search(r"TOKEN|SECRET|API_KEY", key, re.I):
            message = message.replace(value, "[redacted]")
    return message[-1600:]


def failure_stage(message: str) -> str:
    explicit=re.search(r"^PADDLEOCR_STAGE=(submission|polling|download|materialization|validation)$",message,re.M)
    if explicit:
        return explicit.group(1)
    low = message.lower()
    for stage, needles in (("submission", ("submission",)),
                           ("polling", ("polling", "job failed", "did not finish")),
                           ("download", ("download",)),
                           ("validation", ("image targets", "image link", "jsonl", "layoutparsingresults"))):
        if any(n in low for n in needles):
            return stage
    return "backend_or_materialization"


async def run_script(command: list[str], stage: str) -> str:
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=7500)
    except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(),timeout=5)
            except asyncio.TimeoutError:
                os.killpg(process.pid,signal.SIGKILL)
                await process.wait()
        if isinstance(exc,asyncio.TimeoutError):
            raise FigureIntakeError(stage,"The PaddleOCR skill process timed out; the source and prior runs are preserved.") from None
        raise
    out, err = stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")
    if process.returncode:
        message = sanitize(err or out or f"Skill process exited with status {process.returncode}")
        detected=failure_stage(message)
        raise FigureIntakeError(detected if detected!="backend_or_materialization" else stage, message)
    return out


async def select_pages(source: Path, count: int, start: int, end: int, folder: Path) -> Path:
    if start == 1 and end == count:
        return source
    separate, unite = shutil.which("pdfseparate"), shutil.which("pdfunite")
    if not separate or not unite:
        raise FigureIntakeError("page_selection", "Poppler pdfseparate and pdfunite are required to isolate the selected pages.")
    pages = folder/"selected-pages"
    pages.mkdir()
    await run_script([separate, "-f", str(start), "-l", str(end), str(source), str(pages/"page-%d.pdf")], "page_selection")
    selection = folder/f"pages-{start}-{end}.pdf"
    await run_script([unite, *[str(pages/f"page-{i}.pdf") for i in range(start, end+1)], str(selection)], "page_selection")
    return selection


async def run_mcp(source: Path, scripts: Path, output: Path) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config_path = Path(os.environ.get("CODEX_HOME", Path.home()/".codex"))/"config.toml"
    try:
        config = tomllib.loads(config_path.read_text()).get("mcp_servers", {}).get("paddleocr_vl", {})
    except (OSError, ValueError):
        config = {}
    if not config.get("command") or config.get("enabled") is False:
        raise FigureIntakeError("preflight", "The configured paddleocr_vl MCP server is unavailable. Enable its stdio server and restart the portal; no API fallback was used.")
    started = time.perf_counter()
    try:
        parameters = StdioServerParameters(command=config["command"], args=config.get("args", []),
                                           env={**os.environ, **config.get("env", {})})
        # The server may write diagnostics containing configuration. Do not persist them.
        with open(os.devnull, "w") as errlog:
            async with stdio_client(parameters, errlog=errlog) as streams:
                async with ClientSession(*streams) as session:
                    await asyncio.wait_for(session.initialize(), timeout=config.get("startup_timeout_sec", 60))
                    listing = await session.list_tools()
                    if "paddleocr_vl" not in {t.name for t in listing.tools}:
                        raise FigureIntakeError("mcp", "The configured server does not expose paddleocr_vl.")
                    result = await session.call_tool("paddleocr_vl", {
                        "input_data": str(source), "file_type": "pdf", "output_mode": "detailed", "return_images": True,
                    }, read_timeout_seconds=timedelta(seconds=7200))
        if result.isError:
            raise FigureIntakeError("mcp", "PaddleOCR MCP returned an error; the source and prior runs were preserved.")
    except FigureIntakeError:
        raise
    except Exception as exc:
        # Configured MCP credentials may not be in this process's environment.
        raise FigureIntakeError("mcp", f"PaddleOCR MCP call failed ({type(exc).__name__}); no API fallback was used.") from None
    elapsed = time.perf_counter()-started
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(result.model_dump_json(by_alias=True))
    try:
        return await run_script([sys.executable, str(scripts/"materialize_mcp_result.py"),
            "--source", str(source), "--result-json", str(temporary), "--mcp-seconds", str(elapsed),
            "--output-root", str(output)], "materialization")
    finally:
        temporary.unlink(missing_ok=True)


IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)|<img\b[^>]*?\bsrc\s*=\s*([\"'])(.*?)\3[^>]*>",re.I)


def index_figures(run: Path, start: int, end: int) -> tuple[list[dict], dict]:
    pictures = (run/"pictures").resolve()
    markdown_files = sorted((run/"md").glob("*.md"))
    if not markdown_files or not pictures.is_dir() or not (run/"bibtex/citation.bib").is_file() or not (run/"benchmark.txt").is_file():
        raise FigureIntakeError("validation", "The skill output is incomplete: Markdown, pictures, benchmark and BibTeX are required.")
    figures, seen, warnings = [], set(), []
    numbered=[re.fullmatch(r"doc_(\d+)",path.stem) for path in markdown_files]
    if all(numbered) and {int(match.group(1)) for match in numbered} != set(range(end-start+1)):
        raise FigureIntakeError("validation","OCR did not return a Markdown page for every selected page.")
    for markdown in markdown_files:
        text = markdown.read_text(encoding="utf-8")
        number = re.fullmatch(r"doc_(\d+)", markdown.stem)
        page = start+int(number.group(1)) if number else (start if start == end else None)
        if page is not None and page > end:
            raise FigureIntakeError("validation", "OCR page count exceeds the selected page range.")
        matches = list(IMAGE_LINK.finditer(text))
        for i, match in enumerate(matches):
            raw = html.unescape((match.group(2) or match.group(4)).strip())
            if match.group(2):
                raw=raw.split(maxsplit=1)[0].strip("<>")
            if raw.startswith(("http:", "https:", "data:")):
                warnings.append("An OCR image was not materialized locally and is not selectable.")
                continue
            path = (markdown.parent/unquote(raw)).resolve()
            if pictures not in path.parents or not path.is_file():
                raise FigureIntakeError("validation", "An OCR Markdown image link is missing or escapes pictures/.")
            if path in seen:
                continue
            seen.add(path)
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
            except (OSError, ValueError):
                warnings.append(f"An unsupported picture ({path.suffix}) was omitted from the gallery.")
                continue
            following = text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)][:700]
            preceding = text[matches[i-1].end() if i else 0:match.start()][-500:]
            context=html.unescape(re.sub(r"<[^>]*>","",following+"\n"+preceding))
            caption = next((re.sub(r"[*#_]", "", line).strip() for line in context.splitlines()
                            if re.match(r"\s*[*#_]*\s*(?:fig(?:ure)?\.?|scheme)\s*\d", line, re.I)), "")[:600]
            label = re.match(r"(?:fig(?:ure)?\.?|scheme)\s*\d+[a-z]?", caption, re.I)
            title = label.group(0) if label else f"Picture {len(figures)+1}"
            figures.append({"id": f"F{len(figures)+1:03d}", "page": page, "title": title,
                            "caption": caption, "image_size": [width, height],
                            "path": str(path.relative_to(run)), "markdown": markdown.name})
    benchmark = (run/"benchmark.txt").read_text()
    fields = dict(line.split(": ", 1) for line in benchmark.splitlines() if ": " in line)
    reported_pages = fields.get("Detected pages")
    if reported_pages and reported_pages.isdigit() and int(reported_pages) != end-start+1:
        raise FigureIntakeError("validation", "OCR did not return all selected pages; no complete gallery was claimed.")
    bib = (run/"bibtex/citation.bib").read_text()
    missing = [key for key in ("title", "author", "year", "doi") if not re.search(r"\b"+key+r"\s*=\s*\{[^}]+\}", bib, re.I)]
    if missing:
        warnings.append("Citation metadata is incomplete; verify " + ", ".join(missing) + ".")
    if start > 1:
        warnings.append("This is a selected-page excerpt; verify its citation against the original PDF.")
    if not figures:
        warnings.append("PaddleOCR returned no locally viewable pictures in these pages.")
    report = {"run_path": str(run), "page_count": end-start+1, "page_range": [start, end],
              "markdown_count": len(markdown_files), "picture_count": len(list(pictures.iterdir())),
              "figure_count": len(figures), "bibtex_path": str(run/"bibtex/citation.bib"),
              "timings": {key: fields.get(key) for key in ("API processing seconds", "MCP processing seconds", "Materialization seconds", "Total seconds") if fields.get(key)},
              "warnings": list(dict.fromkeys(warnings))}
    return figures, report


async def extract_figures(source: Path, count: int, start: int, end: int, folder: Path, progress) -> tuple[list[dict], dict]:
    backend = selected_backend()
    source = source.resolve()
    if not source.is_file():
        raise FigureIntakeError("preflight", "The uploaded PDF no longer exists.")
    scripts = skill_directory()/"scripts"
    python = None
    if backend == "api":
        if not os.environ.get("PADDLEOCR_API_TOKEN", "").strip():
            raise FigureIntakeError("preflight", "PADDLEOCR_API_TOKEN is not set. Set it in the portal process environment and restart; do not place it in a source file or paste it into this form.")
        python = await asyncio.to_thread(api_python)
    await progress("page_selection", "Preparing the selected PDF pages")
    submitted = await select_pages(source, count, start, end, folder)
    await progress("ocr", "Finding pictures with PaddleOCR")
    output = folder/"runs"
    output.mkdir()
    if backend == "api":
        stdout = await run_script([python, str(Path(__file__).with_name("paddleocr_skill_runner.py")),
            str(scripts/"extract_api.py"), str(submitted), "--output-root", str(output)], "api")
    else:
        stdout = await run_mcp(submitted, scripts, output)
    match = re.search(r"^output=(.+)$", stdout, re.M)
    if not match:
        raise FigureIntakeError("materialization", "The skill did not report a completed run directory.")
    run = Path(match.group(1)).resolve()
    if output.resolve() not in run.parents:
        raise FigureIntakeError("validation", "The skill output is outside this intake run.")
    await progress("validation", "Preparing the figure gallery")
    figures, report = await asyncio.to_thread(index_figures, run, start, end)
    for figure in figures:
        figure["path"] = str((run/figure["path"]).relative_to(folder))
    report.update(backend=backend, model=os.environ.get("PADDLEOCR_MODEL", "PaddleOCR-VL-1.6") if backend == "api" else "paddleocr_vl",
                  original_source=str(source), submitted_source=str(submitted))
    return figures, report
