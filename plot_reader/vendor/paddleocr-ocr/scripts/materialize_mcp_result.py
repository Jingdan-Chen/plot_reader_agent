#!/usr/bin/env python3
"""Materialize a detailed PaddleOCR MCP response into a paper run directory."""

import argparse
import base64
import binascii
import json
import mimetypes
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Absolute source PDF path")
    parser.add_argument("--result-json", required=True, help="Temporary JSON file containing the MCP result")
    parser.add_argument("--output-root", help="Output parent; defaults to the source directory")
    parser.add_argument("--mcp-seconds", type=float, help="Elapsed MCP call time")
    parser.add_argument("--keep-result", action="store_true", help="Keep the temporary result JSON")
    return parser.parse_args()


def pdf_metadata(source):
    metadata = {}
    try:
        output = subprocess.run(["pdfinfo", str(source)], check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return metadata
    for line in output.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
    return metadata


def decode_image(data, mime_type):
    if data.startswith("data:"):
        _, encoded = data.split(",", 1)
    else:
        encoded = data
    try:
        return base64.b64decode(encoded), mime_type
    except (binascii.Error, ValueError) as exc:
        raise ValueError("MCP returned an invalid base64 image block") from exc


def image_extension(mime_type):
    extension = mimetypes.guess_extension(mime_type or "") or ".bin"
    return ".jpg" if extension == ".jpe" else extension


def text_blocks(payload):
    return [block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"]


def page_blocks(payload):
    structured = payload.get("structuredContent") or {}
    pages = structured.get("pages") if isinstance(structured, dict) else None
    if not isinstance(pages, list) or not pages:
        return None
    rendered = []
    for page in pages:
        if isinstance(page, str):
            rendered.append(page)
        elif isinstance(page, dict):
            rendered.append(page.get("markdown") or page.get("text") or "")
    return rendered if any(rendered) else None


def find_doi(text):
    match = DOI_RE.search(text)
    return match.group(0).rstrip(".,;)") if match else ""


def clean_title(text, metadata):
    if metadata.get("title"):
        return metadata["title"]
    for line in text.splitlines():
        candidate = re.sub(r"^#+\s*", "", line).strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 12:
            return candidate
    return "Untitled document"


def authors_from(metadata, text):
    if metadata.get("author"):
        return [part.strip() for part in re.split(r"\s+and\s+|;", metadata["author"]) if part.strip()]
    for line in text.splitlines()[:80]:
        candidate = re.sub(r"\$[^$]*\$", "", line).strip(" *|")
        if candidate and not candidate.startswith(("#", "<", "Key ", "Department ", "Received ", "ABSTRACT")):
            if "," in candidate and len(candidate) < 180:
                return [part.strip() for part in re.split(r",\s+|\s+and\s+", candidate) if part.strip()]
    return []


def crossref(doi="", title=""):
    try:
        if doi:
            url = "https://api.crossref.org/works/" + quote(doi, safe="")
        elif title:
            url = "https://api.crossref.org/works?" + urlencode({"query.bibliographic": title, "rows": 1})
        else:
            return {}
        request = Request(url, headers={"User-Agent": "paddleocr-ocr/1.0"})
        with urlopen(request, timeout=10) as response:
            message = json.load(response).get("message", {})
            return (message.get("items") or [{}])[0] if not doi else message
    except Exception:
        return {}


def citation_key(authors, year, title):
    first = re.sub(r"[^A-Za-z0-9]", "", authors[0].split(",")[0] if authors else "Document")
    year_text = str(year) if re.fullmatch(r"\d{4}", str(year)) else "nd"
    keyword = re.sub(r"[^A-Za-z0-9]", "", title.split()[0] if title else "Paper")
    return f"{first or 'Document'}{year_text}{keyword or 'Paper'}"


def bibtex_escape(value):
    return str(value).replace("\\", "\\textbackslash ").replace("{", "\\{").replace("}", "\\}")


def write_bibtex(directory, metadata, first_text):
    doi = find_doi("\n".join([first_text, json.dumps(metadata)]))
    provisional_title = clean_title(first_text, metadata)
    remote = crossref(doi, provisional_title)
    doi = doi or remote.get("DOI", "")
    title = (remote.get("title") or [None])[0] or clean_title(first_text, metadata)
    remote_authors = remote.get("author") or []
    authors = [f"{item.get('family', '')}, {item.get('given', '')}".strip(", ") for item in remote_authors]
    authors = authors or authors_from(metadata, first_text)
    date_parts = ((remote.get("published-print") or remote.get("published-online") or {}).get("date-parts") or [[]])[0]
    date_text = metadata.get("creationdate", "")
    year_match = re.search(r"\b(?:19|20)\d{2}\b", date_text) or re.search(r"\b(?:19|20)\d{2}\b", first_text)
    year = date_parts[0] if date_parts else (year_match.group(0) if year_match else "")
    journal = (remote.get("container-title") or [None])[0] or metadata.get("subject", "")
    volume = remote.get("volume") or ""
    issue = remote.get("issue") or ""
    pages = remote.get("page") or remote.get("article-number") or ""
    key = citation_key(authors, year, title)
    fields = [("author", " and ".join(authors)), ("title", title), ("year", year)]
    if journal:
        fields.append(("journal", journal))
    if volume:
        fields.append(("volume", volume))
    if issue:
        fields.append(("number", issue))
    if pages:
        fields.append(("pages", pages))
    if doi:
        fields.extend([("doi", doi), ("url", f"https://doi.org/{doi}")])
    entry_type = "article" if journal else "misc"
    lines = [f"@{entry_type}{{{key},"]
    for index, (field, value) in enumerate(fields):
        suffix = "," if index < len(fields) - 1 else ""
        lines.append(f"  {field:<7} = {{{bibtex_escape(value)}}}{suffix}")
    lines.append("}")
    (directory / "citation.bib").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doi, key


def output_directory(source, output_root):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "document"
    base = Path(output_root or source.parent)
    candidate = base / f"{stem}--{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stem}--{stamp}-{suffix:02d}"
        suffix += 1
    return candidate


def verify_image_links(md_dir, picture_dir):
    missing = []
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for markdown_file in md_dir.glob("*.md"):
        markdown = markdown_file.read_text(encoding="utf-8")
        for target in image_pattern.findall(markdown):
            if target.startswith("../pictures/"):
                target_path = (markdown_file.parent / target).resolve()
                if not target_path.is_file() or picture_dir.resolve() not in target_path.parents:
                    missing.append(f"{markdown_file.name}: {target}")
    if missing:
        raise RuntimeError("Markdown image targets are missing: " + ", ".join(missing))


def materialize(args):
    started = time.perf_counter()
    source = Path(args.source).expanduser().resolve()
    result_path = Path(args.result_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source document not found: {source}")
    if not result_path.is_file():
        raise FileNotFoundError(f"MCP result JSON not found: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("isError"):
        raise RuntimeError("MCP returned an error result; no output was created")
    output = output_directory(source, args.output_root)
    md_dir, picture_dir, bibtex_dir = output / "md", output / "pictures", output / "bibtex"
    md_dir.mkdir(parents=True)
    picture_dir.mkdir()
    bibtex_dir.mkdir()
    blocks = payload.get("content", [])
    image_count = 0
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            image_count += 1
            data, mime_type = decode_image(block.get("data", ""), block.get("mimeType", ""))
            filename = f"paddleocr_image_{image_count:03d}{image_extension(mime_type)}"
            (picture_dir / filename).write_bytes(data)
            parts.append(f"\n![PaddleOCR image {image_count}](../pictures/{filename})\n")
    pages = page_blocks(payload)
    if pages:
        for index, page in enumerate(pages):
            (md_dir / f"doc_{index:04d}.md").write_text(page, encoding="utf-8")
        markdown_count = len(pages)
        combined = "\n\n".join(pages)
    else:
        combined = "\n\n".join(parts)
        (md_dir / "document.md").write_text(combined, encoding="utf-8")
        markdown_count = 1
    text = combined
    verify_image_links(md_dir, picture_dir)
    metadata = pdf_metadata(source)
    page_match = re.search(r"\bPages:\s*(\d+)", text)
    page_count = int(page_match.group(1)) if page_match else int(metadata.get("pages", "0") or 0)
    doi, key = write_bibtex(bibtex_dir, metadata, text[:12000])
    materialization_seconds = time.perf_counter() - started
    mcp_seconds = args.mcp_seconds
    total_seconds = (mcp_seconds + materialization_seconds) if mcp_seconds is not None else None
    benchmark = [
        "PaddleOCR-VL-1.6 MCP benchmark",
        "============================",
        f"Input: {source}",
        "MCP tool: paddleocr_vl",
        "Inference source: aistudio",
        f"Detected pages: {page_count or 'unknown'}",
        f"Markdown files: {markdown_count}",
        f"Pictures: {image_count}",
        f"BibTeX key: {key}",
        f"DOI: {doi or 'not resolved'}",
        f"MCP processing seconds: {mcp_seconds if mcp_seconds is not None else 'not supplied'}",
        f"Materialization seconds: {materialization_seconds:.3f}",
        f"Total seconds: {total_seconds:.3f}" if total_seconds is not None else "Total seconds: not available",
    ]
    (output / "benchmark.txt").write_text("\n".join(benchmark) + "\n", encoding="utf-8")
    if not args.keep_result:
        result_path.unlink(missing_ok=True)
    print(f"output={output}")
    print(f"pages={page_count or 'unknown'} markdown={markdown_count} pictures={image_count} bibtex={key}")
    print(f"materialization_seconds={materialization_seconds:.3f}")


if __name__ == "__main__":
    try:
        materialize(arguments())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}")
