#!/usr/bin/env python3
"""Submit a document to the PaddleOCR AI Studio jobs API and materialize its result."""

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:
    raise SystemExit("Error: requests is required; install it with 'python3 -m pip install requests'") from exc

from materialize_mcp_result import pdf_metadata, write_bibtex

DEFAULT_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_MODEL = "PaddleOCR-VL-1.6"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Absolute/local file path or HTTP(S) URL")
    parser.add_argument("--output-root")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--job-timeout", type=float, default=7200.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args()


def api_json(response, stage):
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = (getattr(response, "text", "") or "")[:500]
        raise RuntimeError(f"{stage} failed: HTTP {response.status_code}: {detail}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{stage} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{stage} returned an unexpected response")
    return payload


def is_url(value):
    return urlparse(value).scheme.lower() in {"http", "https"}


def clean_name(value, fallback):
    name = Path(urlparse(value).path).name if is_url(value) else Path(value).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or fallback


def submit(session, job_url, token, model, source, timeout):
    headers = {"Authorization": f"bearer {token}"}
    options = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }
    if is_url(source):
        response = session.post(
            job_url,
            headers={**headers, "Content-Type": "application/json"},
            json={"fileUrl": source, "model": model, "optionalPayload": options},
            timeout=timeout,
        )
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source document not found: {path}")
        with path.open("rb") as handle:
            response = session.post(
                job_url,
                headers=headers,
                data={"model": model, "optionalPayload": json.dumps(options)},
                files={"file": (path.name, handle)},
                timeout=timeout,
            )
    data = api_json(response, "Job submission").get("data") or {}
    job_id = data.get("jobId")
    if not job_id:
        raise RuntimeError("Job submission response did not contain data.jobId")
    return str(job_id), headers


def poll(session, job_url, job_id, headers, poll_seconds, job_timeout, request_timeout):
    deadline = time.monotonic() + job_timeout
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Job {job_id} did not finish within {job_timeout:g} seconds")
        response = session.get(f"{job_url}/{job_id}", headers=headers, timeout=request_timeout)
        data = api_json(response, "Job polling").get("data") or {}
        state = str(data.get("state", "")).lower()
        if state == "done":
            result_url = (data.get("resultUrl") or {}).get("jsonUrl")
            if not result_url:
                raise RuntimeError("Completed job did not contain data.resultUrl.jsonUrl")
            return result_url, data
        if state == "failed":
            raise RuntimeError(f"PaddleOCR job failed: {data.get('errorMsg') or 'unknown error'}")
        if state not in {"pending", "running"}:
            raise RuntimeError(f"Job polling returned unknown state: {state or '<missing>'}")
        time.sleep(poll_seconds)


def download(session, url, timeout):
    response = session.get(url, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Result download failed: HTTP {response.status_code}") from exc
    return response


def parse_jsonl(text):
    layouts = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid result JSONL at line {line_number}") from exc
        result = item.get("result") or {}
        page_results = result.get("layoutParsingResults") or []
        if not isinstance(page_results, list):
            raise RuntimeError(f"Unexpected layoutParsingResults at JSONL line {line_number}")
        layouts.extend(page_results)
    if not layouts:
        raise RuntimeError("PaddleOCR result contained no layoutParsingResults")
    return layouts


def choose_output(source, output_root):
    stem = Path(clean_name(source, "document")).stem
    root = Path(output_root).expanduser().resolve() if output_root else (
        Path.cwd() if is_url(source) else Path(source).expanduser().resolve().parent
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root / f"{stem}--{stamp}"
    suffix = 1
    while output.exists():
        output = root / f"{stem}--{stamp}-{suffix:02d}"
        suffix += 1
    return output


def save_image(session, url, destination, timeout):
    response = download(session, url, timeout)
    destination.write_bytes(response.content)


def verify_local_image_links(md_dir, pictures_dir):
    missing = []
    pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    picture_root = pictures_dir.resolve()
    for markdown_file in md_dir.glob("*.md"):
        for target in pattern.findall(markdown_file.read_text(encoding="utf-8")):
            target = target.strip().split(maxsplit=1)[0].strip("<>")
            if is_url(target) or target.startswith(("data:", "#")):
                continue
            target_path = (markdown_file.parent / target).resolve()
            if not target_path.is_file() or picture_root not in target_path.parents:
                missing.append(f"{markdown_file.name}: {target}")
    if missing:
        raise RuntimeError("Markdown image targets are missing or outside pictures/: " + ", ".join(missing))


def materialize(session, source, layouts, progress, model, api_seconds, output_root, timeout, total_started):
    started = time.perf_counter()
    output = choose_output(source, output_root)
    temporary = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    md_dir, pictures_dir, bibtex_dir = temporary / "md", temporary / "pictures", temporary / "bibtex"
    warnings = []
    try:
        for directory in (md_dir, pictures_dir, bibtex_dir):
            directory.mkdir(parents=True, exist_ok=False)
        picture_count = 0
        markdown_texts = []
        for page_number, page in enumerate(layouts):
            markdown = page.get("markdown") or {}
            text = markdown.get("text") or ""
            images = markdown.get("images") or {}
            for image_number, (image_path, image_url) in enumerate(images.items(), 1):
                picture_count += 1
                filename = f"page_{page_number:04d}_{image_number:03d}_{clean_name(image_path, 'image.bin')}"
                save_image(session, image_url, pictures_dir / filename, timeout)
                text = text.replace(str(image_path), f"../pictures/{filename}")
                text = text.replace(str(image_url), f"../pictures/{filename}")
            for output_number, (image_name, image_url) in enumerate((page.get("outputImages") or {}).items(), 1):
                picture_count += 1
                filename = f"page_{page_number:04d}_output_{output_number:03d}_{clean_name(image_name, 'image.jpg')}"
                save_image(session, image_url, pictures_dir / filename, timeout)
            (md_dir / f"doc_{page_number:04d}.md").write_text(text, encoding="utf-8")
            markdown_texts.append(text)
        verify_local_image_links(md_dir, pictures_dir)
        metadata = {} if is_url(source) else pdf_metadata(Path(source).expanduser().resolve())
        combined = "\n\n".join(markdown_texts)
        doi, citation_key = write_bibtex(bibtex_dir, metadata, combined[:12000])
        extracted = (progress.get("extractProgress") or {}).get("extractedPages")
        page_count = int(extracted) if extracted is not None else len(layouts)
        materialization_seconds = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started
        benchmark = [
            f"{model} API benchmark",
            "============================",
            f"Input: {source}",
            "Backend: api",
            f"API endpoint: {os.environ.get('PADDLEOCR_API_URL', DEFAULT_JOB_URL)}",
            "Inference source: aistudio",
            f"Detected pages: {page_count}",
            f"Markdown files: {len(layouts)}",
            f"Pictures: {picture_count}",
            f"BibTeX key: {citation_key}",
            f"DOI: {doi or 'not resolved'}",
            f"API processing seconds: {api_seconds:.3f}",
            f"Materialization seconds: {materialization_seconds:.3f}",
            f"Total seconds: {total_seconds:.3f}",
            f"Warnings: {'; '.join(warnings) if warnings else 'none'}",
        ]
        (temporary / "benchmark.txt").write_text("\n".join(benchmark) + "\n", encoding="utf-8")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"output={output}")
    print(f"pages={page_count} markdown={len(layouts)} pictures={picture_count} bibtex={citation_key}")
    print(f"api_seconds={api_seconds:.3f} materialization_seconds={materialization_seconds:.3f} total_seconds={total_seconds:.3f}")


def main():
    args = arguments()
    token = os.environ.get("PADDLEOCR_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("Error: PADDLEOCR_API_TOKEN is not set")
    job_url = os.environ.get("PADDLEOCR_API_URL", DEFAULT_JOB_URL).rstrip("/")
    model = os.environ.get("PADDLEOCR_MODEL", DEFAULT_MODEL)
    source = args.source if is_url(args.source) else str(Path(args.source).expanduser().resolve())
    session = requests.Session()
    total_started = time.perf_counter()
    api_started = time.perf_counter()
    job_id, headers = submit(session, job_url, token, model, source, args.request_timeout)
    result_url, progress = poll(
        session, job_url, job_id, headers, args.poll_seconds, args.job_timeout, args.request_timeout
    )
    result_text = download(session, result_url, args.request_timeout).text
    api_seconds = time.perf_counter() - api_started
    layouts = parse_jsonl(result_text)
    materialize(
        session, source, layouts, progress, model, api_seconds,
        args.output_root, args.request_timeout, total_started,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TimeoutError, requests.RequestException) as exc:
        raise SystemExit(f"Error: {exc}")
