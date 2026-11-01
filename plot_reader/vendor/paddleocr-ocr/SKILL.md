---
name: paddleocr-ocr
description: Extract research papers, PDFs, and document images into Markdown with figures and BibTeX using PaddleOCR-VL-1.6. Uses the AI Studio jobs API by default and retains the configured MCP tool as an optional backend.
---

# PaddleOCR-VL paper extraction

Use PaddleOCR-VL-1.6 and the bundled scripts. Do not recreate the API client or materialization logic during a paper intake.

## Backend toggle

Read `PADDLEOCR_BACKEND`; valid values are `api` and `mcp` (case-insensitive). If it is unset, use `api`. A user may explicitly choose a backend for the current request; that choice overrides the environment variable. Reject any other value instead of silently falling back.

Both backends send the document to the same AI Studio inference provider. Never put an access token in this skill, command arguments, logs, temporary result files, benchmark output, or source control.

## API workflow (default)

Preflight:

- Resolve local inputs to absolute paths and confirm they exist. HTTP(S) input URLs are also supported.
- Confirm the `requests` Python package is importable. If absent, install it only with the user's approval.
- Require `PADDLEOCR_API_TOKEN` in the environment. If missing, stop and explain how to set it without asking the user to paste it into a tracked file.

Run:

```bash
python3 scripts/extract_api.py '/absolute/path/to/document.pdf'
```

The script submits the document to `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs`, polls until completion, downloads the JSONL result and images, and creates a non-overwriting sibling directory named `<document-stem>--YYYYMMDD-HHMMSS/`. Optional configuration is available through `--output-root`, `--poll-seconds`, `--job-timeout`, `PADDLEOCR_API_URL`, and `PADDLEOCR_MODEL`.

Do not retry a failed API run more than 3 times. Wait 60 seconds between attempts. A terminal job failure is an attempt failure, not a reason to switch to MCP unless the user explicitly changes the toggle.

## MCP workflow (optional)

When the selected backend is `mcp`, confirm `paddleocr_vl` is callable in the current session, then call it with:

```json
{
  "input_data": "/absolute/path/to/document.pdf",
  "file_type": "pdf",
  "output_mode": "detailed",
  "return_images": true
}
```

Measure the call time, save the detailed response to a temporary JSON file outside the final output directory, then run:

```bash
python3 scripts/materialize_mcp_result.py \
  --source '/absolute/path/to/document.pdf' \
  --result-json '/temporary/path/mcp-result.json' \
  --mcp-seconds '<elapsed MCP seconds>'
```

If the tool is absent, report that a fresh session may be needed after enabling the PaddleOCR MCP server. Do not switch to API unless the user changes the toggle.

## Required output and report

Each successful backend creates a new run containing `benchmark.txt`, `md/`, `pictures/`, and `bibtex/citation.bib`. Verify that every local Markdown image link resolves inside `pictures/`.

Report the selected backend, model/tool, page count, run path, Markdown count, picture count, BibTeX path, backend/materialization/total timings, and warnings. The BibTeX entry describes the analyzed document; when DOI enrichment is unavailable, retain the valid metadata-based fallback and identify missing fields.

On submission, polling, download, MCP, materialization, or validation failure, preserve the source and all prior runs, report the failed stage exactly, and do not claim OCR success.

