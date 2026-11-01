"""Local Codex browser interface for guided plot digitisation."""
from __future__ import annotations

import asyncio
import csv
import copy
from contextlib import asynccontextmanager
from functools import wraps
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
import uuid

from PIL import Image, ImageDraw
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
import uvicorn

from . import core
from .browser_measurement import measure_selected
from .browser_grounding import ground_scatter
from .timing import AnalysisTiming, current_timing, run_local, local_work, timing_span
from .pdf_figures import FigureIntakeError, extract_figures, page_range, sanitize
from .browser_evidence import (coordinate_evidence, selection_overlay,
                               apply_selection_review, unsupported_centers, focused_marker_candidates, unique_selection_labels,
                               bind_review_candidates)
from .scatter_beta import plan_scatter, ground_scatter_beta
from .luna_runtime import (COMMENT_SCHEMA, SELECTED_SCHEMA, SELECTION_REVIEW_SCHEMA, FOCUSED_REVIEW_SCHEMA, LunaError, LunaRunner,
                           worker_dir, assistant_profile, EFFORT_CHOICES, DEFAULT_EFFORT)

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).with_name("browser_ui")
DATA_ROOT = Path(os.environ.get("PLOT_READER_WEB_DATA", ROOT / "artifacts" / "browser")).resolve()
MAX_UPLOAD = 60 * 1024 * 1024
MAX_PDF_PAGES = 350
MAX_IMAGE_PIXELS = 100_000_000
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED = ALLOWED_IMAGE | {".pdf"}
MEASUREMENT_DEFAULTS = {
    "selected": {
        "plot": {
            "Scatter plot": "All data points",
            "Markers with fitted lines": "All data points in every marker series",
            "Line plot": "All visible peaks and turning points across all data series",
            "default": "All visible measurable features across all data series",
        },
        "spectroscopy": {
            "NMR": "All visible NMR peaks",
            "IR": "All visible absorption bands",
            "Raman": "All visible Raman peaks",
            "UV–Vis": "All visible absorption peaks",
            "default": "All visible spectral peaks and bands",
        },
        "other": "All visible measurable features",
    },
    "complete": {
        "plot": "All visible data series, including markers and fitted curves",
        "spectroscopy": "All visible spectral traces",
        "other": "All visible curves and data series",
    },
    "x_positions": "All visible data series at the requested x positions",
}


def _default_targets(category: str, mode: str, description: str, x_positions: list) -> str:
    if mode == "selected" and x_positions:
        return MEASUREMENT_DEFAULTS["x_positions"]
    choice = MEASUREMENT_DEFAULTS[mode][category]
    if isinstance(choice, dict):
        return choice.get(description.split(",", 1)[0].strip(), choice["default"])
    return choice


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("invalid local item ID")
    return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / _safe_id(job_id)


def _upload_dir(upload_id: str) -> Path:
    return DATA_ROOT / "sources" / _safe_id(upload_id)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _run(command: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _pdf_page_count(path: Path) -> int:
    tool = shutil.which("pdfinfo")
    if not tool:
        raise ValueError("Poppler pdfinfo is required to read PDFs")
    proc = _run([tool, str(path)])
    if proc.returncode:
        raise ValueError("PDF could not be read: " + proc.stderr[-500:])
    match = re.search(r"^Pages:\s+(\d+)\s*$", proc.stdout, re.M)
    if not match:
        raise ValueError("Could not determine PDF page count")
    count = int(match.group(1))
    if count < 1 or count > MAX_PDF_PAGES:
        raise ValueError(f"PDFs must have between 1 and {MAX_PDF_PAGES} pages")
    return count


def _pdf_text_pages(path: Path, count: int) -> list[str]:
    tool = shutil.which("pdftotext")
    if not tool:
        return [""] * count
    proc = _run([tool, "-layout", str(path), "-"], timeout=120)
    if proc.returncode:
        return [""] * count
    pages = proc.stdout.split("\f")
    if len(pages) < count:
        pages.extend([""] * (count - len(pages)))
    return pages[:count]


def _match_pdf_pages(pages: list[str], description: str) -> list[dict[str, Any]]:
    words = [x.casefold() for x in re.findall(r"[\w.]+", description) if len(x) > 1]
    numbered = re.search(r"\b(?:fig(?:ure)?\.?\s*)(\d+[a-z]?)\b", description, re.I)
    ranked = []
    for number, text in enumerate(pages, 1):
        low = text.casefold()
        score = sum(1 for word in set(words) if word in low)
        if numbered and re.search(r"\bfig(?:ure)?\.?\s*" + re.escape(numbered.group(1).casefold()) + r"\b", low):
            score += 8
        ranked.append({"page": number, "score": score,
                       "caption": next((line.strip()[:240] for line in text.splitlines()
                                        if re.search(r"\bfig(?:ure)?\.?\s*\d", line, re.I)), "")})
    ranked.sort(key=lambda x: (-x["score"], x["page"]))
    return ranked[:8]


def _render_pdf_page(path: Path, page: int, output: Path, dpi: int = 120) -> Path:
    tool = shutil.which("pdftoppm")
    if not tool:
        raise ValueError("Poppler pdftoppm is required to render PDFs")
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    proc = _run([tool, "-f", str(page), "-l", str(page), "-singlefile", "-png",
                 "-r", str(dpi), str(path), str(prefix)], timeout=120)
    result = prefix.with_suffix(".png")
    if proc.returncode or not result.is_file():
        raise ValueError("Could not render the selected PDF page")
    if result != output:
        os.replace(result, output)
    return output


def _crop_source(upload: dict[str, Any], target: dict[str, Any], output: Path,
                 dpi: int = 180) -> dict[str, Any]:
    source = _upload_dir(upload["id"]) / upload["filename"]
    if upload["kind"] == "pdf" and target.get("figure_id"):
        figure, source = _figure_file(upload["id"], target["figure_run_id"], target["figure_id"])
        with Image.open(source) as im:
            width,height=im.size
            crop=target.get("crop") or [0,0,width,height]
            box=[round(v) for v in crop]
            if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                raise ValueError("The selected figure crop is outside its image")
            im.crop(box).convert("RGB").save(output,format="PNG")
        return {"page":figure["page"],"dpi":None,"figure_id":figure["id"],
                "figure_run_id":target["figure_run_id"],"crop_rendered_pixels":box,
                "rendered_image_pixels":[width,height],"source_kind":"paddleocr_picture"}
    if upload["kind"] == "pdf":
        page = int(target["page"])
        rendered = output.parent / "rendered_page.png"
        _render_pdf_page(source, page, rendered, dpi)
        sx, sy = dpi / 72.0, dpi / 72.0
        crop = target.get("crop") or [0, 0, 1, 1]
        box = [round(crop[0] * sx), round(crop[1] * sy),
               round(crop[2] * sx), round(crop[3] * sy)]
        with Image.open(rendered) as im:
            width, height = im.size
            box = [max(0, min(width, box[0])), max(0, min(height, box[1])),
                   max(0, min(width, box[2])), max(0, min(height, box[3]))]
            if box[2] - box[0] < 80 or box[3] - box[1] < 80:
                raise ValueError("The confirmed crop is too small; include the plot and its axis labels")
            im.crop(box).convert("RGB").save(output, format="PNG")
        return {"page": page, "dpi": dpi, "page_points": [width / sx, height / sy],
                "crop_points": crop, "crop_rendered_pixels": box,
                "rendered_image_pixels": [round(width / sx * sx), round(height / sy * sy)]}
    with Image.open(source) as im:
        im = im.convert("RGB")
        width, height = im.size
        crop = target.get("crop") or [0, 0, width, height]
        box = [round(crop[0]), round(crop[1]), round(crop[2]), round(crop[3])]
        if box[2] - box[0] < 80 or box[3] - box[1] < 80:
            raise ValueError("The confirmed crop is too small; include the plot and its axis labels")
        im.crop(box).save(output, format="PNG")
    return {"page": None, "dpi": None, "crop_points": crop, "crop_rendered_pixels": box,
            "rendered_image_pixels": [width, height]}


def _form_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _authorize(request: Request) -> bool:
    state = request.app.state
    host = request.headers.get("host", "").split(":", 1)[0].strip("[]").casefold()
    if host not in {"127.0.0.1", "localhost"}:
        return False
    if request.url.path in {"/", "/assets/app.css", "/assets/app.js"}:
        return request.query_params.get("token") == state.token or request.url.path != "/"
    token = request.headers.get("x-plot-reader-token") or request.query_params.get("token")
    return secrets.compare_digest(token or "", state.token)


async def _auth_route(request: Request) -> bool:
    if await _authorize(request):
        return True
    return False


async def index(request: Request) -> Response:
    if not await _auth_route(request):
        return PlainTextResponse("Open Plot Reader from the URL printed when the local server started.", 403)
    page = (WEB_ROOT / "index.html").read_text()
    profile=assistant_profile()
    page=page.replace('__ASSISTANT_NAME__',html.escape(profile['name'])).replace('__ASSISTANT_PROFILE__',html.escape(profile['label']))
    options = []
    for effort in EFFORT_CHOICES[profile['provider']]:
        label = effort.capitalize() + (' (default)' if effort == DEFAULT_EFFORT else '')
        selected = ' selected' if effort == DEFAULT_EFFORT else ''
        effective = assistant_profile(effort)
        options.append(f'<option value="{html.escape(effort)}" data-profile-label="{html.escape(effective["label"])}"{selected}>{html.escape(label)}</option>')
    page = page.replace('__MODEL_EFFORT_OPTIONS__', ''.join(options))
    return HTMLResponse(page.replace("__MEASUREMENT_DEFAULTS__", json.dumps(MEASUREMENT_DEFAULTS).replace("<", "\\u003c")))


async def upload(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    form = await request.form(max_files=1, max_fields=3, max_part_size=MAX_UPLOAD)
    file = form.get("file")
    explanation = str(form.get("explanation", "")).strip()
    if not file or not getattr(file, "filename", None):
        return _form_error("Choose a PDF or image to upload")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        return _form_error("Upload a PDF, PNG, JPEG, or WebP image")
    raw = await file.read(MAX_UPLOAD + 1)
    await file.close()
    if len(raw) > MAX_UPLOAD:
        return _form_error("The file is larger than 60 MB")
    item_id = uuid.uuid4().hex
    folder = _upload_dir(item_id)
    folder.mkdir(parents=True, exist_ok=True)
    filename = "source" + suffix
    path = folder / filename
    path.write_bytes(raw)
    record: dict[str, Any] = {"id": item_id, "filename": filename,
                              "original_name": Path(file.filename).name[:200],
                              "kind": "pdf" if suffix == ".pdf" else "image",
                              "explanation": explanation, "sha256": _sha(path),
                              "created_at": _now(), "bytes": len(raw)}
    try:
        if record["kind"] == "pdf":
            count = await run_local(_pdf_page_count, path)
            record.update({"page_count": count, "requires_page_range": count > 10,
                           "page_candidates": []})
        else:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                if im.width * im.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 100 megapixel limit")
                record["image_size"] = list(im.size)
                thumb = im.convert("RGB").copy()
                thumb.thumbnail((1500, 1200))
                thumb.save(folder / "preview.png", format="PNG")
                record["preview_scale"] = [thumb.width / im.width, thumb.height / im.height]
            target={"upload_id":item_id,"page":None,"crop":[0,0,*record["image_size"]],
                    "explanation":explanation,"confirmed_at":_now()}
            _json(folder/"target.json",target)
            record["target"]=target
        _json(folder / "source.json", record)
    except (ValueError, OSError) as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return _form_error(str(exc))
    return JSONResponse(record)


def _figure_file(upload_id: str, run_id: str, figure_id: str) -> tuple[dict, Path]:
    if not re.fullmatch(r"F\d{3,6}",str(figure_id)):
        raise ValueError("Invalid figure ID")
    folder=_upload_dir(upload_id)/"figure-intake"/_safe_id(run_id)
    manifest=_read(folder/"figures.json")
    figure=next((f for f in manifest["figures"] if f["id"]==figure_id),None)
    if figure is None:
        raise ValueError("Choose a picture from this PDF's extracted gallery")
    path=(folder/figure["path"]).resolve()
    if folder.resolve() not in path.parents or not path.is_file():
        raise ValueError("Extracted picture is unavailable")
    return figure,path


async def _run_figure_intake(app, source: dict, state: dict) -> None:
    source_folder=_upload_dir(source["id"])
    folder=source_folder/"figure-intake"/state["run_id"]
    folder.mkdir(parents=True,exist_ok=True)

    def save():
        _json(folder/"intake.json",state)
        _json(source_folder/"figure-intake.json",state)

    async def progress(stage,message):
        state.update(stage=stage,message=message,updated_at=_now())
        save()

    try:
        figures,report=await extract_figures(source_folder/source["filename"],source["page_count"],
                                             state["page_start"],state["page_end"],folder,progress)
        _json(folder/"figures.json",{"figures":figures,"report":report})
        public=[{k:v for k,v in f.items() if k!="path"} for f in figures]
        state.update(status="ready",stage="complete",message=f"Found {len(figures)} pictures",
                     figures=public,report=report,finished_at=time.time())
    except asyncio.CancelledError:
        state.update(status="error",stage="interrupted",error="Figure discovery was interrupted. Your PDF and prior runs are preserved.",finished_at=time.time())
        save()
        raise
    except Exception as exc:
        state.update(status="error",stage=getattr(exc,"stage","intake"),error=sanitize(str(exc)),finished_at=time.time())
    save()


async def pdf_figures(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired",403)
    try:
        source_folder=_upload_dir(request.path_params["upload_id"])
        source=_read(source_folder/"source.json")
        if source["kind"]!="pdf":
            raise ValueError("Figure discovery requires a PDF")
        state_path=source_folder/"figure-intake.json"
        previous=_read(state_path) if state_path.is_file() else {}
        if request.method=="GET":
            return JSONResponse(previous or {"status":"awaiting_range" if source["page_count"]>10 else "not_started"})
        body=await request.json()
        start,end=page_range(source["page_count"],body.get("page_start"),body.get("page_end"))
        tasks=getattr(request.app.state,"figure_tasks",{})
        request.app.state.figure_tasks=tasks
        active=tasks.get(source["id"])
        if active and not active.done():
            return _form_error("Figure discovery is already running for this PDF",409)
        same_range=(previous.get("page_start"),previous.get("page_end"))==(start,end)
        if same_range and previous.get("status")=="ready":
            return JSONResponse(previous)
        failed=[]
        for saved in (source_folder/"figure-intake").glob("*/intake.json"):
            item=_read(saved)
            if ((item.get("page_start"),item.get("page_end"))==(start,end) and item.get("status")=="error"
                    and item.get("stage") not in {"preflight","page_selection","interrupted"}):
                failed.append(item)
        attempt=len(failed)
        if failed:
            if len(failed)>=3:
                return _form_error("PaddleOCR failed three times for these pages. Resolve the reported failure before another attempt.",409)
            wait=60-(time.time()-max(item.get("finished_at",0) for item in failed))
            if wait>0:
                return _form_error(f"Wait {math.ceil(wait)} seconds before retrying the failed PaddleOCR run.",429)
        state={"status":"processing","stage":"preflight","message":"Preparing figure discovery",
               "run_id":uuid.uuid4().hex,"page_start":start,"page_end":end,"attempt":attempt+1,
               "figures":[],"created_at":_now()}
        _json(state_path,state)
        tasks[source["id"]]=asyncio.create_task(_run_figure_intake(request.app,source,state))
        return JSONResponse(state,status_code=202)
    except (ValueError,FigureIntakeError,KeyError,FileNotFoundError,json.JSONDecodeError) as exc:
        return _form_error(str(exc))


async def pdf_figure_preview(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired",403)
    try:
        _,path=_figure_file(request.path_params["upload_id"],request.path_params["run_id"],request.path_params["figure_id"])
        return FileResponse(path,headers={"Cache-Control":"no-store"})
    except (ValueError,FileNotFoundError,KeyError,json.JSONDecodeError):
        return _form_error("Extracted picture not found",404)


async def preview(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        source = _read(_upload_dir(request.path_params["upload_id"]) / "source.json")
        page = request.path_params.get("page")
        path = (_upload_dir(source["id"]) / f"page-{int(page)}.png") if page else (_upload_dir(source["id"]) / "preview.png")
        if not path.is_file():
            return _form_error("Preview not found", 404)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
        return _form_error("Source not found", 404)


async def render_page(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        body = await request.json()
        source = _read(_upload_dir(body["upload_id"]) / "source.json")
        if source["kind"] != "pdf":
            raise ValueError("This source is not a PDF")
        page = int(body["page"])
        if not 1 <= page <= source["page_count"]:
            raise ValueError("Page is outside this PDF")
        path = _upload_dir(source["id"]) / f"page-{page}.png"
        if not path.is_file():
            await run_local(_render_pdf_page, _upload_dir(source["id"]) / source["filename"], path, 70)
        if not any(item["page"] == page for item in source["page_candidates"]):
            source["page_candidates"].append({"page": page, "score": 0, "caption": ""})
        _json(_upload_dir(source["id"]) / "source.json", source)
        return JSONResponse({"url": f"/api/source/{source['id']}/preview/{page}?token={request.app.state.token}"})
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _form_error(str(exc))


async def confirm_target(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        body = await request.json()
        source = _read(_upload_dir(body["upload_id"]) / "source.json")
        page = body.get("page")
        width, height = source.get("image_size", [0, 0])
        if source["kind"] == "pdf":
            if not body.get("figure_id") or not body.get("figure_run_id"):
                raise ValueError("Choose an extracted picture from the PDF gallery")
            figure,_=_figure_file(source["id"],body["figure_run_id"],body["figure_id"])
            page=figure["page"]
            width,height=figure["image_size"]
            crop=body.get("crop") or [0,0,width,height]
        else:
            page = None
            crop = body.get("crop") or [0, 0, *source["image_size"]]
        if (len(crop) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in crop)
                or crop[2] <= crop[0] or crop[3] <= crop[1] or crop[2]>width or crop[3]>height):
            raise ValueError("Choose a valid crop that includes the plot axes")
        explanation = str(body.get("explanation", source["explanation"])).strip()
        target = {"upload_id": source["id"], "page": page, "crop": crop,
                  "explanation": explanation, "confirmed_at": _now()}
        if source["kind"]=="pdf":
            target.update(figure_id=figure["id"],figure_run_id=body["figure_run_id"],
                          explanation=figure["caption"] or figure["title"])
        _json(_upload_dir(source["id"]) / "target.json", target)
        return JSONResponse(target)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _form_error(str(exc))


def _require_no_jev(request_data: dict[str, Any]) -> None:
    # Reject imported beta requests explicitly; never silently change providers.
    if request_data.get("jev_mode", "off") != "off":
        raise ValueError("JEV image assistance is available only on the beta branch. Start a new run without JEV on main.")


async def create_job(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    body = await request.json()
    try:
        upload_id = _safe_id(str(body["upload_id"]))
        source = _read(_upload_dir(upload_id) / "source.json")
        target = _read(_upload_dir(upload_id) / "target.json")
        if target["upload_id"] != upload_id:
            raise ValueError("Confirm the selected figure before continuing")
        category, mode = str(body["category"]), str(body["mode"])
        if category not in {"spectroscopy", "plot", "other"}:
            raise ValueError("Choose a channel")
        if mode not in {"selected", "complete"}:
            raise ValueError("Choose selected measurements or complete curves")
        if category == "other" and not str(body.get("description", "")).strip():
            raise ValueError("Describe the plot type and what should be extracted")
        running = []
        for item in (DATA_ROOT / "jobs").glob("*/job.json"):
            try:
                if _read(item).get("status") in {"queued", "running", "needs_input", "cancel_requested"}:
                    running.append(item)
            except (OSError, json.JSONDecodeError):
                continue
        if running:
            raise ValueError("One extraction at a time is supported. Finish or cancel the current run first.")
        positions = []
        for value in re.split(r"[,;\n]+", str(body.get("x_positions", ""))):
            if value.strip():
                positions.append(float(value.strip()))
        if any(not (float("-inf") < value < float("inf")) for value in positions):
            raise ValueError("x positions must be finite numbers")
        description = str(body.get("description", "")).strip()
        targets = str(body.get("targets") or "").strip() or _default_targets(category, mode, description, positions)
        allow_questions = body.get("allow_agent_questions", False)
        if type(allow_questions) is not bool:
            raise ValueError("Allow agent questions must be true or false")
        effort = body.get("model_effort", DEFAULT_EFFORT)
        if not isinstance(effort, str):
            raise ValueError("Choose an agent effort")
        profile = assistant_profile(effort)
        workflow = body.get("scatter_workflow", "beta")
        if not isinstance(workflow, str) or workflow not in {"legacy", "beta"}:
            raise ValueError("Choose concurrent or classical scatter extraction")
        _require_no_jev(body)
        request_data = {"scatter_workflow": workflow, "upload_id": upload_id, "source_sha256": source["sha256"],
                        "target": target, "category": category, "mode": mode,
                        "channel_description": description,
                        "targets": targets,
                        "x_positions": positions,
                        "allow_agent_questions": allow_questions,
                        "model_effort": profile["effort_choice"],
                        "normalization": bool(body.get("normalization", False)),
                        "normalization_reference": str(body.get("normalization_reference", "")).strip() or None,
                        "created_at": _now()}
        if request_data["normalization"] and not request_data["normalization_reference"]:
            raise ValueError("Choose a normalization reference feature")
        job_id = uuid.uuid4().hex
        folder = _job_dir(job_id)
        folder.mkdir(parents=True, exist_ok=True)
        _json(folder / "request.json", request_data)
        _json(folder / "job.json", {"id": job_id, "status": "queued", "created_at": _now(),
                                    "stage": "queued", "request": request_data,
                                    "engine": "experimental-round9_full" if mode == "complete" else "frozen-full"})
        await _emit(request.app, job_id, "queued", "Queued extraction")
        task = asyncio.create_task(_run_job(request.app, job_id))
        request.app.state.tasks[job_id] = task
        return JSONResponse({"id": job_id, "status": "queued"}, status_code=202)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError, LunaError) as exc:
        return _form_error(str(exc))


async def _emit(app: Starlette, job_id: str, kind: str, message: str = "", **extra: Any) -> None:
    event = {"at": _now(), "type": kind, "message": message, **extra}
    try:
        folder = _job_dir(job_id)
        if message and kind in {'stage','luna_started','luna_completed','engine_started','review_note','clarification','queued','complete','failed','cancelled'}:
            path=folder/'job.json'
            if path.is_file():
                job=_read(path)
                if kind=='stage' and extra.get('stage'):job['stage']=extra['stage']
                job['updated_at']=event['at']
                job['latest_action']={'message':message[:220],'at':event['at'],'type':kind}
                _json(path,job)
        with (folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        for queue in list(app.state.subscribers.get(job_id, [])):
            await queue.put(event)
    except (OSError, ValueError):
        pass


async def _job_update(app: Starlette, job_id: str, **changes: Any) -> dict[str, Any]:
    recorder = current_timing.get()
    if recorder is not None and recorder.folder == _job_dir(job_id) and changes.get("stage"):
        recorder.set_stage("user_wait" if changes.get("status") == "needs_input" else changes["stage"])
    path = _job_dir(job_id) / "job.json"
    job = _read(path)
    job.update(changes, updated_at=_now())
    _json(path, job)
    return job


async def _sync_event(app: Starlette, job_id: str, kind: str, message: str, **extra: Any) -> None:
    await _emit(app, job_id, kind, message, **extra)


def _page_manifest(request_data: dict[str, Any], output: Path) -> dict[str, Any]:
    source = _read(_upload_dir(request_data["upload_id"]) / "source.json")
    return _crop_source(source, request_data["target"], output)


def _selected_schema(request_data: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(SELECTED_SCHEMA)
    if not request_data.get("allow_agent_questions", False):
        schema["properties"]["clarifications"]["maxItems"] = 0
    return schema


def _question_policy_prompt(request_data: dict[str, Any]) -> str:
    if request_data.get("allow_agent_questions", False):
        return "Agent questions are enabled. If the image and request do not resolve an important choice, add one concise clarification with a small set of image-based choices."
    return ("Agent questions are disabled. Do not ask the user questions or request confirmation. "
            "Return clarifications as an empty array. Continue with source-supported measurements, "
            "mark individually ambiguous features unresolved, and explain limitations in notes. "
            "Do not invent missing values, axis anchors, feature identities, normalization references or baselines.")


def _question_policy(plan: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]:
    """Keep a noninteractive run moving even if a saved/model plan contains questions."""
    plan = copy.deepcopy(plan)
    if not request_data.get("allow_agent_questions", False) and plan.get("clarifications"):
        questions = plan.pop("clarifications")
        plan["clarifications"] = []
        plan.setdefault("unanswered_clarifications", []).extend(questions)
        message = "Agent questions are disabled; interpretation needs review: " + "; ".join(q["question"] for q in questions)
        plan.setdefault("question_policy_warnings", []).append(message)
        plan.setdefault("notes", []).append(message)
    if not request_data.get("allow_agent_questions", False) and not plan.get("selections") and not (plan.get("scatter_beta") and plan.get("marker_series") and not plan.get("_beta_grounded")):
        message = "Requested features could not be supported from the image; agent questions are disabled."
        plan["selections"] = [{"label": "Unresolved requested features", "series": "Unresolved", "mode": "exact",
            "candidate_id": None, "pixel": None, "x_value": None, "review_id": None,
            "review_note": message, "status": "unresolved", "reason": message}]
    if (not request_data.get("allow_agent_questions", False) and request_data.get("normalization")
            and _normalization_needs_clarification(plan)):
        plan["reference_label"] = None
        plan["baseline_pixel"] = None
        message = "Requested normalization is unsupported by the visible reference/baseline. Original values are reported; relative heights are unavailable."
        for key in ("question_policy_warnings", "notes"):
            if message not in plan.setdefault(key, []):
                plan[key].append(message)
    return plan


def _normalization_needs_clarification(plan: dict[str, Any]) -> bool:
    reference = next((item for item in plan.get("selections", []) if item.get("label") == plan.get("reference_label")), None)
    return (not reference or reference.get("status") == "unresolved" or plan.get("baseline_pixel") is None
            or (plan.get("y_axis") or {}).get("scale") == "log10")


def _selected_prompt(request_data: dict[str, Any], inspection: dict[str, Any],
                     followup: str = "") -> str:
    return f"""Measure the user's requested features from the attached plot image. Use only image evidence and the supplied local inspection record. Do not use tools. Do not invent tick values, feature assignments, units, or missing coordinates. Keep x_axis or y_axis null if that axis cannot be calibrated. Each calibrated axis needs exactly two anchors and at least one independent check. Reversed axes are valid. The source-coordinate grid labels ORIGINAL source pixels, x right and y down. Locate pixels against those labels; never infer a pixel from an expected data trend. Automatic plot_bounds and candidates are fallible heuristics: confirm against the unannotated source, especially when axes or scatter candidates are missing. Legend symbols and text are not data markers. For ambiguous printed labels, choose an inspected review region ID and explain the visible label-to-feature association in review_note. Follow the question policy below when an important choice is unresolved. Do not collapse a bulk request for all data points into one unresolved placeholder: make one selection per visible marker, with a stable series identity and point label. If the local inspector has no marker candidate at a visible point, use the image-space pixel center and mode exact so the local engine can still map it through the calibrated axes. Mark only individually ambiguous or overlapping points unresolved.

User request: {request_data['target']['explanation']}
Channel: {request_data['category']}; {request_data.get('channel_description') or 'standard Cartesian plot'}.
Requested feature description: {request_data.get('targets') or 'all values at the requested x positions'}.
Requested x positions: {json.dumps(request_data.get('x_positions', []))}.
Normalize peak heights: {request_data['normalization']}; requested reference: {request_data.get('normalization_reference') or 'none'}.
Question policy: {_question_policy_prompt(request_data)}
{('User clarification: ' + followup) if followup else ''}

Inspection record (treat IDs as candidates; prominence is not assignment confidence):
{json.dumps({k: inspection.get(k) for k in ('image_size','plot_bounds','ticks','candidates','label_assignments','review_regions','warning')}, ensure_ascii=False)}

Return only schema-conforming JSON. selections must cover every requested label and x position. Give every selection a short series identity in series. Use mode curve and x_value for an x-position query; otherwise use a reviewed candidate ID or original-image pixel. If the request is for a labeled peak and no inspected peak candidate supports local refinement, use mode exact with the visible apex pixel. The local engine maps that reviewed pixel through the axis calibration. Use mode peak only when the supplied plot bounds and image evidence support local apex refinement. For normalization, provide reference_label and baseline_pixel only when both are visibly supported.
"""


def _is_bulk_point_request(request_data: dict[str, Any]) -> bool:
    text = str(request_data.get("targets", "")).casefold()
    return request_data.get("category") == "plot" and request_data.get("mode") == "selected" and bool(
        re.search(r"\b(all|every|each)\b.{0,40}\b(data points?|markers?)\b", text)
    )


def _is_aggregate_unresolved(plan: dict[str, Any]) -> bool:
    selections = plan.get("selections", [])
    return len(selections) == 1 and selections[0].get("status") == "unresolved"


def _candidate_labels(request_data: dict[str, Any]) -> list[str]:
    text = request_data.get("targets", "")
    found = re.findall(r"(?:^|[,;\n])\s*([A-Za-z][A-Za-z0-9_-]*)\s*(?=[,;\n]|$)", text)
    labels = list(dict.fromkeys(found))
    labels.extend(f"x = {x:g}" for x in request_data.get("x_positions", []))
    return labels


def _flatten_selected(measured: dict[str, Any], request_data: dict[str, Any],
                      target_meta: dict[str, Any], comment: str) -> list[dict[str, Any]]:
    rows = []
    for point in measured.get("points", []):
        rows.append({"panel": "P1", "series": point.get("series") or request_data.get("targets") or "requested series",
                     "feature": f"{point['overlay_id']}. {point.get('label')}" if point.get("overlay_id") else point.get("label"), "status": point.get("status"),
                     "x": point.get("x"), "y": point.get("y"),
                     "height_relative": point.get("normalized_height"),
                     "x_uncertainty": point.get("x_uncertainty_transformed"),
                     "y_uncertainty": point.get("y_uncertainty_transformed"),
                     "x_unit": (measured.get("x_axis") or {}).get("unit") or request_data.get("x_unit"),
                     "y_unit": (measured.get("y_axis") or {}).get("unit") or request_data.get("y_unit"),
                     "reason": point.get("reason"), "pixel": point.get("pixel"),
                     "method": point.get("method"), "review_id": point.get("review_id"),
                     "review_note": point.get("review_note")})
    return rows


def _unresolved_measurement(selections: list[dict[str, Any]], image_path: str,
                            inspection: dict[str, Any], reason: str) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    rows = []
    for selection in selections:
        label = str(selection.get("label", "requested feature"))
        rows.append({"label": label, "status": "unresolved", "reason": reason,
                     "assignment_status": "unresolved", "x": None, "y": None,
                     "pixel": None, "pixel_uncertainty": None,
                     "x_uncertainty_transformed": None, "y_uncertainty_transformed": None,
                     "method": "unresolved"})
    if rows:
        draw.rectangle((0, 0, min(image.width, 900), min(32 + len(rows)*17, image.height)), fill="white")
        for i, row in enumerate(rows):
            draw.text((8, 8+i*17), f"Unresolved {row['label']}: x-axis calibration unavailable", fill="#a00000")
    overlay = Path(inspection["overview_path"]).parent / "measured.png"
    image.save(overlay)
    return {"implementation_id": inspection["implementation_id"], "session_id": inspection["session_id"],
            "source_image": image_path, "image_size": inspection["image_size"],
            "x_axis": None, "y_axis": None, "baseline_pixel": None, "reference_label": None,
            "calibration_residual_transformed": {"x": None, "y": None}, "points": rows,
            "overlay_path": str(overlay)}


def _complete_rows(result_path: Path) -> list[dict[str, Any]]:
    result = _read(result_path)
    rows = []
    for panel in result.get("panels", []):
        for series in panel.get("series", []):
            for row in [*series.get("features", []), *series.get("queries", []), *series.get("curve", [])]:
                px, py = row.get("original_pixel_xy", [None, None])
                uncertainty = row.get("calibrated_uncertainty") or {}
                x_interval = uncertainty.get("x_interval") or []
                y_interval = uncertainty.get("y_interval") or []
                rows.append({"panel": panel.get("id"), "series": series.get("description") or series.get("id"),
                             "feature": row.get("label") or row.get("id") or row.get("feature_type"),
                             "status": row.get("status"), "x": row.get("calibrated_x"),
                             "y": row.get("calibrated_y"), "height_relative": None,
                             "x_uncertainty": abs(x_interval[1] - x_interval[0]) / 2 if len(x_interval) == 2 and all(isinstance(x, (int, float)) for x in x_interval) else None,
                             "y_uncertainty": abs(y_interval[1] - y_interval[0]) / 2 if len(y_interval) == 2 and all(isinstance(y, (int, float)) for y in y_interval) else None,
                             "x_unit": row.get("x_unit"), "y_unit": row.get("y_unit"),
                             "reason": row.get("reason"), "pixel": [px, py],
                             "method": row.get("feature_type"), "review_id": None, "review_note": None})
    return rows


def _complete_schema() -> dict[str, Any]:
    number = {"type": "number"}
    pair = {"type": "array", "items": number, "minItems": 2, "maxItems": 2}
    rect = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    tick = {"type": "object", "properties": {"pixel": number, "value": number},
            "required": ["pixel", "value"], "additionalProperties": False}
    segment = {"type": "object", "properties": {
        "id": {"type": "string"}, "pixel_bounds": pair,
        "anchors": {"type": "array", "items": tick, "minItems": 2},
        "checks": {"type": "array", "items": tick},
        "scale": {"type": "string", "enum": ["linear", "log10"]}},
        "required": ["id", "pixel_bounds", "anchors", "checks", "scale"], "additionalProperties": False}
    axis = {"type": "object", "properties": {
        "anchors": {"type": "array", "items": tick}, "checks": {"type": "array", "items": tick},
        "scale": {"type": "string", "enum": ["linear", "log10"]},
        "segments": {"type": "array", "items": segment},
        "unit": {"type": ["string", "null"]}},
        "required": ["anchors", "checks", "scale", "segments", "unit"], "additionalProperties": False}
    axis_or_null = {"anyOf": [axis, {"type": "null"}]}
    series = {"type": "object", "properties": {
        "id": {"type": "string"}, "kind": {"type": "string", "enum": ["curve", "markers"]},
        "description": {"type": "string"}, "requested_target_ids": {"type": "array", "items": {"type": "string"}},
        "bounds": rect, "seeds": {"type": "array", "items": pair},
        "color_rgb": {"anyOf": [{"type": "array", "items": number, "minItems": 3, "maxItems": 3}, {"type": "null"}]},
        "exclude_bounds": {"type": "array", "items": rect},
        "baseline": {"anyOf": [{"type": "object", "properties": {"anchors": {"type": "array", "items": pair, "minItems": 2}}, "required": ["anchors"], "additionalProperties": False}, {"type": "null"}]}},
        "required": ["id", "kind", "description", "requested_target_ids", "bounds", "seeds", "color_rgb", "exclude_bounds", "baseline"],
        "additionalProperties": False}
    panel = {"type": "object", "properties": {
        "id": {"type": "string"}, "bounds": rect, "x_axis": axis_or_null, "y_axis": axis_or_null,
        "series": {"type": "array", "items": series}},
        "required": ["id", "bounds", "x_axis", "y_axis", "series"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "inventory_complete": {"type": "boolean"}, "unresolved_inventory": {"type": "array", "items": {"type": "string"}},
        "panels": {"type": "array", "items": panel}},
        "required": ["inventory_complete", "unresolved_inventory", "panels"], "additionalProperties": False}


def _report_csv(rows: list[dict[str, Any]]) -> str:
    fields = ["panel", "series", "feature", "status", "x", "y", "height_relative",
              "reason", "method", "review_note"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_outputs(folder: Path, rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    with (folder / "table.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        stream.write(_report_csv(rows))
    result["rows"] = rows
    result["csv_path"] = "table.csv"
    recorder = current_timing.get()
    if recorder is not None and recorder.folder == folder:
        result["timing"] = recorder.finish()
    _json(folder / "result.json", result)


async def _run_subprocess(app: Starlette, job_id: str, command: list[str],
                          *, timeout: int = 180, env: dict[str, str] | None = None,
                          cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    label = "checkpoint.validate" if "validate" in command else "checkpoint.extract"
    with timing_span("local_python", label):
        await _emit(app, job_id, "engine_started", "Running local measurement engine")
        proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, cwd=cwd, env=env,
                                                    start_new_session=True)
        app.state.processes[job_id] = proc
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            os.killpg(proc.pid, signal.SIGTERM)
            raise RuntimeError("Local extraction timed out")
        except asyncio.CancelledError:
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGTERM)
            raise
        finally:
            app.state.processes.pop(job_id, None)
        if proc.returncode:
            raise RuntimeError(stderr.decode("utf-8", "replace")[-3000:] or "Local extraction failed")
        return subprocess.CompletedProcess(command, proc.returncode, stdout.decode(), stderr.decode())


async def _complete_extract(app: Starlette, job_id: str, folder: Path,
                            image_path: Path, request_data: dict[str, Any],
                            runner: LunaRunner) -> dict[str, Any]:
    checkpoint_root=Path(__file__).with_name('checkpoints')
    if not checkpoint_root.is_dir():checkpoint_root=ROOT/'benchmarks'/'evolution_v4'
    selection_path = checkpoint_root / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError("The selected complete-curve checkpoint is unavailable")
    selection = _read(selection_path)
    snapshot = checkpoint_root / "snapshots" / selection["snapshot"]
    manifest_path = snapshot / "snapshot.json"
    if _sha(manifest_path) != selection["snapshot_sha256"]:
        raise RuntimeError("Selected experimental checkpoint manifest failed its integrity check")
    manifest = _read(manifest_path)
    for relative, expected in manifest.items():
        if _sha(snapshot / relative) != expected:
            raise RuntimeError(f"Selected experimental checkpoint failed integrity check: {relative}")
    schema = _complete_schema()
    _json(folder / "plan.schema.json", schema)
    prompt = f"""Inventory every visible Cartesian panel and distinct series in the attached image. Use image evidence only; do not use tools or invent values. Return a plan conforming to the supplied schema. Use original image pixels. For each calibrated axis provide two tick anchors and at least one independent check; use null if unsupported. Distinguish markers from fitted curves. Give identity seeds on visible trace ink. Do not use caption or molecular drawing as data. Preserve unresolved panel or series identities explicitly. Include requested target descriptions when clear.

Figure description: {request_data['target']['explanation']}
Channel: {request_data['category']}.
Requested series/features: {request_data.get('targets') or request_data.get('channel_description') or 'all visible Cartesian series'}.
This is the experimental complete-curves workflow. Do not generate dense coordinates; the pinned local engine will measure source pixels.
"""
    plan_path = folder / "plan.json"
    try:
        plan = await runner.call(prompt=prompt, schema=schema, images=[str(image_path)],
                                 workdir=worker_dir(str(folder), "complete-plan"))
    except LunaError as exc:
        raise RuntimeError(str(exc)) from exc
    _json(plan_path, plan)
    command = [sys.executable, "-m", "plot_reader.dispatch", "--profile", "experimental",
               "validate", str(image_path), "--plan", str(plan_path)]
    env = {**os.environ, "PYTHONPATH": str(snapshot)}
    await _job_update(app, job_id, stage="complete_validate")
    validate = await _run_subprocess(app, job_id, command, timeout=180,
                                     env={**os.environ, "PYTHONPATH": str(snapshot)}, cwd=str(snapshot))
    validation = json.loads(validate.stdout)
    if validation.get("errors"):
        await _job_update(app, job_id, stage="complete_plan")
        repair_prompt = prompt + "\nCorrect these local validation errors using only the attached image: " + json.dumps(validation["errors"])
        plan = await runner.call(prompt=repair_prompt, schema=schema, images=[str(image_path)],
                                 workdir=worker_dir(str(folder), "complete-repair"))
        _json(plan_path, plan)
        await _job_update(app, job_id, stage="complete_validate")
        validation = json.loads((await _run_subprocess(app, job_id, command, timeout=180,
                                    env={**os.environ, "PYTHONPATH": str(snapshot)}, cwd=str(snapshot))).stdout)
        if validation.get("errors"):
            _json(folder / "validation.json", validation)
            raise RuntimeError("LUNA's plan remains unsupported after one repair; revise the figure description or target")
    output = folder / "complete-output"
    output.mkdir(exist_ok=True)
    command = [sys.executable, '-m', 'plot_reader.dispatch', '--profile', 'experimental',
               'extract', str(image_path), '--plan', str(plan_path), '--output', str(output)]
    await _job_update(app, job_id, stage="complete_extract")
    await _run_subprocess(app, job_id, command, timeout=900,env=env,cwd=str(snapshot))
    result = _read(output / "result.json")
    result["result_path"] = str(output / "result.json")
    return result


async def _refine_unsupported_centers(runner: LunaRunner, plan: dict[str, Any],
                                      source: Path, folder: Path, require_contours: bool = False) -> dict[str, Any]:
    issues = await run_local(unsupported_centers, plan, source)
    size = Image.open(source).size
    selected = plan["selections"]
    for index, item in enumerate(selected):
        pixel = item.get("pixel")
        if (item.get("coordinate_review") != "uncertain" or item.get("status") == "unresolved" or not pixel
                or any(issue["selection_index"] == index for issue in issues)):
            continue
        # A shared origin/overlap stays uncertain; zooming cannot separate it.
        if any(other is not item and other.get("pixel") and
               abs(other["pixel"][0]-pixel[0]) < 3 and abs(other["pixel"][1]-pixel[1]) < 3 for other in selected):
            continue
        x, y = (int(round(v)) for v in pixel)
        issues.append({"selection_index": index, "label": item["label"], "pixel": pixel,
                       "reason": item.get("coordinate_review_note"),
                       "bounds": [max(0, x-100), max(0, y-120), min(size[0], x+100), min(size[1], y+120)]})
    if not issues:
        return plan
    _json(folder / "pixel-support.json", issues)
    focus = issues[:6]
    evidence = await run_local(coordinate_evidence, source, folder / "focused-evidence",
                                      boxes=[tuple(item["bounds"]) for item in focus], spacing=25)
    candidates = await run_local(focused_marker_candidates, source, evidence)
    _json(folder / "focused-candidates.json", {"candidates": candidates, "evidence": evidence})
    subset = {"selections": [plan["selections"][item["selection_index"]] for item in focus]}
    records = [{**item, "selection_id": str(index)} for index, item in enumerate(subset["selections"], 1)]
    await runner.emit({"type": "stage", "stage": "coordinate_review", "message": "Refining uncertain feature centers in local crops"})
    prompt = f"""Perform one focused source-pixel correction. These selected centers lack nearby source ink or had uncertain identity in the global image review. First image: original source. Remaining images: local crops with ORIGINAL source x/y grid labels, x right and y down. Inspect each requested feature's actual shape/series in the source and local crop. Return each selection_id once using ORIGINAL source pixels, never crop-relative coordinates. Move a center onto the requested visible marker or peak; do not snap to a different series, curve, legend or text. If its intended feature is outside the crop, use the full source to locate it. Use uncertain if identity is ambiguous; keep a visible approximate center when supported. The host recomputes numbers. No inferred trends or numerical data estimates.
Selections: {json.dumps(records, ensure_ascii=False)}
Local evidence: {json.dumps(evidence)}
Locally detected contours: {json.dumps(candidates)}
Purple C IDs identify source-measured contour centers. Prefer candidate_id for the intended visible marker (including hollow triangles, diamonds, squares). The host uses that candidate's exact source pixel, so you must not re-estimate its coordinates. Candidates can include non-data ink or other series; inspect the shape and context before choosing. Crops are enlarged 3x, but blue grid labels and candidate pixels remain ORIGINAL source pixels. Use candidate_id null only when no candidate matches the intended feature.
"""
    try:
        review = await runner.call(prompt=prompt, schema=FOCUSED_REVIEW_SCHEMA,
            images=[str(source)] + [item["path"] for item in evidence],
            workdir=worker_dir(str(folder), "coordinate-refinement"))
        _json(folder / "coordinate-refinement.json", review)
        review = bind_review_candidates(review, candidates, subset["selections"] if require_contours else None)
        revised, _ = apply_selection_review(subset, review, Image.open(source).size)
        for issue, selection in zip(focus, revised["selections"]):
            plan["selections"][issue["selection_index"]] = selection
    except (LunaError, ValueError) as exc:
        _json(folder / "coordinate-refinement-error.json", {"error": str(exc)})
    remaining = await run_local(unsupported_centers, plan, source)
    for item in remaining:
        selection = plan["selections"][item["selection_index"]]
        selection["coordinate_review"] = "uncertain"
        selection["coordinate_review_note"] = (selection.get("coordinate_review_note", "") +
            f" Selected center is {item['ink_distance_pixels']} pixels from the nearest source ink; needs location review.").strip()
    _json(folder / "pixel-support-final.json", remaining)
    return plan


def _timed_job(function):
    @wraps(function)
    async def wrapped(app, job_id, *args, **kwargs):
        folder = _job_dir(job_id)
        parent = current_timing.get()
        if parent is not None and parent.folder == folder:
            return await function(app, job_id, *args, **kwargs)
        recorder = AnalysisTiming(folder)
        token = current_timing.set(recorder)
        try:
            return await function(app, job_id, *args, **kwargs)
        finally:
            try:
                status = _read(folder / "job.json").get("status")
                if status == "needs_input":
                    recorder.set_stage("user_wait")
                    recorder.save()
                else:
                    recorder.finish()
            finally:
                current_timing.reset(token)
    return wrapped


@_timed_job
async def _run_job(app: Starlette, job_id: str, resumed_plan: dict[str, Any] | None = None) -> None:
    folder = _job_dir(job_id)
    job = _read(folder / "job.json")
    request_data = _read(folder / "request.json")
    task_runner: LunaRunner | None = None

    async def emit(event: dict[str, Any]) -> None:
        message = event.get("text", event.get("message", ""))
        await _emit(app, job_id, event.get("type", "progress"), message,
                    **{k: v for k, v in event.items() if k not in {"type", "text", "message"}})

    try:
        _require_no_jev(request_data)
        await _job_update(app, job_id, status="running", stage="prepare")
        await _emit(app, job_id, "stage", "Preparing confirmed figure", stage="prepare")
        image_path = folder / "source-image.png"
        target_meta = await run_local(_page_manifest, request_data, image_path)
        upload_record = _read(_upload_dir(request_data["upload_id"]) / "source.json")
        inspection: dict[str, Any] = {}
        if request_data["mode"] == "selected":
            await _job_update(app, job_id, stage="inspect")
            await _emit(app, job_id, "stage", "Inspecting ticks and feature candidates", stage="inspect")
            if resumed_plan is not None:
                inspection = _read(folder / "inspection.json")
            else:
                labels = _candidate_labels(request_data)
                inspection = await run_local(core.inspect_plot, str(image_path),
                                                      request_data.get("targets") or "requested plot features", labels or None)
                _json(folder / "inspection.json", inspection)
        task_runner = LunaRunner(emit, effort=request_data.get("model_effort"))
        app.state.runners[job_id] = task_runner
        calibration_warnings: list[str] = []
        if request_data["mode"] == "selected":
            evidence = await run_local(coordinate_evidence, image_path, folder / "evidence")
            _json(folder / "evidence.json", evidence)
            review_images = list(dict.fromkeys(x for x in inspection.get("review_image_paths", [])
                                if x != inspection.get("overview_path") and Path(x).is_file()))[:4]
            planning_evidence = evidence if _is_bulk_point_request(request_data) else evidence[:1]
            images = [str(image_path)] + [item["path"] for item in planning_evidence] + review_images
            prompt = _selected_prompt(request_data, inspection) + "\nAttached coordinate evidence: " + json.dumps(planning_evidence)
            plan = resumed_plan
            beta = (request_data.get("scatter_workflow", "legacy") == "beta" and
                    request_data["category"] == "plot" and _is_bulk_point_request(request_data)
                    and not request_data.get("x_positions") and not request_data.get("normalization"))
            if plan is None:
                if beta:
                    plan = await plan_scatter(task_runner, image_path, folder, request_data, inspection)
                else:
                    plan = await task_runner.call(prompt=prompt, schema=_selected_schema(request_data), images=images,
                                                  workdir=worker_dir(str(folder), "selected-plan"))
            if not beta and _is_bulk_point_request(request_data) and _is_aggregate_unresolved(plan):
                await _emit(app, job_id, "stage", "Expanding the request to individual plotted points", stage="inspect")
                repair_prompt = prompt + "\nYour prior plan collapsed the whole request to one unresolved item. Revise it as one selection per visible marker in the requested chart. Use visible marker centers from the image, assign a stable series identity using marker and legend evidence, and use mode exact where local marker refinement is unavailable. Continue with calibrated values even when calibration checks need a warning; do not discard the point selections."
                try:
                    repaired = await task_runner.call(prompt=repair_prompt, schema=_selected_schema(request_data), images=images,
                                                      workdir=worker_dir(str(folder), "selected-repair"))
                    if len(repaired.get("selections", [])) > len(plan.get("selections", [])):
                        plan = repaired
                    else:
                        plan.setdefault("notes", []).append("The point-by-point repair remained ambiguous; the requested points are retained as unresolved.")
                except LunaError as exc:
                    plan.setdefault("notes", []).append(f"The point-by-point repair could not complete; the unresolved item is retained: {exc}")
            plan = _question_policy(unique_selection_labels(plan), request_data)
            if request_data.get("allow_agent_questions", False) and plan.get("clarifications"):
                _json(folder / "plan.json", plan)
                await _job_update(app, job_id, status="needs_input", stage="clarification", plan=plan)
                await _emit(app, job_id, "clarification", "The agent needs your help to confirm the figure interpretation", questions=plan["clarifications"])
                return
            if (request_data.get("allow_agent_questions", False) and request_data.get("normalization")
                    and _normalization_needs_clarification(plan)):
                plan["clarifications"] = [{"id": "normalization-reference",
                    "question": "The image did not clearly establish both a normalization reference and a baseline. Should I measure the original values without normalization, or can you identify the reference and visible baseline?",
                    "choices": ["Keep original values; skip normalization", "I will clarify the reference and baseline"], "region": None}]
                _json(folder / "plan.json", plan)
                await _job_update(app, job_id, status="needs_input", stage="clarification", plan=plan)
                await _emit(app, job_id, "clarification", "Normalization needs confirmation", questions=plan["clarifications"])
                return
            if plan.get("notes"):
                await _emit(app, job_id, "review_note", "Image review notes", notes=plan["notes"])
            if beta and plan.get("scatter_beta") and not plan.get("_beta_grounded"):
                await _job_update(app, job_id, stage="coordinate_review")
                plan = await ground_scatter_beta(task_runner, plan, image_path, folder, request_data)
                plan = _question_policy(plan, request_data)
            await _job_update(app, job_id, stage="measure", plan=plan)
            if plan.get("x_axis") is None:
                measured = _unresolved_measurement(plan["selections"], str(image_path), inspection,
                                                   "x-axis calibration is unsupported by the visible image")
            else:
                x_axis = {k: v for k, v in plan["x_axis"].items() if k in {"anchors", "checks", "scale"}}
                y_axis = ({k: v for k, v in plan["y_axis"].items() if k in {"anchors", "checks", "scale"}}
                          if plan.get("y_axis") else None)
                request_data["x_unit"] = plan["x_axis"].get("unit")
                request_data["y_unit"] = plan.get("y_axis", {}).get("unit") if plan.get("y_axis") else None
                _json(folder / "request.json", request_data)
                measured = await run_local(measure_selected, inspection["session_id"],
                                                   x_axis, plan["selections"], y_axis,
                                                   plan.get("reference_label") if request_data.get("normalization") else None,
                                                   plan.get("baseline_pixel") if request_data.get("normalization") else None,
                                                   False)
                _json(folder / "draft-measurement.json", measured)
                if plan.get("_beta_grounded"):
                    coordinate_warnings = [f"{item['label']}: {item.get('coordinate_review_note', '')}"
                        for item in plan['selections'] if item.get('coordinate_review') == 'uncertain']
                    measured['inventory_complete'] = plan.get('inventory_complete', False)
                    measured['unresolved_inventory'] = plan.get('unresolved_inventory', [])
                    coordinate_warnings.extend(measured['unresolved_inventory'])
                else:
                    await _job_update(app, job_id, stage="coordinate_review")
                    await _emit(app, job_id, "stage", "Checking every selected pixel against the source", stage="coordinate_review")
                    draft_points = {item["label"]: item for item in measured["points"]}
                    review_records = [{**item, "selection_id": str(index),
                                       "pixel": draft_points.get(item["label"], {}).get("pixel") or item.get("pixel")}
                                      for index, item in enumerate(plan["selections"], 1)]
                    draft_overlay = folder / "draft-selections.png"
                    await run_local(selection_overlay, image_path, review_records, draft_overlay)
                    reviewed_evidence = await run_local(coordinate_evidence, image_path, folder / "review-evidence")
                    review_tiles = reviewed_evidence[1:] or reviewed_evidence
                    source_candidates = await run_local(focused_marker_candidates, image_path, review_tiles, 1)
                    _json(folder / "review-candidates.json", {"candidates": source_candidates, "evidence": review_tiles})
                    coordinate_prompt = f"""Audit EACH numbered marker location against the unannotated source and coordinate-grid crops. The second image is a DRAFT overlay with selection IDs, not ground truth. Blue grid labels are ORIGINAL source pixels (x right, y DOWN); never return crop-relative coordinates. Automatic candidates and plot bounds may be wrong. Legend entries, axis ink, text, blank space, and fitted-curve ink are not experimental marker centers. Identify the requested marker SHAPE and series before choosing its center. Inspect every item independently; shared time positions do not establish y. Correct obvious mislocations by reading the actual marker center against the source grid, including points mistakenly over the legend. Do not interpolate a curve or invent a point at an assumed time. Preserve overlapping-marker uncertainty; an approximate visible center may be retained with verdict uncertain and a clear reason. Return each selection_id exactly once. confirmed/corrected require the reviewed ORIGINAL pixel. uncertain may use a supported approximate pixel or null if no position can be read. Do not change data values or calibrations; the host computes numbers from reviewed pixels.
    User target: {request_data['target']['explanation']}
    Feature channel: {request_data['category']}; {request_data.get('channel_description')}. For spectra, audit visible peak apices instead of marker shapes. For curve queries, audit the intended trace at x_value; the host preserves that requested x position. A fitted curve is appropriate only when the user requested curve values.
    Selections: {json.dumps(review_records, ensure_ascii=False)}
    Images after source and draft overlay: {json.dumps(review_tiles)}
    Source-measured contour candidates: {json.dumps(source_candidates)}
    Purple C IDs in the crop images mark source-measured contour centers. For hollow square, triangle, diamond, or circle markers, choose the matching candidate_id whenever visible; the host substitutes its exact source pixel. Do not re-estimate that candidate's numerical coordinates. A candidate can also be text or a legend item, so verify shape, series and plot context. When no candidate supports a hollow marker, use candidate_id null and verdict uncertain; preserve an approximate pixel with a warning. For plus/cross markers and curve/peak queries, manual source pixels are allowed where connected ink prevents a bounded contour.
    """
                    try:
                        if _is_bulk_point_request(request_data) and request_data["category"] == "plot":
                            plan = await ground_scatter(task_runner, plan, image_path, folder, request_data['target']['explanation'])
                        else:
                            review = await task_runner.call(prompt=coordinate_prompt, schema=FOCUSED_REVIEW_SCHEMA,
                                images=[str(image_path), str(draft_overlay)] + [item["path"] for item in review_tiles],
                                workdir=worker_dir(str(folder), "coordinate-review"))
                            _json(folder / "coordinate-review.json", review)
                            require_contours = request_data["category"] == "plot" and (
                                "scatter" in request_data.get("channel_description", "").lower() or _is_bulk_point_request(request_data))
                            review = bind_review_candidates(review, source_candidates, plan["selections"] if require_contours else None,
                                                            max_shift=60 if require_contours else None)
                            plan, coordinate_warnings = apply_selection_review(plan, review, tuple(inspection["image_size"]))
                            plan = await _refine_unsupported_centers(task_runner, plan, image_path, folder, require_contours)
                        coordinate_warnings = [f"{item['label']}: {item.get('coordinate_review_note', '')}"
                                               for item in plan["selections"] if item.get("coordinate_review") == "uncertain"]
                        measured = await run_local(measure_selected, inspection["session_id"],
                            x_axis, plan["selections"], y_axis,
                            plan.get("reference_label") if request_data.get("normalization") else None,
                            plan.get("baseline_pixel") if request_data.get("normalization") else None,
                            False)
                    except (LunaError, ValueError) as exc:
                        coordinate_warnings = [f"Pixel review could not complete: {exc}. Selected measurements are retained for review."]
                        for selection in plan["selections"]:
                            selection["coordinate_review"] = "uncertain"
                            selection["coordinate_review_note"] = coordinate_warnings[0]
                measured["coordinate_review_warnings"] = coordinate_warnings
                if coordinate_warnings:
                    await _emit(app, job_id, "review_note", "Pixel assignments needing review", notes=coordinate_warnings)
            selections_by_label = {item["label"]: item for item in plan["selections"]}
            for index, point in enumerate(measured.get("points", []), 1):
                selection = selections_by_label.get(point["label"], {})
                point.update(series=selection.get("series", "Series"), overlay_id=index,
                             coordinate_review=selection.get("coordinate_review"),
                             coordinate_review_note=selection.get("coordinate_review_note"))
                if selection.get("coordinate_review") == "uncertain" and point.get("status") == "resolved":
                    point.update(status="needs_review", reason=selection.get("coordinate_review_note"))
                point["review_note"] = selection.get("coordinate_review_note") or point.get("review_note")
                if plan.get("unanswered_clarifications") and point.get("status") == "resolved":
                    point.update(status="needs_review", reason="Interpretation needs review because extraction continued without clarification.")
            measured["question_policy_warnings"] = plan.get("question_policy_warnings", [])
            _json(folder / "plan.json", plan)
            await _job_update(app, job_id, plan=plan)
            calibration_warnings = measured.get("calibration_warnings", [])
            if calibration_warnings:
                await _emit(app, job_id, "review_note", "Axis calibration warning · measurements continued", notes=calibration_warnings)
            _json(folder / "engine-result.json", measured)
            rows = _flatten_selected(measured, request_data, target_meta, "")
            overlay = folder / "annotated.png"
            await run_local(selection_overlay, image_path, measured.get("points", []), overlay)
        else:
            await _job_update(app, job_id, stage="complete_plan")
            await _emit(app, job_id, "stage", "Planning all panels and series", stage="complete_plan")
            measured = await _complete_extract(app, job_id, folder, image_path, request_data, task_runner)
            _json(folder / "engine-result.json", measured)
            rows = _complete_rows(Path(measured["result_path"]))
            overlay = Path(measured["overlay_path"])
        retained_overlay = folder / "annotated.png"
        if overlay.resolve() != retained_overlay.resolve():
            shutil.copy2(overlay, retained_overlay)
        overlay = retained_overlay
        await _job_update(app, job_id, stage="review")
        await _emit(app, job_id, "stage", "Reviewing measurements and writing explanation", stage="review")
        # The measured table is the numerical source of truth. LUNA's review
        # can flag assignments for human attention but cannot change any value.
        explanation_schema = COMMENT_SCHEMA
        explain_prompt = f"""Write a concise scientific explanation of this image-based plot digitisation. State the target, what the local pixel measurement engine calibrated and measured, whether any normalization was used, and key unresolved items/limitations. Only state facts supported by the result or source preview. You may list review_issues when the assignments or image evidence look questionable. Do not change or restate the table as if values were certain.

Request: {request_data['target']['explanation']}
Channel: {request_data['category']}.
Mode: {request_data['mode']}.
Normalization requested: {request_data['normalization']}.
User questions allowed: {request_data.get('allow_agent_questions', False)}. Write an explanation, not follow-up questions.
Question policy warnings: {json.dumps(measured.get('question_policy_warnings', []), ensure_ascii=False)}
Calibration warnings: {json.dumps(measured.get('calibration_warnings', []), ensure_ascii=False)}
Pixel assignment warnings: {json.dumps(measured.get('coordinate_review_warnings', []), ensure_ascii=False)}
Rows/status: {json.dumps(rows[:120], ensure_ascii=False)}
Engine summary: {json.dumps({k: measured.get(k) for k in ('x_axis','y_axis','baseline_pixel','reference_label','calibration_residual_transformed','calibration_warnings','inventory_complete','unresolved_inventory','resolved_rows','rows') if k in measured}, ensure_ascii=False)}
"""
        try:
            narrative = await task_runner.call(prompt=explain_prompt, schema=explanation_schema,
                                               images=[str(image_path), str(overlay)],
                                               workdir=worker_dir(str(folder), "explanation"))
            comment = narrative.get("comment", "")
            if narrative.get("review_issues"):
                await _emit(app, job_id, "review_note", "Review flagged items for attention", notes=narrative["review_issues"])
        except LunaError as exc:
            comment = f"Extraction completed. The local measurement engine returned {sum(r['status'] == 'resolved' for r in rows)} resolved and {sum(r['status'] != 'resolved' for r in rows)} unresolved rows. Explanation generation failed: {exc}"
        narrative_issues = narrative.get("review_issues", []) if "narrative" in locals() else []
        result = {"id": job_id, "status": "complete", "experimental": request_data["mode"] == "complete",
                  "request": request_data, "source": {"name": upload_record["original_name"],
                      "sha256": upload_record["sha256"], **target_meta},
                  "engine": job.get("engine"), "comment": comment, "inference_profile": task_runner.profile,
                  "review_issues": list(dict.fromkeys([*calibration_warnings, *measured.get("coordinate_review_warnings", []), *measured.get("question_policy_warnings", []), *narrative_issues])),
                  "measurement": {k: measured.get(k) for k in ("implementation_id", "x_axis", "y_axis", "baseline_pixel", "reference_label", "calibration_residual_transformed", "calibration_warnings", "coordinate_review_warnings", "question_policy_warnings", "inventory_complete", "unresolved_inventory") if k in measured},
                  "overlay_path": str(overlay.relative_to(folder)), "completed_at": _now()}
        _write_outputs(folder, rows, result)
        await _job_update(app, job_id, status="complete", stage="complete", result_ready=True)
        await _emit(app, job_id, "complete", "Extraction is ready", row_count=len(rows))
    except asyncio.CancelledError:
        if task_runner:
            await task_runner.cancel()
        await _job_update(app, job_id, status="cancelled", stage="cancelled")
        await _emit(app, job_id, "cancelled", "Extraction cancelled; saved results are retained")
    except Exception as exc:
        current = _read(folder / "job.json")
        if current.get("status") == "cancel_requested":
            await _job_update(app, job_id, status="cancelled", stage="cancelled", error="")
            await _emit(app, job_id, "cancelled", "Extraction cancelled; saved results are retained")
        else:
            await _job_update(app, job_id, status="failed", stage="failed", error=str(exc)[:2000])
            await _emit(app, job_id, "failed", str(exc)[:1000])
    finally:
        app.state.runners.pop(job_id, None)
        app.state.tasks.pop(job_id, None)


async def job_status(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        job = _read(_job_dir(request.path_params["job_id"]) / "job.json")
        return JSONResponse(job)
    except (ValueError, FileNotFoundError):
        return _form_error("Run not found", 404)


async def answer_clarification(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        job_id = _safe_id(request.path_params["job_id"])
        folder = _job_dir(job_id)
        job = _read(folder / "job.json")
        if job.get("status") != "needs_input":
            raise ValueError("This run is not waiting for a clarification")
        body = await request.json()
        question_id = str(body.get("question_id", ""))
        answer = str(body.get("answer", "")).strip()
        if not answer and not body.get("skip"):
            raise ValueError("Choose an option or enter a short clarification")
        count = int(job.get("clarification_count", 0))
        if count >= 4:
            raise ValueError("This run has reached the clarification limit. Restart with a narrower description.")
        plan = job.get("plan", {})
        question = next((x for x in plan.get("clarifications", []) if x.get("id") == question_id), None)
        if not question:
            raise ValueError("Clarification item not found")
        history = job.get("answers", [])
        history.append({"question": question["question"], "answer": answer or "Skipped; return unresolved"})
        _json(folder / "answers.json", history)
        await _job_update(request.app, job_id, status="queued", stage="clarification_answered",
                          answers=history, clarification_count=count + 1)
        task = asyncio.create_task(_resume_selected(request.app, job_id, question, answer, bool(body.get("skip"))))
        request.app.state.tasks[job_id] = task
        return JSONResponse({"id": job_id, "status": "queued"}, status_code=202)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _form_error(str(exc))


@_timed_job
async def _resume_selected(app: Starlette, job_id: str, question: dict[str, Any],
                           answer: str, skip: bool) -> None:
    folder = _job_dir(job_id)
    job = _read(folder / "job.json")
    data = _read(folder / "request.json")
    plan = job.get("plan", {})
    inspection = _read(folder / "inspection.json")
    image_path = folder / "source-image.png"

    async def emit(event: dict[str, Any]) -> None:
        await _emit(app, job_id, event.get("type", "progress"), event.get("message", event.get("text", "")))

    runner = LunaRunner(emit, effort=data.get("model_effort"))
    app.state.runners[job_id] = runner
    try:
        _require_no_jev(data)
        await _job_update(app, job_id, status="running", stage="clarification")
        if not skip:
            text = f"Question {question['id']}: {question['question']}\nUser answer: {answer}"
            evidence = await run_local(coordinate_evidence, image_path, folder / "evidence")
            if plan.get('scatter_beta'):
                plan = await plan_scatter(runner, image_path, folder, data, inspection, text)
            else:
                plan = await runner.call(prompt=_selected_prompt(data, inspection, text) + "\nCoordinate evidence: " + json.dumps(evidence), schema=_selected_schema(data),
                                         images=[str(image_path)] + [item["path"] for item in evidence],
                                         workdir=worker_dir(str(folder), f"clarification-{job.get('clarification_count', 1)}"))
            if data.get("allow_agent_questions", False) and plan.get("clarifications"):
                await _job_update(app, job_id, status="needs_input", stage="clarification", plan=plan)
                await _emit(app, job_id, "clarification", "One more image choice needs your confirmation", questions=plan["clarifications"])
                return
        else:
            plan["clarifications"] = []
            if question.get("id") == "normalization-reference":
                data["normalization"] = False
                _json(folder / "request.json", data)
            else:
                plan["selections"].append({"label": f"Unresolved: {question['question'][:55]}", "series": "Unresolved", "mode": "exact", "candidate_id": None,
                                           "pixel": None, "x_value": None, "review_id": None, "review_note": None,
                                           "status": "unresolved", "reason": "user skipped clarification"})
        await _run_job(app, job_id, plan)
    except Exception as exc:
        await _job_update(app, job_id, status="failed", stage="failed", error=str(exc)[:2000])
        await _emit(app, job_id, "failed", str(exc)[:1000])
    finally:
        app.state.runners.pop(job_id, None)
        app.state.tasks.pop(job_id, None)


async def cancel_job(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        job_id = _safe_id(request.path_params["job_id"])
        await _job_update(request.app, job_id, status="cancel_requested", stage="cancel_requested")
        runner = request.app.state.runners.get(job_id)
        if runner:
            await runner.cancel()
        proc = request.app.state.processes.get(job_id)
        if proc and proc.returncode is None:
            os.killpg(proc.pid, signal.SIGTERM)
        task = request.app.state.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        else:
            await _job_update(request.app, job_id, status="cancelled", stage="cancelled")
            await _emit(request.app, job_id, "cancelled", "Extraction cancelled; saved results are retained")
        return JSONResponse({"id": job_id, "status": "cancel_requested"})
    except (ValueError, FileNotFoundError):
        return _form_error("Run not found", 404)


async def restart_job(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        if any(not task.done() for task in request.app.state.tasks.values()):
            return _form_error("Another extraction is active; wait or cancel it first", 409)
        old_id = _safe_id(request.path_params["job_id"])
        old = _read(_job_dir(old_id) / "job.json")
        if old.get("status") not in {"interrupted", "failed", "cancelled", "complete"}:
            raise ValueError("Only completed or stopped runs can be restarted")
        data = _read(_job_dir(old_id) / "request.json")
        _require_no_jev(data)
        source = _read(_upload_dir(data["upload_id"]) / "source.json")
        target = _read(_upload_dir(data["upload_id"]) / "target.json")
        if source["sha256"] != data["source_sha256"] or target.get("upload_id") != data["upload_id"]:
            raise ValueError("Saved source or confirmed target has changed")
        new_id = uuid.uuid4().hex
        folder = _job_dir(new_id)
        folder.mkdir(parents=True, exist_ok=True)
        _json(folder / "request.json", data)
        resumed_plan = None
        if data.get("mode") == "selected" and old.get("plan"):
            try:
                inspection = _read(_job_dir(old_id) / "inspection.json")
                await run_local(core._read_session, inspection["session_id"])
                _json(folder / "inspection.json", inspection)
                resumed_plan = old["plan"]
            except (ValueError, KeyError, OSError):
                pass
        _json(folder / "job.json", {"id": new_id, "status": "queued", "created_at": _now(),
                                    "stage": "queued", "request": data,
                                    "engine": old.get("engine"), "restarted_from": old_id})
        await _emit(request.app, new_id, "queued", "Explicitly restarted from saved source")
        task = asyncio.create_task(_run_job(request.app, new_id, resumed_plan))
        request.app.state.tasks[new_id] = task
        return JSONResponse({"id": new_id, "status": "queued", "restarted_from": old_id}, status_code=202)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _form_error(str(exc))


async def correct_points(request: Request) -> Response:
    """Explicit local edits create a new measured revision; no inference runs."""
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    correction_token = None
    try:
        if any(not task.done() for task in request.app.state.tasks.values()):
            return _form_error("Wait for the active extraction before saving corrections", 409)
        old_id = _safe_id(request.path_params["job_id"])
        old_folder = _job_dir(old_id)
        old = _read(old_folder / "job.json")
        data = _read(old_folder / "request.json")
        if old.get("status") != "complete" or data.get("mode") != "selected":
            raise ValueError("Point corrections require a completed selected-measurement run")
        payload = await request.json()
        edits = payload.get("edits", [])
        additions = payload.get("additions", [])
        if not isinstance(edits, list) or not isinstance(additions, list) or not (edits or additions):
            raise ValueError("Choose at least one point to correct or add")
        plan = _read(old_folder / "plan.json")
        old_measured = _read(old_folder / "engine-result.json")
        width, height = Image.open(old_folder / "source-image.png").size
        labels = {p['label']: p for p in plan['selections']}
        seen = set()
        omitted = set()
        for edit in edits:
            index = edit.get('index')
            if type(index) is not int or index < 0 or index >= len(old_measured['points']) or index in seen:
                raise ValueError('Invalid or duplicate measurement row')
            seen.add(index)
            pixel = edit.get('pixel')
            if pixel is not None and (not isinstance(pixel, list) or len(pixel)!=2 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in pixel) or
                not (0 <= pixel[0] < width and 0 <= pixel[1] < height)):
                raise ValueError('Correction must be inside the original source image')
            selection = labels[old_measured['points'][index]['label']]
            if edit.get('omit') is True:
                omitted.add(selection['label'])
                continue
            if selection.get('mode') == 'curve':
                raise ValueError('Curve queries must be rerun; the requested x position is fixed')
            selection.update(pixel=pixel, candidate_id=None, mode='exact',
                status='selected' if pixel is not None else 'unresolved',
                reason=None if pixel is not None else 'Explicit review marked this feature unresolved',
                coordinate_review='corrected' if pixel is not None and not edit.get('uncertain') else 'uncertain',
                coordinate_review_note=str(edit.get('note') or ('Center placed during explicit source-image review' if pixel is not None else 'Explicit review marked this feature unresolved'))[:1500],
                review_id=None, review_note='Explicit source-image correction')
            selection.pop('grounding', None)
        known_series={p.get('series','Series') for p in plan['selections']}
        plan['selections']=[p for p in plan['selections'] if p['label'] not in omitted]
        for addition in additions:
            pixel=addition.get('pixel');label=addition.get('label');series=addition.get('series')
            if (not isinstance(label,str) or not label.strip() or len(label)>160 or label in labels or series not in known_series):
                raise ValueError('Added points need a unique feature label and an existing series')
            if (not isinstance(pixel,list) or len(pixel)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pixel)
                or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)):
                raise ValueError('Added point must be inside the original source image')
            selection={'label':label,'series':series,'pixel':pixel,'candidate_id':None,'mode':'exact',
                'status':'selected','reason':None,'review_id':None,'review_note':'Explicit source-image addition',
                'coordinate_review':'uncertain' if addition.get('uncertain') else 'corrected','coordinate_review_note':str(addition.get('note') or 'Visible marker added during explicit source-image review')[:1500]}
            plan['selections'].append(selection);labels[label]=selection
        if not plan['selections']: raise ValueError('A result must retain at least one feature')
        new_id = uuid.uuid4().hex
        folder = _job_dir(new_id)
        recorder = AnalysisTiming(folder)
        correction_token = current_timing.set(recorder)
        recorder.set_stage("correct_points")
        inspection = _read(old_folder / 'inspection.json')
        await run_local(core._read_session, inspection['session_id'])
        measured = await run_local(measure_selected, inspection['session_id'], plan['x_axis'],
            plan['selections'], plan.get('y_axis'), plan.get('reference_label') if data.get('normalization') else None,
            plan.get('baseline_pixel') if data.get('normalization') else None, False)
        for i, point in enumerate(measured['points'], 1):
            selection = labels[point['label']]
            point.update(series=selection.get('series','Series'),overlay_id=i,
                coordinate_review=selection.get('coordinate_review'),review_note=selection.get('coordinate_review_note'))
            if selection.get('coordinate_review')=='uncertain' and point.get('status')=='resolved':
                point.update(status='needs_review',reason=selection.get('coordinate_review_note'))
        warnings = [p.get('coordinate_review_note','Uncertain feature') for p in plan['selections'] if p.get('coordinate_review')=='uncertain']
        measured['coordinate_review_warnings'] = warnings
        shutil.copy2(old_folder / 'source-image.png', folder / 'source-image.png')
        _json(folder/'inspection.json', inspection); _json(folder/'request.json', data)
        _json(folder/'plan.json', plan); _json(folder/'engine-result.json', measured)
        _json(folder/'corrections.json', {'parent':old_id,'edits':edits,'additions':additions,'inference_used':False})
        local_work(selection_overlay,folder/'source-image.png', measured['points'], folder/'annotated.png')
        previous = _read(old_folder/'result.json')
        rows = _flatten_selected(measured,data,previous.get('source',{}),'')
        comment = (f"Saved {len(edits)} source-image corrections and {len(additions)} added markers. The frozen measurement engine recalculated the table from the saved source pixels using the existing "
            f"{plan['x_axis'].get('scale','linear')} x-axis and {(plan.get('y_axis') or {}).get('scale','unscaled')} y-axis calibration. "
            + ("Baseline-relative normalization uses the saved reference feature. " if data.get('normalization') else "Normalization is off. ")
            + f"{sum(r['status']!='resolved' for r in rows)} rows still need review; unavailable values remain blank. No new model inference was used.")
        result = {**previous,'id':new_id,'completed_at':_now(),'comment':comment,'overlay_path':'annotated.png',
            'corrected_from':old_id,'review_issues':[*measured.get('calibration_warnings',[]),*warnings],
            'measurement':{k:measured.get(k) for k in ('implementation_id','x_axis','y_axis','baseline_pixel','reference_label','calibration_warnings','calibration_residual_transformed','coordinate_review_warnings')}}
        _write_outputs(folder,rows,result)
        _json(folder/'job.json',{'id':new_id,'status':'complete','stage':'complete','created_at':_now(),
            'request':data,'engine':old.get('engine'),'plan':plan,'result_ready':True,'corrected_from':old_id,
            'display_name':'Reviewed source points'})
        return JSONResponse({'id':new_id,'status':'complete','corrected_from':old_id},status_code=201)
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as exc:
        return _form_error(str(exc))
    finally:
        if correction_token is not None:
            try:
                recorder.finish()
            finally:
                current_timing.reset(correction_token)


async def history(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    records = []
    for path in sorted((DATA_ROOT / "jobs").glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = _read(path)
            record = {k: job.get(k) for k in ("id", "status", "created_at", "updated_at", "stage", "engine", "restarted_from")}
            record["display_name"] = job.get("request", {}).get("target", {}).get("explanation") or "Saved extraction"
            records.append(record)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return JSONResponse(records[:100])


async def result(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        folder = _job_dir(request.path_params["job_id"])
        data = _read(folder / "result.json")
        if "inference_profile" not in data:
            saved_profiles = sorted((folder / "worker").glob("*/inference.json"))
            for saved_profile in saved_profiles:
                try:
                    data["inference_profile"] = _read(saved_profile)
                    break
                except (OSError, json.JSONDecodeError):
                    continue
        return JSONResponse(data)
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        return _form_error("Result not ready", 404)


async def artifact(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        folder = _job_dir(request.path_params["job_id"])
        name = request.path_params["name"]
        allowed = {"table.csv": "text/csv", "result.json": "application/json",
                   "timing.json": "application/json",
                   "engine-result.json": "application/json", "plan.json": "application/json",
                   "overlay.png": "image/png", "source-image.png": "image/png"}
        if name not in allowed:
            raise ValueError("Artifact is not available")
        if name == "table.csv":
            rows = _read(folder / "result.json")["rows"]
            return Response("\ufeff" + _report_csv(rows), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="table.csv"',
                                     "Cache-Control": "no-store"})
        path = folder / name
        if name == "overlay.png":
            path = folder / _read(folder / "result.json")["overlay_path"]
        if not path.is_file():
            raise FileNotFoundError
        return FileResponse(path, media_type=allowed[name], filename=name,
                            headers={"Cache-Control": "no-store"})
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
        return _form_error("Artifact not found", 404)


async def region_preview(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        folder = _job_dir(request.path_params["job_id"])
        job = _read(folder / "job.json")
        question = next((item for item in job.get("plan", {}).get("clarifications", [])
                         if item.get("id") == request.path_params["question_id"]), None)
        if not question or not question.get("region"):
            return _form_error("No review region is available", 404)
        bounds = question["region"]
        image_path = folder / "source-image.png"
        with Image.open(image_path) as source:
            width, height = source.size
            if len(bounds) != 4 or not all(isinstance(v, (int, float)) for v in bounds):
                raise ValueError("Invalid review region")
            x0, y0, x1, y1 = [round(v) for v in bounds]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("Review region is outside the confirmed figure")
            pad = max(20, round(min(x1-x0, y1-y0) * .4))
            crop = source.crop((max(0,x0-pad), max(0,y0-pad), min(width,x1+pad), min(height,y1+pad))).convert("RGB")
            draw = ImageDraw.Draw(crop)
            draw.rectangle((min(pad,x0), min(pad,y0), min(crop.width-1,pad+x1-x0), min(crop.height-1,pad+y1-y0)), outline="#ff3030", width=3)
            dest = folder / "review-region.png"
            crop.save(dest)
        return FileResponse(dest, media_type="image/png", headers={"Cache-Control":"no-store"})
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _form_error(str(exc), 404)


async def events(request: Request) -> Response:
    if not await _auth_route(request):
        return _form_error("Local session expired", 403)
    try:
        job_id = _safe_id(request.path_params["job_id"])
        folder = _job_dir(job_id)
        job = _read(folder / "job.json")
        async def stream():
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            app = request.app
            app.state.subscribers.setdefault(job_id, []).append(queue)
            events_file = folder / "events.jsonl"
            if events_file.is_file():
                for line in events_file.read_text().splitlines():
                    yield "data: " + line + "\n\n"
            yield "data: " + json.dumps({"type": "snapshot", "status": job.get("status"),
                                          "stage": job.get("stage")}) + "\n\n"
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                        yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
                        if item.get("type") in {"complete", "failed", "cancelled", "clarification"}:
                            break
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                app.state.subscribers.get(job_id, []).remove(queue)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        return _form_error("Run not found", 404)


async def _startup(app: Starlette) -> None:
    for path in (DATA_ROOT / "sources", DATA_ROOT / "jobs"):
        path.mkdir(parents=True, exist_ok=True)
    for path in (DATA_ROOT / "jobs").glob("*/job.json"):
        try:
            job = _read(path)
            if job.get("status") in {"queued", "running", "needs_input", "cancel_requested"}:
                job.update(status="interrupted", stage="interrupted", error="Server restarted; start a new run explicitly", updated_at=_now())
                _json(path, job)
        except (OSError, json.JSONDecodeError):
            pass
    app.state.figure_tasks={}
    for path in (DATA_ROOT/"sources").glob("*/figure-intake.json"):
        try:
            intake=_read(path)
            if intake.get("status")=="processing":
                intake.update(status="error",stage="interrupted",error="The portal restarted during figure discovery. Retry the selected pages.")
                _json(path,intake)
        except (OSError,json.JSONDecodeError):
            pass
    if os.environ.get("PLOT_READER_OPEN_BROWSER") == "1":
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{app.state.port}/?token={app.state.token}")


routes = [
    Route("/", index),
    Route("/api/upload", upload, methods=["POST"]),
    Route("/api/source/{upload_id:str}/figures",pdf_figures,methods=["GET","POST"]),
    Route("/api/source/{upload_id:str}/figures/{run_id:str}/{figure_id:str}",pdf_figure_preview),
    Route("/api/source/{upload_id:str}/preview", preview),
    Route("/api/source/{upload_id:str}/preview/{page:int}", preview),
    Route("/api/render-page", render_page, methods=["POST"]),
    Route("/api/target", confirm_target, methods=["POST"]),
    Route("/api/jobs", create_job, methods=["POST"]),
    Route("/api/jobs", history, methods=["GET"]),
    Route("/api/jobs/{job_id:str}", job_status),
    Route("/api/jobs/{job_id:str}/events", events),
    Route("/api/jobs/{job_id:str}/answer", answer_clarification, methods=["POST"]),
    Route("/api/jobs/{job_id:str}/cancel", cancel_job, methods=["POST"]),
    Route("/api/jobs/{job_id:str}/restart", restart_job, methods=["POST"]),
    Route("/api/jobs/{job_id:str}/corrections", correct_points, methods=["POST"]),
    Route("/api/jobs/{job_id:str}/result", result),
    Route("/api/jobs/{job_id:str}/region/{question_id:str}", region_preview),
    Route("/api/jobs/{job_id:str}/artifact/{name:str}", artifact),
    Mount("/assets", app=StaticFiles(directory=WEB_ROOT), name="assets"),
]
@asynccontextmanager
async def _lifespan(application: Starlette):
    await _startup(application)
    try:
        yield
    finally:
        tasks=list(getattr(application.state,"figure_tasks",{}).values())
        for task in tasks:
            if not task.done(): task.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)


app = Starlette(routes=routes, lifespan=_lifespan)
app.state.token = ""
app.state.tasks = {}
app.state.runners = {}
app.state.processes = {}
app.state.subscribers = {}


def main() -> None:
    port = int(os.environ.get("PLOT_READER_WEB_PORT", "8765"))
    token = secrets.token_urlsafe(32)
    app.state.token = token
    app.state.port = port
    url = f"http://127.0.0.1:{port}/?token={token}"
    print(f"Plot Reader is running locally: {url}", flush=True)
    print("Keep this terminal open while using the tool. Press Ctrl-C to stop.", flush=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
