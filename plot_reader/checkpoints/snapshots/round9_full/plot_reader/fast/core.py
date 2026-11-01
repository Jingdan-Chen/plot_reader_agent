from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

SESSION_ROOT = Path(tempfile.gettempdir()) / "plot-reader-sessions"
NUMBER = re.compile(r"^[+−-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$")


def implementation_id() -> str:
    """Identify actual loaded engine/interface bytes, independent of Git metadata."""
    digest = hashlib.sha256()
    for name in ("core.py", "cli.py", "server.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()[:20]


class UnresolvedFeature(ValueError):
    """A valid request whose requested feature is not supported by the image."""


@dataclass(frozen=True)
class Axis:
    anchors: list[dict[str, float]]
    scale: str = "linear"
    checks: list[dict[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.scale not in {"linear", "log10"}:
            raise ValueError("axis scale must be linear or log10")
        if len(self.anchors) != 2:
            raise ValueError("exactly two anchors are required")
        pixels = [float(a["pixel"]) for a in self.anchors]
        values = [float(a["value"]) for a in self.anchors]
        if not all(math.isfinite(v) for v in pixels + values):
            raise ValueError("axis anchors must be finite")
        if abs(pixels[1] - pixels[0]) < 10:
            raise ValueError("axis anchors are too close")
        if self.scale == "log10" and any(v <= 0 for v in values):
            raise ValueError("log10 anchors must be positive")
        if values[0] == values[1]:
            raise ValueError("axis values must differ")

    def transform(self, value: float) -> float:
        return math.log10(value) if self.scale == "log10" else value

    def convert(self, pixel: float) -> float:
        a, b = self.anchors
        t0, t1 = self.transform(float(a["value"])), self.transform(float(b["value"]))
        t = t0 + (pixel - float(a["pixel"])) * (t1 - t0) / (float(b["pixel"]) - float(a["pixel"]))
        return 10 ** t if self.scale == "log10" else t

    def pixel_for(self, value: float) -> float:
        if self.scale == "log10" and value <= 0:
            raise ValueError("log10 query must be positive")
        a, b = self.anchors
        t0, t1 = self.transform(float(a["value"])), self.transform(float(b["value"]))
        return float(a["pixel"]) + (self.transform(value)-t0) * (float(b["pixel"])-float(a["pixel"])) / (t1-t0)

    def transformed_error(self, pixel_error: float) -> float:
        a, b = self.anchors
        return abs(pixel_error * (self.transform(float(b["value"])) - self.transform(float(a["value"]))) / (float(b["pixel"]) - float(a["pixel"])))

    def residual(self) -> float | None:
        if not self.checks:
            return None
        return max(abs(self.transform(self.convert(float(c["pixel"]))) - self.transform(float(c["value"]))) for c in self.checks)


def _path(session_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("invalid session ID")
    path = SESSION_ROOT / session_id
    if not (path / "session.json").is_file():
        raise FileNotFoundError("plot session not found")
    return path


def _load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _ocr(image: Image.Image) -> list[dict[str, Any]]:
    # Sparse-text OCR works well on large scanned figures. Small exported plots
    # have thin, tiny labels; enlarging them before page-layout OCR recovers the
    # feature letters and tick labels while all returned boxes stay in source
    # pixels.
    scale = 3 if image.width < 950 and image.height < 750 else 2 if image.width < 1200 and image.height < 900 else 1
    ocr_image = image.resize((image.width*scale, image.height*scale), Image.Resampling.LANCZOS) if scale > 1 else image
    data = pytesseract.image_to_data(ocr_image, config="--psm 6" if scale > 1 else "--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, raw in enumerate(data["text"]):
        word = raw.strip()
        if not word:
            continue
        x, y, w, h = (float(data[key][i])/scale for key in ("left", "top", "width", "height"))
        words.append({"text": word, "box": [round(x), round(y), round(x+w), round(y+h)], "confidence": float(data["conf"][i])})
    return words


def _tick_candidates(words: list[dict[str, Any]], width: int, height: int) -> dict[str, list[dict[str, Any]]]:
    # OCR often flattens 10^0, 10^1, ... into 10°, 10!, 102, 103.
    # Interpret this pattern only when several labels form a regularly spaced
    # row/column with consecutive exponents, so ordinary values 102 and 103
    # are not reinterpreted in isolation.
    powers = []
    for index, word in enumerate(words):
        match = re.fullmatch(r"10([0-9°!])", word["text"].strip())
        if match:
            digit = match.group(1)
            exponent = 0 if digit == "°" else 1 if digit == "!" else int(digit)
            box = word["box"]
            powers.append((index, exponent, (box[0]+box[2])/2, (box[1]+box[3])/2))
    overrides: dict[int, float] = {}
    for axis in (0, 1):
        cross = 1-axis
        for anchor in powers:
            group = [p for p in powers if abs(p[2+cross]-anchor[2+cross]) < (height*.04 if axis == 0 else width*.04)]
            if len(group) < 3:
                continue
            group.sort(key=lambda p: p[2+axis])
            exponents = [p[1] for p in group]
            positions = [p[2+axis] for p in group]
            steps = np.diff(positions)
            if min(steps) > 0 and (all(b-a == 1 for a, b in zip(exponents, exponents[1:])) or all(b-a == -1 for a, b in zip(exponents, exponents[1:]))) and max(steps)/min(steps) < 1.35:
                overrides.update({p[0]: float(10**p[1]) for p in group})
    numeric: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        cleaned = word["text"].replace("−", "-").replace(",", ".")
        if index not in overrides and not NUMBER.fullmatch(cleaned):
            continue
        try:
            val = overrides.get(index, float(cleaned) if NUMBER.fullmatch(cleaned) else math.nan)
        except ValueError:
            continue
        x0, y0, x1, y1 = word["box"]
        if y1-y0 < 5 or word["confidence"] < 25:
            continue
        center = [(x0 + x1) / 2, (y0 + y1) / 2]
        numeric.append({"center":center,"value":val,"box":word["box"],"text":word["text"]})
    # Tick labels form a horizontal row and a vertical column. This avoids
    # mistaking y labels near the bottom for x labels, or text in the plot.
    lower=[w for w in numeric if w["center"][1] > height*.72]
    rows:dict[int,list[dict[str,Any]]]={}
    for w in lower:
        key=round(w["center"][1]/max(10,height*.025))
        rows.setdefault(key,[]).append(w)
    xwords=max(rows.values(),key=lambda row:len({round(w["center"][0]/10) for w in row}),default=[])
    if len({round(w["center"][0]/10) for w in xwords})<2:
        xwords=[]
    xrow_y=float(np.median([w["center"][1] for w in xwords])) if xwords else height
    left=[w for w in numeric if w["center"][0]<width*.22 and w["center"][1]<xrow_y-5 and w["center"][1]>height*.05]
    cols:dict[int,list[dict[str,Any]]]={}
    for w in left:
        key=round(w["center"][0]/max(10,width*.02))
        cols.setdefault(key,[]).append(w)
    ywords=max(cols.values(),key=lambda col:len({round(w["center"][1]/10) for w in col}),default=[])
    ticks:dict[str,list[dict[str,Any]]]={"x":[],"y":[]}
    for axis,items,index in (("x",xwords,0),("y",ywords,1)):
        for w in items:
            ticks[axis].append({"id":f"{axis.upper()}{len(ticks[axis])+1}","pixel":w["center"][index],"value":w["value"],"box":w["box"],"text":w["text"]})
    ticks["x"].sort(key=lambda t: t["pixel"])
    ticks["y"].sort(key=lambda t: t["pixel"])
    return ticks


def _plot_bounds(ticks: dict[str, list[dict[str, Any]]], width: int, height: int) -> list[int]:
    xs = [t["pixel"] for t in ticks["x"]]
    ys = [t["pixel"] for t in ticks["y"]]
    if len(xs) >= 2:
        left, right = max(0, int(min(xs) - 8)), min(width, int(max(xs) + 8))
        # A cropped or low-contrast outer tick may be absent. Leave room for
        # data beyond the last recognized label without extrapolating values.
        if right < width * 0.9:
            right = int(width * 0.97)
        if left > width * 0.1:
            left = int(width * 0.03)
    else:
        left, right = int(width * 0.04), int(width * 0.96)
    if len(ys) >= 2:
        top, bottom = max(0, int(min(ys) - 8)), min(height, int(max(ys) + 8))
    elif xs:
        top, bottom = int(height * 0.12), min(height, int(max(t["box"][1] for t in ticks["x"]) - 6))
    else:
        top, bottom = int(height * 0.13), int(height * 0.9)
    if right - left < width * 0.3 or bottom - top < height * 0.3:
        return [int(width * 0.04), int(height * 0.12), int(width * 0.96), int(height * 0.9)]
    return [left, top, right, bottom]


def _snap_ticks_to_marks(image: Image.Image, ticks: dict[str, list[dict[str, Any]]]) -> None:
    """Move OCR text centers to nearby drawn tick marks on light plots."""
    array = np.asarray(image)
    if np.median(array) < 180:
        return
    dark = np.max(array, axis=2) < 100
    height, width = dark.shape
    lower = dark[int(height*.6):int(height*.95)]
    if lower.size:
        row_counts = lower.sum(axis=1)
        axis_y = int(height*.6) + int(np.argmax(row_counts))
        if row_counts.max() > width*.3 and axis_y+10 < height:
            for tick in ticks["x"]:
                center = float(tick["pixel"])
                left, right = max(0, round(center)-12), min(width, round(center)+13)
                scores = dark[axis_y+1:axis_y+10, left:right].sum(axis=0)
                if len(scores) and scores.max() >= 4:
                    positions = np.flatnonzero(scores == scores.max()) + left
                    nearest = min(positions, key=lambda x: abs(x-center))
                    if abs(nearest-center) <= 10:
                        tick["text_pixel"] = center
                        tick["pixel"] = float(nearest)
                        tick["pixel_source"] = "tick_mark"
    left_band = dark[:, int(width*.04):int(width*.3)]
    if left_band.size:
        col_counts = left_band.sum(axis=0)
        axis_x = int(width*.04) + int(np.argmax(col_counts))
        if col_counts.max() > height*.3 and axis_x >= 10:
            for tick in ticks["y"]:
                center = float(tick["pixel"])
                top, bottom = max(0, round(center)-10), min(height, round(center)+11)
                scores = dark[top:bottom, axis_x-10:axis_x].sum(axis=1)
                if len(scores) and scores.max() >= 4:
                    positions = np.flatnonzero(scores == scores.max()) + top
                    nearest = min(positions, key=lambda y: abs(y-center))
                    if abs(nearest-center) <= 8:
                        tick["text_pixel"] = center
                        tick["pixel"] = float(nearest)
                        tick["pixel_source"] = "tick_mark"


def _ink_mask(image: Image.Image, bounds: list[int], words: list[dict[str, Any]], remove_lines: bool = True) -> np.ndarray:
    array = np.asarray(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # Otsu handles grayscale scans; color traces still produce a dark foreground.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for word in words:
        if word["confidence"] < 60:
            continue
        if not any(char.isalnum() for char in word["text"]):
            continue
        x0, y0, x1, y1 = word["box"]
        if x1-x0 > image.width*.3 or y1-y0 > image.height*.2:
            continue
        cv2.rectangle(mask, (max(0, x0 - 3), max(0, y0 - 3)), (min(mask.shape[1] - 1, x1 + 3), min(mask.shape[0] - 1, y1 + 3)), 0, -1)
    x0, y0, x1, y1 = bounds
    outside = np.zeros_like(mask)
    outside[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    if not remove_lines:
        return outside
    # A steep signal can contain a long vertical run. Only remove strokes that
    # span nearly the full plot, rather than classifying a third-height flank
    # as a grid line. Gaps left at true grid crossings stay explicit downstream.
    horizontal = cv2.morphologyEx(outside, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int((x1-x0)*.85)), 1)))
    vertical = cv2.morphologyEx(outside, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, int((y1-y0)*.90)))))
    return cv2.subtract(outside, cv2.bitwise_or(horizontal, vertical))


def _trace_mask(mask: np.ndarray, bounds: list[int]) -> np.ndarray:
    """Retain connected, horizontally extended traces; discard label glyphs."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    span = bounds[2]-bounds[0]
    selected = [i for i in range(1, count) if stats[i, cv2.CC_STAT_WIDTH] >= max(25, span*.08) and stats[i, cv2.CC_STAT_AREA] >= max(100, span*.12)]
    if not selected:
        return mask
    return (np.isin(labels,selected).astype(np.uint8)*255)


def _profile(mask: np.ndarray, bounds: list[int]) -> np.ndarray:
    x0,y0,x1,y1 = bounds
    trace = mask[y0:y1,x0:x1]
    valid = trace.any(axis=0)
    result = np.full(x1-x0, np.nan)
    result[valid] = y0 + np.argmax(trace[:,valid] > 0, axis=0)
    return result


def _supported_apex(profile: np.ndarray, at: int, radius: int = 8) -> bool:
    """Require visible local descent on BOTH sides, without crossing gaps."""
    if not 1 <= at < len(profile)-1 or not np.isfinite(profile[at]):
        return False
    left = profile[max(0,at-radius):at]
    right = profile[at+1:min(len(profile),at+radius+1)]
    if len(left)<2 or len(right)<2 or not np.isfinite(profile[at-1:at+2]).all():
        return False
    # Only the connected segment nearest the proposed top is evidence.
    if np.isnan(left).any(): left = left[np.flatnonzero(np.isnan(left))[-1]+1:]
    if np.isnan(right).any(): right = right[:np.flatnonzero(np.isnan(right))[0]]
    if len(left)<2 or len(right)<2:
        return False
    y = profile[at]
    return bool(min(left.min(),right.min()) >= y-1.5 and left.max()>=y+2 and right.max()>=y+2)


def _peak_locations(profile: np.ndarray, prominence: float = 3, limit: int = 40) -> list[tuple[int,float]]:
    """Smooth contiguous observed segments only; absent ink is never a valley."""
    valid = np.isfinite(profile)
    edges = np.diff(np.r_[False,valid,False].astype(int))
    found: dict[int,float] = {}
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        segment = profile[start:end]
        if len(segment)<7: continue
        for sigma in (1.,2.):
            peaks,props = find_peaks(-gaussian_filter1d(segment,sigma),prominence=prominence,distance=3)
            for at,score in zip(peaks,props["prominences"]):
                at += start
                lo,hi = max(start,at-2),min(end,at+3)
                lowest = np.nanmin(profile[lo:hi])
                tops = np.flatnonzero(profile[lo:hi]<=lowest+.5)+lo
                apex = int(tops[np.argmin(abs(tops-at))])
                if _supported_apex(profile,apex):
                    nearby = next((old for old in found if abs(old-apex)<=4),None)
                    if nearby is None: found[apex]=float(score)
                    elif score>found[nearby]:
                        del found[nearby];found[apex]=float(score)
    return sorted(found.items(),key=lambda pair:pair[1],reverse=True)[:limit]


def _candidates(image: Image.Image, bounds: list[int], words: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    mask = _ink_mask(image, bounds, words)
    x0, y0, x1, y1 = bounds
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    result: list[dict[str, Any]] = []
    # Compact components are plausible scatter points. Keep their centers.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(crop, 8)
    for i in range(1, count):
        cx, cy = centroids[i]
        bx, by, w, h, area = stats[i]
        if 8 <= area <= 2000 and 3 <= w <= 45 and 3 <= h <= 45 and 0.35 <= w / h <= 2.8:
            result.append({"kind": "marker", "pixel": [round(float(x0+cx), 1), round(float(y0+cy), 1)], "score": round(float(area / (w*h)), 3)})
    # Local column-height maxima are candidate apexes. Restrict to connected ink
    # that extends close enough to the plot baseline to avoid title/label text.
    trace = _trace_mask(mask,bounds)[y0:y1,x0:x1]
    column_y = _profile(_trace_mask(mask,bounds),bounds)
    for col,prominence in _peak_locations(column_y,limit=limit):
        result.append({"kind":"peak", "pixel":[float(x0+col),float(column_y[col])],
                       "score":round(prominence,2), "prominence_pixels":round(prominence,2),
                       "detection_status":"validated_apex"})
    # No image-specific ppm or label rules are used here.
    result.sort(key=lambda c: c["score"], reverse=True)
    result = result[:limit]
    for candidate in result:
        if candidate["kind"] not in {"peak", "marker"}:
            continue
        px, py = candidate["pixel"]
        hints = []
        for word in words:
            label = word["text"].strip()
            if len(label) != 1 or not label.isalpha() or word["confidence"] < 70:
                continue
            bx0, by0, bx1, by1 = word["box"]
            dx = (bx0+bx1)/2-px
            dy = (by0+by1)/2-py
            if abs(dx) <= 35 and -45 <= dy <= 12:
                hints.append((abs(dx)+abs(dy), label.upper(), word["confidence"]))
        if hints:
            _, label, confidence = min(hints)
            candidate["label_hint"] = label
            candidate["label_hint_confidence"] = confidence
    for n, candidate in enumerate(result, 1):
        candidate["id"] = f"P{n}"
    return result


def _draw_overview(image: Image.Image, bounds: list[int], ticks: dict[str, list[dict[str, Any]]], candidates: list[dict[str, Any]], path: Path) -> None:
    display = image.copy()
    draw = ImageDraw.Draw(display)
    draw.rectangle(bounds, outline="#2d70ff", width=3)
    for axis in ("x", "y"):
        for tick in ticks[axis]:
            x0, y0, x1, y1 = tick["box"]
            draw.rectangle((x0-2, y0-2, x1+2, y1+2), outline="#078640", width=2)
    for candidate in candidates:
        x, y = candidate["pixel"]
        color = "#d62627" if candidate["kind"] == "peak" else "#8b38c5"
        draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=3)
        draw.text((x+6, y-11), candidate["id"], fill=color, stroke_fill="white", stroke_width=2)
    display.thumbnail((1600, 1200))
    display.save(path)


def inspect_plot(image_path: str, measurement: str = "requested points") -> dict[str, Any]:
    source = Path(image_path).expanduser().resolve(strict=True)
    image = _load_image(source)
    width, height = image.size
    words = _ocr(image)
    ticks = _tick_candidates(words, width, height)
    _snap_ticks_to_marks(image, ticks)
    bounds = _plot_bounds(ticks, width, height)
    candidates = _candidates(image, bounds, words)
    session_id = uuid.uuid4().hex
    folder = SESSION_ROOT / session_id
    folder.mkdir(parents=True, exist_ok=False)
    session = {"implementation_id": implementation_id(), "session_id": session_id, "image_path": str(source), "image_size": [width, height], "measurement": measurement, "bounds": bounds, "ticks": ticks, "words": words, "candidates": candidates}
    (folder / "session.json").write_text(json.dumps(session, indent=2))
    overview = folder / "overview.png"
    _draw_overview(image, bounds, ticks, candidates, overview)
    return {"implementation_id": implementation_id(), "session_id": session_id, "image_size": [width, height], "plot_bounds": bounds, "ticks": ticks, "candidates": candidates, "ocr_text": [w for w in words if w["confidence"] >= 35][:80], "overview_path": str(overview), "warning": "OCR and candidates are proposals. Verify visible axis ticks and requested features before measuring."}


def _read_session(session_id: str) -> tuple[Path, dict[str, Any], Image.Image]:
    folder = _path(session_id)
    session = json.loads((folder / "session.json").read_text())
    if session.get("implementation_id") != implementation_id():
        raise ValueError("session implementation differs from this tool; inspect the image again")
    return folder, session, _load_image(session["image_path"])


def refine_region(session_id: str, bounds: list[int]) -> dict[str, Any]:
    folder, session, image = _read_session(session_id)
    if len(bounds) != 4:
        raise ValueError("bounds must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = [int(v) for v in bounds]
    width, height = image.size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("region is outside original image")
    # Find smaller local features using the original full-trace connectivity,
    # then rank them only within the requested region.
    trace=_trace_mask(_ink_mask(image,session["bounds"],session["words"]),session["bounds"])
    top_by_x = _profile(trace,[x0,y0,x1,y1])
    local=[]
    next_id=1+sum(c["id"].startswith("R") for c in session["candidates"])
    for at,prominence in _peak_locations(top_by_x,prominence=2,limit=12):
        candidate={"id":f"R{next_id}","kind":"local_peak","pixel":[float(x0+at),float(top_by_x[at])],
                   "score":round(prominence,2),"detection_status":"validated_apex"}
        next_id+=1
        local.append(candidate)
    session["candidates"].extend(local)
    (folder/"session.json").write_text(json.dumps(session,indent=2))
    crop = image.crop((x0, y0, x1, y1))
    factor = min(4, max(1, int(900 / max(x1-x0, y1-y0))))
    if factor > 1:
        crop = crop.resize(((x1-x0)*factor, (y1-y0)*factor), Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(crop)
    for candidate in local:
        px,py=candidate["pixel"]
        u,v=(px-x0)*factor,(py-y0)*factor
        draw.ellipse((u-5,v-5,u+5,v+5),outline="#d62627",width=2)
        draw.text((u+6,v-10),candidate["id"],fill="#d62627",stroke_fill="white",stroke_width=2)
    index = len(list(folder.glob("region-*.png"))) + 1
    path = folder / f"region-{index}.png"
    crop.save(path)
    candidates = [c for c in session["candidates"] if x0 <= c["pixel"][0] < x1 and y0 <= c["pixel"][1] < y1]
    return {"session_id": session_id, "bounds_original": bounds, "display_scale": factor, "crop_path": str(path), "candidates": candidates}


def _resolve_point(selection: dict[str, Any], candidates: list[dict[str, Any]], image: Image.Image, bounds: list[int], words: list[dict[str, Any]]) -> tuple[float, float, str, float]:
    if selection.get("candidate_id"):
        match = next((c for c in candidates if c["id"] == selection["candidate_id"]), None)
        if match is None:
            raise ValueError(f"candidate {selection['candidate_id']} not found")
        x, y = map(float, match["pixel"])
        method = f"candidate:{match['kind']}"
    elif "pixel" in selection:
        x, y = map(float, selection["pixel"])
        method = "selected_pixel"
    else:
        raise ValueError("selection needs candidate_id or pixel")
    width, height = image.size
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("selected point is outside image")
    mode = selection.get("mode", "exact")
    uncertainty = 1.5
    is_peak = method in {"candidate:peak","candidate:local_peak"}
    if mode == "peak" or is_peak:
        radius = int(selection.get("radius",8))
        if not 1<=radius<=30:
            raise ValueError("peak radius must be 1..30 pixels")
        mask=_trace_mask(_ink_mask(image,bounds,words),bounds)
        profile=_profile(mask,bounds)
        at=round(x)-bounds[0]
        if mode=="peak":
            local=[(px,score) for px,score in _peak_locations(profile,prominence=2)
                   if abs(px-at)<radius and abs(profile[px]-y)<=radius*3]
            if not local:
                raise UnresolvedFeature("no supported interior apex in selected neighborhood; refine the region")
            local.sort(key=lambda p:abs(p[0]-at))
            if len(local)>1 and abs(local[0][0]-at)==abs(local[1][0]-at):
                raise UnresolvedFeature("multiple equally close apices; choose a candidate from a local review")
            at=local[0][0]
            x,y=float(bounds[0]+at),float(profile[at])
            method="local_peak"
        if not _supported_apex(profile,at):
            raise UnresolvedFeature("selected peak has no supported descent on both sides")
    elif mode == "scatter":
        method = "scatter_center" if method.startswith("candidate") else method
        uncertainty = 2.0
    elif mode == "curve":
        method = "curve_position"
        uncertainty = 2.0
    elif mode != "exact":
        raise ValueError("mode must be exact, peak, scatter, or curve")
    return float(x), float(y), method, float(uncertainty)


def _curve_at_x(image: Image.Image, bounds: list[int], words: list[dict[str, Any]], x: float, hint_y: float | None) -> tuple[float, float]:
    x0,y0,x1,y1=bounds
    if not x0 <= x < x1:
        raise ValueError("curve query is outside the visible plot")
    mask=_ink_mask(image,bounds,words,remove_lines=False)
    ys=[]
    for px in range(max(x0,round(x)-2),min(x1,round(x)+3)):
        ys.extend(np.flatnonzero(mask[y0:y1,px])+y0)
    if not ys:
        raise ValueError("no visible curve at requested x position")
    values=np.asarray(sorted(ys))
    gaps=np.flatnonzero(np.diff(values)>5)+1
    clusters=[c for c in np.split(values,gaps) if len(c)>=2]
    if not clusters:
        raise ValueError("no continuous curve segment at requested x position")
    centers=[float(np.median(c)) for c in clusters]
    if hint_y is None and len(centers)>1:
        raise ValueError("multiple curves at requested x; provide pixel:[x,approx_y] to identify the series")
    index=0 if hint_y is None else int(np.argmin([abs(c-hint_y) for c in centers]))
    if hint_y is not None and abs(centers[index]-hint_y)>25:
        raise ValueError("selected curve is too far from supplied series hint")
    thickness=max(1.0,(float(clusters[index].max())-float(clusters[index].min()))/2)
    return centers[index],thickness


def _axis_from_selection(spec: dict[str, Any], tick_list: list[dict[str, Any]]) -> Axis:
    """Resolve concise OCR tick IDs, or accept explicit image-pixel anchors."""
    if "anchor_ids" not in spec:
        if "check_ids" in spec:
            raise ValueError("check_ids requires anchor_ids")
        return Axis(**spec)
    if "anchors" in spec or "checks" in spec:
        raise ValueError("use either tick IDs or explicit pixel anchors")
    ids = list(spec["anchor_ids"])
    if len(ids) != 2 or ids[0] == ids[1]:
        raise ValueError("exactly two distinct anchor_ids are required")
    by_id = {tick["id"]: tick for tick in tick_list}
    def point(tick_id: str) -> dict[str, float]:
        if tick_id not in by_id:
            raise ValueError(f"tick {tick_id} was not found in this plot session")
        tick = by_id[tick_id]
        return {"pixel": float(tick["pixel"]), "value": float(tick["value"])}
    check_ids = list(spec.get("check_ids", []))
    if any(tick_id in ids for tick_id in check_ids):
        raise ValueError("check tick must be independent of anchor ticks")
    return Axis([point(tick_id) for tick_id in ids], spec.get("scale", "linear"), [point(tick_id) for tick_id in check_ids])


def measure_plot(session_id: str, x_axis: dict[str, Any], selections: list[dict[str, Any]], y_axis: dict[str, Any] | None = None, reference_label: str | None = None, baseline_pixel: float | None = None) -> dict[str, Any]:
    folder, session, image = _read_session(session_id)
    if not selections:
        raise ValueError("at least one point is required")
    if len(selections) > 100:
        raise ValueError("at most 100 requested points")
    xmap = _axis_from_selection(x_axis, session["ticks"]["x"])
    ymap = _axis_from_selection(y_axis, session["ticks"]["y"]) if y_axis else None
    xres = xmap.residual()
    yres = ymap.residual() if ymap else None
    if xres is not None and xres > xmap.transformed_error(4):
        raise ValueError("x-axis check tick disagrees by more than 4 pixels")
    if yres is not None and yres > ymap.transformed_error(4):
        raise ValueError("y-axis check tick disagrees by more than 4 pixels")
    rows = []
    seen = set()
    for selection in selections:
        label = str(selection["label"])
        if label in seen:
            raise ValueError(f"duplicate label {label}")
        seen.add(label)
        if selection.get("candidate_id") and len(label) == 1 and label.isalpha():
            selected_candidate = next((c for c in session["candidates"] if c["id"] == selection["candidate_id"]), None)
            if selected_candidate and selected_candidate.get("label_hint") and selected_candidate["label_hint"] != label.upper() and selected_candidate.get("label_hint_confidence", 0) >= 75:
                raise ValueError(f"selected {selection['candidate_id']} is near printed label {selected_candidate['label_hint']}, not {label}; verify image label assignment")
        if selection.get("mode") == "curve" and "x_value" in selection:
            x=xmap.pixel_for(float(selection["x_value"]))
            hint_y=float(selection["pixel"][1]) if "pixel" in selection else None
            y,pixel_uncertainty=_curve_at_x(image,session["bounds"],session["words"],x,hint_y)
            method="curve_trace_at_x"
        else:
            try:
                x, y, method, pixel_uncertainty = _resolve_point(selection, session["candidates"], image, session["bounds"], session["words"])
            except UnresolvedFeature as error:
                rows.append({"label":label,"status":"unresolved","detection_status":"unsupported",
                             "assignment_status":"unresolved","reason":str(error),"alternatives":[],
                             "x":None,"y":None,"pixel":None,"pixel_uncertainty":None,
                             "x_uncertainty_transformed":None,"y_uncertainty_transformed":None,"method":"unresolved"})
                continue
        xp = xmap.convert(x)
        yp = ymap.convert(y) if ymap else None
        xe = xmap.transformed_error(pixel_uncertainty + 1)
        ye = ymap.transformed_error(pixel_uncertainty + 1) if ymap else None
        if xres is not None:
            xe = math.hypot(xe, xres)
        if yres is not None and ye is not None:
            ye = math.hypot(ye, yres)
        rows.append({"label": label, "status":"resolved", "detection_status":"validated_apex" if "peak" in method else "selected_feature", "assignment_status":"model_selected", "coordinate_uncertainty_note":"conditional on correct feature assignment", "x": xp, "y": yp, "pixel": [x, y], "x_uncertainty_transformed": xe, "y_uncertainty_transformed": ye, "pixel_uncertainty": pixel_uncertainty, "method": method})
    if reference_label:
        ref = next((r for r in rows if r["label"] == reference_label), None)
        if ref is None:
            raise ValueError("reference label must be among selections")
        if ref.get("status") == "unresolved":
            raise ValueError("normalization reference is unresolved; resolve it or measure without normalization")
        if baseline_pixel is None:
            raise ValueError("normalized heights require a baseline pixel")
        baseline_pixel = float(baseline_pixel)
        if not math.isfinite(baseline_pixel) or not 0 <= baseline_pixel < image.height:
            raise ValueError("baseline pixel is outside image")
        if ymap is not None and ymap.scale == "log10":
            raise ValueError("baseline-relative height is unsupported on a log y-axis")
        ref_height = baseline_pixel - ref["pixel"][1]
        if abs(ref_height) < 2:
            raise ValueError("reference point is too close to baseline")
        for row in rows:
            if row.get("status") == "unresolved":
                continue
            height = baseline_pixel - row["pixel"][1]
            row["height_pixels"] = height
            row["normalized_height"] = height / ref_height
            # First-order propagation in source-image pixels.  The baseline is
            # independently located, while the reference is exact by definition.
            if row is ref:
                row["normalized_height_uncertainty"] = 0.0
            else:
                sigma_y = row["pixel_uncertainty"]
                sigma_ref = ref["pixel_uncertainty"]
                sigma_baseline = 1.5
                row["normalized_height_uncertainty"] = math.sqrt((sigma_y/ref_height)**2 + (height*sigma_ref/ref_height**2)**2 + ((row["pixel"][1]-ref["pixel"][1])*sigma_baseline/ref_height**2)**2)
        ref["normalized_height"] = 1.0
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for row in rows:
        if row.get("status") == "unresolved":
            continue
        x, y = row["pixel"]
        draw.ellipse((x-7, y-7, x+7, y+7), outline="#e31c23", width=3)
        draw.text((x+9, y-12), f"{row['label']} {row['x']:.4g}", fill="#a00000", stroke_fill="white", stroke_width=2)
    overlay_path = folder / "measured.png"
    overlay.save(overlay_path)
    json_path = folder / "measurements.json"
    result = {"implementation_id": implementation_id(), "session_id": session_id, "source_image": session["image_path"], "image_size": session["image_size"], "x_axis": asdict(xmap), "y_axis": asdict(ymap) if ymap else None, "baseline_pixel": baseline_pixel, "reference_label": reference_label, "calibration_residual_transformed": {"x": xres, "y": yres}, "points": rows, "overlay_path": str(overlay_path), "json_path": str(json_path), "csv_path": str(folder / "measurements.csv")}
    json_path.write_text(json.dumps(result, indent=2))
    with (folder / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "status", "reason", "assignment_status", "x", "y", "height_pixels", "normalized_height", "normalized_height_uncertainty", "pixel_x", "pixel_y", "x_uncertainty_transformed", "y_uncertainty_transformed", "method"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"label": row["label"], "status":row.get("status"), "reason":row.get("reason"), "assignment_status":row.get("assignment_status"), "x": row["x"], "y": row["y"], "height_pixels": row.get("height_pixels"), "normalized_height": row.get("normalized_height"), "normalized_height_uncertainty": row.get("normalized_height_uncertainty"), "pixel_x": row["pixel"][0] if row["pixel"] else None, "pixel_y": row["pixel"][1] if row["pixel"] else None, "x_uncertainty_transformed": row["x_uncertainty_transformed"], "y_uncertainty_transformed": row["y_uncertainty_transformed"], "method": row["method"]})
    return result
