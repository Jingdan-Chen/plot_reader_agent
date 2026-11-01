"""Experimental source-template evidence for repeated dark scatter markers.

This module proposes image-grounded candidates, never measured data or model
confidence. It does not change the browser's default extraction workflow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter


def legend_components(source: Path, regions: list[list[float]]) -> list[dict]:
    """Give isolated legend glyphs stable IDs; LUNA chooses, not draws, kernels."""
    gray = np.asarray(Image.open(source).convert("L"))
    height, width = gray.shape
    found = set()
    for box in regions:
        left, top, right, bottom = [int(round(v)) for v in box]
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Invalid legend region")
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            (gray[top:bottom, left:right] < 180).astype(np.uint8), 8)
        for x, y, w, h, area in stats[1:]:
            # Includes text: the semantic stage must explicitly identify glyphs.
            if min(w, h) >= 3 and max(w, h) <= max(12, .06 * max(width, height)) and area >= 8:
                found.add((int(left+x), int(top+y), int(left+x+w), int(top+y+h)))
    return [{"id": f"G{i}", "bounds": list(b)}
            for i, b in enumerate(sorted(found, key=lambda b: (b[1], b[0])), 1)]


def component_atlas(source: Path, components: list[dict], output: Path) -> None:
    im = Image.open(source).convert("RGB")
    cell, cols = 100, 8
    atlas = Image.new("RGB", (cols*cell, max(1, math.ceil(len(components)/cols))*120), "#edf1f5")
    d = ImageDraw.Draw(atlas)
    for i, item in enumerate(components):
        x, y = (i % cols)*cell, (i // cols)*120
        a, b, c, e = item["bounds"]
        patch = im.crop((a-4, b-4, c+4, e+4))
        patch.thumbnail((90, 90))
        factor = min(90/patch.width, 90/patch.height)
        patch = patch.resize((round(patch.width*factor), round(patch.height*factor)), Image.Resampling.NEAREST)
        atlas.paste(patch, (x+(cell-patch.width)//2, y+24))
        d.text((x+5, y+4), item["id"], fill="black")
    atlas.save(output)


def validate_profile(profile: dict, components: list[dict], series: list[str]) -> None:
    if profile.get("applicable") is not True:
        raise ValueError("Repeated-marker method is not applicable: " + profile.get("reason", "unspecified"))
    items = profile.get("templates", [])
    ids = {p["id"] for p in components}
    if sorted(p["series"] for p in items) != sorted(series):
        raise ValueError("Exactly one template is required for each series")
    if any(p["component_id"] not in ids for p in items):
        raise ValueError("Unknown source glyph ID")
    if len({p["component_id"] for p in items}) != len(items):
        raise ValueError("A glyph cannot silently define multiple series")


def _kernel(gray: np.ndarray, bounds: list[int], scale: float) -> tuple[np.ndarray, tuple[float, float]]:
    a, b, c, d = bounds
    glyph = (1-gray[b:d, a:c].astype(np.float32)/255)
    if glyph.size == 0 or min(glyph.shape) < 2 or np.ptp(glyph) < .15:
        raise ValueError("Template must contain a visible symbol and background")
    w, h = max(3, round(glyph.shape[1]*scale)), max(3, round(glyph.shape[0]*scale))
    glyph = cv2.resize(glyph, (w, h), interpolation=cv2.INTER_LINEAR)
    pad = max(3, round(.3*max(w, h)))
    kernel = np.pad(glyph, pad)
    # Bounding-box center, not ink centroid (which biases hollow triangles).
    return kernel, (pad+(w-1)/2, pad+(h-1)/2)


def _line_response(ink: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Competing line-only explanation; do not erase lines from the source."""
    h, w = size
    response = np.full((ink.shape[0]-h+1, ink.shape[1]-w+1), -1., np.float32)
    c = ((w-1)/2, (h-1)/2)
    for angle in range(0, 180, 15):
        radians = math.radians(angle)
        dx, dy = math.cos(radians)*max(w, h), math.sin(radians)*max(w, h)
        for thickness in sorted({1, max(2, round(min(w, h)/12))}):
            line = np.zeros((h, w), np.float32)
            cv2.line(line, (round(c[0]-dx), round(c[1]-dy)),
                     (round(c[0]+dx), round(c[1]+dy)), 1., thickness)
            response = np.maximum(response, cv2.matchTemplate(ink, line, cv2.TM_CCOEFF_NORMED))
    return response


def _subpixel(response: np.ndarray, x: int, y: int) -> tuple[list[float], bool]:
    """Bounded 2-D quadratic maximum; saddle/flat/edge peaks stay integral."""
    if x < 1 or y < 1 or x >= response.shape[1]-1 or y >= response.shape[0]-1:
        return [0., 0.], False
    z = response[y-1:y+2, x-1:x+2].astype(float)
    gradient = np.array([(z[1, 2]-z[1, 0])/2, (z[2, 1]-z[0, 1])/2])
    cross = (z[2, 2]-z[2, 0]-z[0, 2]+z[0, 0])/4
    hessian = np.array([[z[1, 2]-2*z[1, 1]+z[1, 0], cross],
                        [cross, z[2, 1]-2*z[1, 1]+z[0, 1]]])
    if np.max(np.linalg.eigvalsh(hessian)) >= -1e-4:
        return [0., 0.], False
    delta = -np.linalg.solve(hessian, gradient)
    if not np.all(np.isfinite(delta)) or np.max(np.abs(delta)) > .75:
        return [0., 0.], False
    return delta.tolist(), True


def scan_templates(source: Path, components: list[dict], profile: dict, layout: dict,
                   *, scales=(.85, 1., 1.15), threshold=.57, line_margin=.06) -> tuple[list[dict], dict[str, np.ndarray]]:
    """NCC maxima, source coverage, line competition, and per-series NMS.

    Cross-series conflicts remain alternatives, never silently merged. Native
    response maps have centers at their actual image coordinates, not the
    top-left convention of cv2.matchTemplate.
    """
    gray = np.asarray(Image.open(source).convert("L"))
    ink = 1-gray.astype(np.float32)/255
    height, width = gray.shape
    if not scales or any(not math.isfinite(s) or not .5 <= s <= 2 for s in scales):
        raise ValueError("Template scales must be finite and between .5 and 2")
    if not math.isfinite(threshold) or not 0 < threshold <= 1 or not math.isfinite(line_margin) or not 0 <= line_margin <= 1:
        raise ValueError("Invalid response thresholds")
    validate_profile(profile, components, [p["series"] for p in profile["templates"]])
    by_id = {p["id"]: p for p in components}
    left, top, right, bottom = layout["plot_bounds"]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Invalid plot bounds")
    exclusions = layout.get("exclude_regions", [])
    yy, xx = np.mgrid[:height, :width]
    valid = (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
    for a, b, c, d in exclusions:
        valid &= ~((xx >= a) & (xx <= c) & (yy >= b) & (yy <= d))
    line_cache, maps, candidates = {}, {}, []
    for item in profile["templates"]:
        raw = []
        native = np.zeros((height, width), np.float32)
        for scale in scales:
            kernel, center = _kernel(gray, by_id[item["component_id"]]["bounds"], scale)
            kh, kw = kernel.shape
            if kh >= height or kw >= width:
                continue
            response = cv2.matchTemplate(ink, kernel, cv2.TM_CCOEFF_NORMED)
            support = (kernel > .4).astype(np.float32)
            coverage = cv2.matchTemplate(ink, support, cv2.TM_CCORR) / max(1, support.sum())
            if kernel.shape not in line_cache:
                line_cache[kernel.shape] = _line_response(ink, kernel.shape)
            contrast = response-line_cache[kernel.shape]
            accepted = (response >= threshold) & (coverage >= .5) & (contrast >= line_margin)
            # Preserve response values, with impossible line-only explanations gated.
            signal = np.where(accepted, response, 0).astype(np.float32)
            shifted = cv2.warpAffine(signal, np.float32([[1, 0, center[0]], [0, 1, center[1]]]),
                                     (width, height), flags=cv2.INTER_LINEAR)
            native = np.maximum(native, shifted)
            size = max(3, round(min(kh, kw)*.55))
            ys, xs = np.where(accepted & (response == maximum_filter(response, size=size)))
            for y, x in zip(ys, xs):
                offset, stable = _subpixel(response, int(x), int(y))
                pixel = [float(x+center[0]+offset[0]), float(y+center[1]+offset[1])]
                px, py = map(round, pixel)
                if not (0 <= px < width and 0 <= py < height and valid[py, px]):
                    continue
                raw.append({"series": item["series"], "pixel": [round(v, 3) for v in pixel],
                            "score": round(float(response[y, x]), 4), "coverage": round(float(coverage[y, x]), 4),
                            "line_margin": round(float(contrast[y, x]), 4), "scale": scale,
                            "template_id": item["component_id"], "center_stable": stable,
                            "diameter": round(min(kh, kw)/1.6, 2)})
        native[~valid] = 0
        maps[item["series"]] = native
        kept = []
        for p in sorted(raw, key=lambda p: p["score"], reverse=True):
            if all(math.dist(p["pixel"], q["pixel"]) > .5*min(p["diameter"], q["diameter"]) for q in kept):
                kept.append(p)
        candidates.extend(kept)
    candidates.sort(key=lambda p: (p["pixel"][0], p["pixel"][1], p["series"]))
    for i, p in enumerate(candidates, 1):
        p["id"] = f"T{i}"
    for p in candidates:
        alternatives = [q for q in candidates if q["series"] != p["series"] and
                        math.dist(p["pixel"], q["pixel"]) <= .5*min(p["diameter"], q["diameter"])]
        p["alternatives"] = [q["id"] for q in alternatives]
        p["status"] = "needs_review" if alternatives or not p["center_stable"] else "candidate"
        p["offset_limit"] = 0  # Model review cannot move a response maximum.
    return candidates, maps


def save_evidence(source: Path, candidates: list[dict], maps: dict[str, np.ndarray],
                  output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    source_im = Image.open(source).convert("RGB")
    w, h = source_im.size
    paths = []
    for i, (series, response) in enumerate(maps.items()):
        # Static diagnostic, paired with untouched source for visual review.
        heat = cv2.applyColorMap(np.uint8(np.clip(response, 0, 1)*255), cv2.COLORMAP_INFERNO)
        heat = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        panel = Image.new("RGB", (w*2, h+55), "white")
        panel.paste(source_im, (0, 55)); panel.paste(heat, (w, 55))
        d = ImageDraw.Draw(panel)
        font = ImageFont.load_default(size=20)
        title = series.translate(str.maketrans({"□": "square", "△": "up triangle", "▽": "down triangle", "◇": "diamond", "×": "cross"}))
        d.text((10, 8), f"{title} | source + candidate IDs", fill="black", font=font)
        d.text((w+10, 8), "Gated NCC (0-1); white rings highlight candidate peaks", fill="black", font=font)
        for p in candidates:
            if p["series"] != series:
                continue
            x, y = p["pixel"]; y += 55
            color = "#e07800" if p["status"] == "needs_review" else "#007c9d"
            d.ellipse((x-10, y-10, x+10, y+10), outline=color, width=2)
            d.text((x+12, y-12), p["id"], fill=color, font=font)
            d.ellipse((w+x-7, y-7, w+x+7, y+7), outline="white", width=1)
        path = output/f"response-{i}.png"
        panel.save(path); paths.append(str(path.resolve()))
    np.savez_compressed(output/"responses.npz", **{f"series_{i}": a for i, a in enumerate(maps.values())})
    (output/"candidates.json").write_text(json.dumps(candidates, indent=2, allow_nan=False))
    return paths
