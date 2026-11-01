"""Source-coordinate evidence and bounded selection review for browser jobs."""
from __future__ import annotations

import copy
from collections import Counter
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def unique_selection_labels(plan: dict[str, Any]) -> dict[str, Any]:
    """Namespace repeated feature labels across series for the frozen engine."""
    plan = copy.deepcopy(plan)
    counts = Counter(item["label"] for item in plan["selections"])
    used: set[str] = set()
    names: dict[str, list[str]] = {}
    for item in plan["selections"]:
        original = item["label"]
        base = f"{item.get('series', 'Series')} · {original}" if counts[original] > 1 else original
        label, suffix = base, 2
        while label in used:
            label = f"{base} ({suffix})"
            suffix += 1
        used.add(label)
        names.setdefault(original, []).append(label)
        item["label"] = label
    reference = plan.get("reference_label")
    if reference in names:
        if len(names[reference]) == 1:
            plan["reference_label"] = names[reference][0]
        else:
            plan["reference_label"] = None
            plan.setdefault("notes", []).append("Normalization reference label repeats across series; choose a series-specific reference or retain original values.")
    return plan


def coordinate_evidence(source: Path, folder: Path, *,
                        boxes: list[tuple[int, int, int, int]] | None = None,
                        spacing: int | None = None) -> list[dict[str, Any]]:
    """Label original pixels explicitly, including in overlapping local crops."""
    image = Image.open(source).convert("RGB")
    width, height = image.size
    folder.mkdir(parents=True, exist_ok=True)
    custom_boxes = boxes is not None
    boxes = list(boxes) if custom_boxes else [(0, 0, width, height)]
    if not custom_boxes and (width > 900 or height > 900):
        overlap = 70
        boxes.extend((max(0, x-overlap), max(0, y-overlap),
                      min(width, x+math.ceil(width/2)+overlap),
                      min(height, y+math.ceil(height/2)+overlap))
                     for y in (0, height//2) for x in (0, width//2))
    records = []
    font_size = 10 if spacing and spacing < 50 else 14
    font = ImageFont.load_default(size=font_size)
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        crop = image.crop(box).convert("RGBA")
        grid = Image.new("RGBA", crop.size)
        draw = ImageDraw.Draw(grid)
        step = spacing or (100 if index == 0 else 50)
        for x in range(math.ceil(x0/step)*step, x1, step):
            draw.line((x-x0, 0, x-x0, y1-y0), fill=(0, 110, 240, 75))
            label_width = draw.textbbox((0, 0), str(x), font=font)[2] + 3
            draw.rectangle((x-x0+1, 0, x-x0+label_width, font_size+3), fill=(240, 248, 255, 245))
            draw.text((x-x0+2, 1), str(x), fill=(0, 70, 170), font=font)
        for y in range(math.ceil(y0/step)*step, y1, step):
            draw.line((0, y-y0, x1-x0, y-y0), fill=(0, 110, 240, 75))
            label_width = draw.textbbox((0, 0), str(y), font=font)[2] + 3
            draw.rectangle((0, y-y0+1, label_width, y-y0+font_size+4), fill=(240, 248, 255, 245))
            draw.text((1, y-y0+2), str(y), fill=(0, 70, 170), font=font)
        path = folder / f"source-grid-{index}.png"
        Image.alpha_composite(crop, grid).convert("RGB").save(path)
        records.append({"path": str(path), "source_bounds": list(box), "display_scale": 1,
                        "coordinates": "Blue grid labels are ORIGINAL source pixels; x right, y down. Do not report crop-relative pixels."})
    return records


def unsupported_centers(plan: dict[str, Any], source: Path, tolerance: float = 12) -> list[dict[str, Any]]:
    """A hollow center can be white; flag only locations far from ANY source ink.

    This is a diagnostic, not a marker identity classifier or a reason to stop.
    """
    image = Image.open(source).convert("RGB")
    ink = np.min(np.asarray(image), axis=2) < 210
    distance = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    result = []
    for index, item in enumerate(plan["selections"]):
        pixel = item.get("pixel")
        if not pixel or item.get("status") == "unresolved":
            continue
        x, y = (int(round(v)) for v in pixel)
        if not (0 <= x < image.width and 0 <= y < image.height):
            continue
        gap = float(distance[y, x])
        if gap > tolerance:
            result.append({"selection_index": index, "label": item["label"], "pixel": pixel,
                           "ink_distance_pixels": round(gap, 1),
                           "bounds": [max(0, x-100), max(0, y-120),
                                      min(image.width, x+100), min(image.height, y+120)]})
    return result


def focused_marker_candidates(source: Path, evidence: list[dict[str, Any]], zoom: int = 3) -> list[dict[str, Any]]:
    """Expose bounded marker contours as IDs, so review need not guess pixels."""
    image = Image.open(source).convert("RGB")
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    candidates = []
    for record in evidence:
        x0, y0, x1, y1 = record["source_bounds"]
        contours, _ = cv2.findContours((gray[y0:y1, x0:x1] < 180).astype(np.uint8),
                                      cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if not (4 <= width <= 32 and 4 <= height <= 32 and .45 <= width/height <= 2.2 and 10 <= area <= 650):
                continue
            hull = cv2.contourArea(cv2.convexHull(contour))
            if not hull or area/hull < .65:
                continue
            moments = cv2.moments(contour)
            pixel = [round(x0 + moments["m10"]/moments["m00"], 2),
                     round(y0 + moments["m01"]/moments["m00"], 2)]
            if any(math.dist(pixel, other["pixel"]) < 4 for other in candidates):
                continue
            vertices = len(cv2.approxPolyDP(contour, .04*cv2.arcLength(contour, True), True))
            candidates.append({"id": f"C{len(candidates)+1}", "pixel": pixel,
                               "bounds": [x0+x, y0+y, x0+x+width, y0+y+height],
                               "shape_hint": {3: "triangle", 4: "quadrilateral"}.get(vertices, "compact contour")})
    for record in evidence:
        x0, y0, x1, y1 = record["source_bounds"]
        crop = Image.open(record["path"]).convert("RGB")
        draw = ImageDraw.Draw(crop)
        for candidate in candidates:
            x, y = candidate["pixel"]
            if x0 <= x < x1 and y0 <= y < y1:
                x, y = x-x0, y-y0
                draw.ellipse((x-6, y-6, x+6, y+6), outline="#bb00a0", width=1)
                draw.text((x+6, y-12), candidate["id"], fill="#bb00a0",
                          font=ImageFont.load_default(size=10), stroke_width=1, stroke_fill="white")
        crop.resize((crop.width*zoom, crop.height*zoom)).save(record["path"])
        record["display_scale"] = zoom
    return candidates


def bind_review_candidates(review: dict[str, Any], candidates: list[dict[str, Any]],
                           selections: list[dict[str, Any]] | None = None,
                           max_shift: float | None = None) -> dict[str, Any]:
    review = copy.deepcopy(review)
    by_id = {item["id"]: item for item in candidates}
    for item in review["items"]:
        if item.get("candidate_id"):
            if item["candidate_id"] not in by_id:
                raise ValueError("review selected an unknown source contour ID")
            chosen = by_id[item["candidate_id"]]
            index = int(item["selection_id"])-1
            seed = selections[index].get("pixel") if selections and 0 <= index < len(selections) else None
            if seed and max_shift is not None and math.dist(seed, chosen["pixel"]) > max_shift:
                nearby = sorted(((math.dist(seed, candidate["pixel"]), candidate) for candidate in candidates
                                 if math.dist(seed, candidate["pixel"]) <= 20), key=lambda pair: pair[0])
                if nearby and (len(nearby) == 1 or nearby[1][0]-nearby[0][0] > 3):
                    item["pixel"] = nearby[0][1]["pixel"]
                    item["reason"] = f"Distant contour {item['candidate_id']} rejected; nearby contour {nearby[0][1]['id']} retained as an approximate center. Series identity needs review."
                else:
                    item["pixel"] = seed
                    item["reason"] = f"Distant contour {item['candidate_id']} rejected; original approximate center retained pending local identity review."
                item["verdict"] = "uncertain"
                continue
            item["pixel"] = chosen["pixel"]
            item["reason"] = f"Source contour {item['candidate_id']}. {item['reason']}"
        elif selections and item["verdict"] != "uncertain":
            index = int(item["selection_id"])-1
            if 0 <= index < len(selections) and re.search(r"square|triangle|diamond|circle", selections[index].get("series", ""), re.I):
                item["verdict"] = "uncertain"
                item["reason"] += " No source-measured contour confirmed this marker center."
    return review


def selection_overlay(source: Path, selections: list[dict[str, Any]], output: Path) -> None:
    """Use short stable IDs so labels do not obscure the figure being reviewed."""
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)
    for index, selection in enumerate(selections, 1):
        pixel = selection.get("pixel")
        if not pixel:
            continue
        x, y = pixel
        color = "#aa6500" if selection.get("status") in {"unresolved", "needs_review"} else "#df3b2f"
        draw.ellipse((x-7, y-7, x+7, y+7), outline=color, width=2)
        draw.text((x+7, y-13), str(index), fill=color, font=font,
                  stroke_width=1, stroke_fill="white")
    image.save(output)


def apply_selection_review(plan: dict[str, Any], review: dict[str, Any],
                           image_size: tuple[int, int]) -> tuple[dict[str, Any], list[str]]:
    """Apply reviewed source pixels, retaining uncertain numbers with warnings."""
    updated = copy.deepcopy(plan)
    width, height = image_size
    items = review.get("items", [])
    expected = {str(index) for index in range(1, len(plan["selections"])+1)}
    ids = [item.get("selection_id") for item in items]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("coordinate review must cover every selection ID exactly once")
    warnings = []
    for item in items:
        selection = updated["selections"][int(item["selection_id"])-1]
        if selection.get("reason") == "user skipped clarification":
            selection["coordinate_review"] = "uncertain"
            selection["coordinate_review_note"] = "User skipped clarification; remains unresolved."
            warnings.append(f"{selection['label']}: User skipped clarification; remains unresolved.")
            continue
        pixel = item.get("pixel")
        if pixel is not None:
            if (len(pixel) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pixel)
                    or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)):
                raise ValueError("coordinate review returned a pixel outside the source image")
            selection.update(pixel=pixel, candidate_id=None,
                             mode="curve" if selection.get("mode") == "curve" else "exact",
                             status="selected", reason=None)
        selection["coordinate_review"] = item["verdict"]
        selection["coordinate_review_note"] = item["reason"]
        if item["verdict"] == "uncertain":
            warnings.append(f"{selection['label']}: {item['reason']}")
        elif not pixel:
            raise ValueError("confirmed or corrected selections need a source pixel")
    return updated, warnings
