from __future__ import annotations

import csv
import hashlib
import json
import math
import os
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

SESSION_ROOT = Path(os.environ.get("PLOT_READER_SESSION_ROOT", "/tmp/plot-reader-sessions" if os.name == "posix" else str(Path(tempfile.gettempdir()) / "plot-reader-sessions"))).resolve()
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


def _local_labels(image: Image.Image, bounds: list[int], candidates: list[dict[str, Any]], requested: list[str], existing_words: list[dict[str,Any]] | None = None) -> list[dict[str, Any]]:
    """Read isolated label glyphs at useful scale, without OCRing trace pixels."""
    wanted={label.casefold() for label in requested}
    mask=_ink_mask(image,bounds,[],remove_lines=False)
    _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    found=[]
    for x,y,w,h,area in stats[1:]:
        if not (5<=w<=95 and 8<=h<=100 and 20<=area<=4000):continue
        if any(len(word["text"])>1 and word["text"].casefold() not in wanted and word["confidence"]>=25 and word["box"][0]<=x+w/2<=word["box"][2] and word["box"][1]<=y+h/2<=word["box"][3] for word in existing_words or []):continue
        if not any(abs(c["pixel"][0]-(x+w/2))<100 and abs(c["pixel"][1]-(y+h/2))<120 for c in candidates):continue
        crop=image.crop((max(0,x-5),max(0,y-5),min(image.width,x+w+5),min(image.height,y+h+5)))
        crop=crop.resize((crop.width*3,crop.height*3),Image.Resampling.LANCZOS)
        data=pytesseract.image_to_data(crop,config="--psm 10",output_type=pytesseract.Output.DICT)
        for text,confidence in zip(data["text"],data["conf"]):
            text=text.strip()
            if text.casefold() in wanted and float(confidence)>=65:
                found.append({"text":text,"box":[int(x),int(y),int(x+w),int(y+h)],"confidence":float(confidence),"source":"local_glyph_ocr"})
    return found


def _associate_labels(candidates: list[dict[str, Any]], words: list[dict[str, Any]], requested: list[str]) -> dict[str, Any]:
    assignments={}
    for label in requested:
        boxes=[w for w in words if w["text"].casefold()==label.casefold() and w["confidence"]>=65]
        ranked=[]
        for c in candidates:
            px,py=c["pixel"]
            distances=[]
            for w in boxes:
                x0,y0,x1,y1=w["box"]
                if x0<=px<=x1 and y0<=py<=y1:
                    continue  # glyph interiors must never be label targets
                dx,dy=(x0+x1)/2-px,(y0+y1)/2-py
                if abs(dx)<=60 and -105<=dy<=60:
                    distances.append((abs(dx)+.25*abs(dy),w))
            if distances:
                distance,word=min(distances,key=lambda pair:pair[0])
                ranked.append((distance,c,word))
        ranked.sort(key=lambda item:item[0])
        unique=bool(ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>8) and ranked[0][2]["confidence"]>=75)
        assignments[label]={"status":"supported" if unique else "needs_review", "candidate_ids":[r[1]["id"] for r in ranked[:4]], "label_boxes":[w["box"] for w in boxes]}
        if unique:
            c=ranked[0][1]
            c["label_hint"]=label.upper()
            c["label_hint_confidence"]=ranked[0][2]["confidence"]
            assignments[label]["matched_label_box"]=ranked[0][2]["box"]
    return assignments


def _review_panels(image: Image.Image, bounds: list[int], candidates: list[dict[str, Any]], folder: Path) -> list[dict[str,Any]]:
    """Readable local evidence with original-coordinate regions and stable IDs."""
    x0,y0,x1,y1=bounds
    count=max(1,math.ceil((x1-x0)/500))
    panels=[]
    font_path="/System/Library/Fonts/Supplemental/Arial.ttf"
    try:font=ImageFont.truetype(font_path,18)
    except OSError:font=ImageFont.load_default(size=18)
    for i in range(count):
        left=max(x0,int(x0+i*(x1-x0)/count)-35)
        right=min(x1,int(x0+(i+1)*(x1-x0)/count)+35)
        local=[c for c in candidates if left<=c["pixel"][0]<right]
        if not local:continue
        top=max(y0,int(min(c["pixel"][1] for c in local))-115)
        bottom=min(y1,int(max(c["pixel"][1] for c in local))+80)
        factor=min(3.,800/(right-left),850/max(1,bottom-top))
        crop=image.crop((left,top,right,bottom)).resize((round((right-left)*factor),round((bottom-top)*factor)),Image.Resampling.LANCZOS)
        draw=ImageDraw.Draw(crop)
        for c in local:
            x,y=(c["pixel"][0]-left)*factor,(c["pixel"][1]-top)*factor
            draw.ellipse((x-4,y-4,x+4,y+4),outline="#cc2020",width=2)
            # Offset annotations downward, leaving the printed glyph intact.
            draw.text((x+7,y+8),c["id"],font=font,fill="#b00000",stroke_width=2,stroke_fill="white")
        panel_id=f"V{i+1}"
        path=folder/f"review-{panel_id}.png"
        crop.save(path)
        panels.append({"id":panel_id,"bounds_original":[left,top,right,bottom],"display_scale":factor,"candidate_ids":[c["id"] for c in local],"image_path":str(path)})
    return panels


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


def _marker_candidates(image: Image.Image, bounds: list[int], words: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Find bounded, symmetric marker outlines including pale and hollow marks."""
    rgb=np.asarray(image)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    color_mask=(hsv[:,:,1]>35).astype(np.uint8)*255
    mask=(gray<155).astype(np.uint8)*255
    x0,y0,x1,y1=bounds
    for current in (mask,color_mask):
        current[:y0]=0;current[y1:]=0;current[:,:x0]=0;current[:,x1:]=0
    # Suppress recognized printed text without erasing an overlapping marker.
    for word in words:
        if word["confidence"]<60 or not any(ch.isalnum() for ch in word["text"]):continue
        a,b,c,d=word["box"]
        if c-a>image.width*.3 or d-b>image.height*.2:continue
        mask[max(0,b-1):d+2,max(0,a-1):c+2]=0
    contours=[]
    for current in (color_mask,mask):
        found,_=cv2.findContours(current,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    result=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not(3<=w<=140 and 3<=h<=140 and .75<=w/h<=1.33):continue
        area=cv2.contourArea(contour)
        perimeter=cv2.arcLength(contour,True)
        if area<5 or perimeter<=0:continue
        hull_area=cv2.contourArea(cv2.convexHull(contour))
        if not hull_area or area/hull_area<.93:continue
        approx=cv2.approxPolyDP(contour,.03*perimeter,True)
        circularity=4*math.pi*area/perimeter**2
        square=len(approx)==4 and cv2.isContourConvex(approx)
        circle=circularity>=.77
        if not(square or circle or max(w,h)<=9):continue
        moments=cv2.moments(contour)
        cx,cy=moments["m10"]/moments["m00"],moments["m01"]/moments["m00"]
        if math.hypot(cx-(x+(w-1)/2),cy-(y+(h-1)/2))>max(1.5,.04*max(w,h)):continue
        result.append({"kind":"marker","pixel":[round(cx,2),round(cy,2)],"score":round(min(1.,area/hull_area),3),"bounds":[x,y,x+w,y+h],"detection_status":"validated_center","shape":"quadrilateral" if square else "round"})
    unique=[]
    for candidate in sorted(result,key=lambda c:(c["bounds"][2]-c["bounds"][0])*(c["bounds"][3]-c["bounds"][1]),reverse=True):
        if not any(math.dist(candidate["pixel"],old["pixel"])<3 for old in unique):unique.append(candidate)
    return sorted(unique,key=lambda c:(c["pixel"][0],c["pixel"][1]))


def _candidates(image: Image.Image, bounds: list[int], words: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    mask = _ink_mask(image, bounds, words)
    x0, y0, x1, y1 = bounds
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    result: list[dict[str, Any]] = []
    markers=_marker_candidates(image,bounds,words)
    result.extend(markers)
    # Local column-height maxima are candidate apexes. Restrict to connected ink
    # that extends close enough to the plot baseline to avoid title/label text.
    trace = _trace_mask(mask,bounds)[y0:y1,x0:x1]
    column_y = _profile(_trace_mask(mask,bounds),bounds)
    _,_,raw_stats,_=cv2.connectedComponentsWithStats(mask,8)
    wide_trace=any(s[cv2.CC_STAT_WIDTH]>(x1-x0)*.35 for s in raw_stats[1:])
    if wide_trace:
        result=[]  # isolated glyph holes beside a continuous trace are not scatter data
    peak_locations=_peak_locations(column_y,limit=limit) if wide_trace or len(markers)<3 else []
    for col,prominence in peak_locations:
        result.append({"kind":"peak", "pixel":[float(x0+col),float(column_y[col])],
                       "score":round(prominence,2), "prominence_pixels":round(prominence,2),
                       "detection_status":"validated_apex"})
    # No image-specific ppm or label rules are used here.
    result.sort(key=lambda c: c["score"], reverse=True)
    # Separate candidate budgets: peak prominence and marker geometry have
    # different units, so one must not push the other out of the candidate list.
    result = [c for c in result if c["kind"]=="peak"][:limit] + [c for c in result if c["kind"]=="marker"][:60]
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


def inspect_plot(image_path: str, measurement: str = "requested points", requested_labels: list[str] | None = None) -> dict[str, Any]:
    source = Path(image_path).expanduser().resolve(strict=True)
    image = _load_image(source)
    width, height = image.size
    words = _ocr(image)
    ticks = _tick_candidates(words, width, height)
    _snap_ticks_to_marks(image, ticks)
    bounds = _plot_bounds(ticks, width, height)
    candidates = _candidates(image, bounds, words)
    requested_labels=list(requested_labels or [])
    if len(requested_labels)>100 or any(not isinstance(v,str) or not v.strip() for v in requested_labels) or len({v.casefold() for v in requested_labels})!=len(requested_labels):
        raise ValueError("requested_labels must be unique nonempty strings, at most 100")
    if requested_labels:
        words.extend(_local_labels(image,bounds,candidates,requested_labels,words))
        candidates=_candidates(image,bounds,words)
    assignments=_associate_labels(candidates,words,requested_labels)
    session_id = uuid.uuid4().hex
    folder = SESSION_ROOT / session_id
    folder.mkdir(parents=True, exist_ok=False)
    reviews=_review_panels(image,bounds,candidates,folder) if requested_labels else []
    session = {"implementation_id": implementation_id(), "session_id": session_id, "image_path": str(source), "image_sha256":hashlib.sha256(source.read_bytes()).hexdigest(), "image_size": [width, height], "measurement": measurement, "bounds": bounds, "ticks": ticks, "words": words, "candidates": candidates,"requested_labels":requested_labels,"label_assignments":assignments,"review_regions":reviews}
    mask=_ink_mask(image,bounds,words)
    trace=_trace_mask(mask,bounds)
    np.savez_compressed(folder/"analysis.npz",trace=trace,profile=_profile(trace,bounds),curve_mask=_ink_mask(image,bounds,words,False))
    (folder / "session.json").write_text(json.dumps(session, indent=2))
    overview = folder / "overview.png"
    _draw_overview(image, bounds, ticks, candidates, overview)
    needed=[a for a in assignments.values() if a["status"]=="needs_review"]
    used_boxes={tuple(a["matched_label_box"]) for a in assignments.values() if a.get("matched_label_box")}
    leftover=[w["box"] for w in words if w.get("source")=="local_glyph_ocr" and tuple(w["box"]) not in used_boxes]
    focus_boxes=[box for a in needed for box in a["label_boxes"]] or leftover
    priority=[r for r in reviews if any(r["bounds_original"][0]<=(b[0]+b[2])/2<r["bounds_original"][2] and r["bounds_original"][1]<=(b[1]+b[3])/2<r["bounds_original"][3] for b in focus_boxes)]
    review_paths=[str(overview)]+[r["image_path"] for r in (priority or reviews)] if needed else [str(overview)]
    return {"implementation_id": implementation_id(), "session_id": session_id, "image_size": [width, height], "plot_bounds": bounds, "ticks": ticks, "candidates": candidates, "label_assignments":assignments,"review_regions":reviews,"review_image_paths":review_paths, "ocr_text": [w for w in words if w["confidence"] >= 35][:80], "overview_path": str(overview), "warning": "Prominence is not assignment confidence. Review needs_review labels in the local panels and record review_id plus review_note in the selection. Coordinate uncertainty assumes the correct feature."}


def _read_session(session_id: str) -> tuple[Path, dict[str, Any], Image.Image]:
    folder = _path(session_id)
    session = json.loads((folder / "session.json").read_text())
    if session.get("implementation_id") != implementation_id():
        raise ValueError("session implementation differs from this tool; inspect the image again")
    if session.get("image_sha256") != hashlib.sha256(Path(session["image_path"]).read_bytes()).hexdigest():
        raise ValueError("source image changed; inspect it again")
    return folder, session, _load_image(session["image_path"])


def _session_analysis(folder: Path, session: dict[str,Any]) -> dict[str,Any]:
    with np.load(folder/"analysis.npz",allow_pickle=False) as saved:
        result={key:saved[key] for key in saved.files}
    result["markers"]=[c for c in session["candidates"] if c["kind"]=="marker"]
    result["peaks"]=_peak_locations(result["profile"],prominence=2)
    return result


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
    trace=_session_analysis(folder,session)["trace"]
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
    review_id=f"Rview{index}"
    session.setdefault("review_regions",[]).append({"id":review_id,"bounds_original":bounds,"image_path":str(path)})
    (folder/"session.json").write_text(json.dumps(session,indent=2))
    candidates = [c for c in session["candidates"] if x0 <= c["pixel"][0] < x1 and y0 <= c["pixel"][1] < y1]
    return {"implementation_id":implementation_id(),"session_id": session_id, "review_id":review_id,"bounds_original": bounds, "display_scale": factor, "crop_path": str(path), "candidates": candidates}


def _resolve_point(selection: dict[str, Any], candidates: list[dict[str, Any]], image: Image.Image, bounds: list[int], words: list[dict[str, Any]], analysis: dict[str,Any] | None = None) -> tuple[float, float, str, float]:
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
    if mode == "peak" or (is_peak and mode!="scatter"):
        radius = int(selection.get("radius",8))
        if not 1<=radius<=30:
            raise ValueError("peak radius must be 1..30 pixels")
        profile=analysis["profile"] if analysis is not None else _profile(_trace_mask(_ink_mask(image,bounds,words),bounds),bounds)
        at=round(x)-bounds[0]
        if mode=="peak":
            local=[(px,score) for px,score in (analysis["peaks"] if analysis is not None else _peak_locations(profile,prominence=2))
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
    elif mode == "scatter" or method=="candidate:marker":
        markers=analysis["markers"] if analysis is not None else _marker_candidates(image,bounds,words)
        nearby=[]
        for marker in markers:
            a,b,c,d=marker["bounds"]
            distance=math.hypot(marker["pixel"][0]-x,marker["pixel"][1]-y)
            if a-2<=x<c+2 and b-2<=y<d+2 or distance<=5:
                nearby.append((distance,marker))
        nearby.sort(key=lambda pair:pair[0])
        if not nearby:
            raise UnresolvedFeature("no symmetric marker center supported at selection; overlapping or broken markers need review")
        if len(nearby)>1 and nearby[1][0]-nearby[0][0]<3:
            raise UnresolvedFeature("selection overlaps competing markers")
        x,y=nearby[0][1]["pixel"]
        method = "scatter_center"
        uncertainty = 2.0
    elif mode == "curve":
        method = "curve_position"
        uncertainty = 2.0
    elif mode != "exact":
        raise ValueError("mode must be exact, peak, scatter, or curve")
    return float(x), float(y), method, float(uncertainty)


def _curve_at_x(image: Image.Image, bounds: list[int], words: list[dict[str, Any]], x: float, hint_y: float | None, mask: np.ndarray | None = None) -> tuple[float, float]:
    x0,y0,x1,y1=bounds
    if not x0 <= x < x1:
        raise ValueError("curve query is outside the visible plot")
    if mask is None:mask=_ink_mask(image,bounds,words,remove_lines=False)
    ys=[]
    for px in range(max(x0,round(x)-2),min(x1,round(x)+3)):
        ys.extend(np.flatnonzero(mask[y0:y1,px])+y0)
    if not ys:
        raise UnresolvedFeature("no visible curve at requested x position")
    values=np.asarray(sorted(ys))
    gaps=np.flatnonzero(np.diff(values)>5)+1
    clusters=[c for c in np.split(values,gaps) if len(c)>=2]
    if not clusters:
        raise UnresolvedFeature("no continuous curve segment at requested x position")
    centers=[float(np.median(c)) for c in clusters]
    if hint_y is None and len(centers)>1:
        raise UnresolvedFeature("multiple curves at requested x; provide pixel:[x,approx_y] to identify the series")
    index=0 if hint_y is None else int(np.argmin([abs(c-hint_y) for c in centers]))
    if hint_y is not None and abs(centers[index]-hint_y)>25:
        raise UnresolvedFeature("selected curve is too far from supplied series hint")
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


def _unresolved_row(label: str, reason: str, alternatives: list[str] | None = None) -> dict[str,Any]:
    return {"label":label,"status":"unresolved","detection_status":"unsupported_or_unreviewed",
            "assignment_status":"unresolved","reason":reason,"alternatives":alternatives or [],
            "x":None,"y":None,"pixel":None,"pixel_uncertainty":None,
            "x_uncertainty_transformed":None,"y_uncertainty_transformed":None,"method":"unresolved"}


def _assignment_evidence(session: dict[str,Any], selection: dict[str,Any]) -> tuple[str,str | None,list[str]]:
    label=str(selection["label"])
    assignment=next((value for key,value in session.get("label_assignments",{}).items() if key.casefold()==label.casefold()),None)
    if assignment is None:
        return "model_selected",None,[]
    alternatives=assignment.get("candidate_ids",[])
    if assignment["status"]=="supported" and selection.get("candidate_id")==next(iter(alternatives),None):
        return "ocr_supported",None,alternatives
    review=next((r for r in session.get("review_regions",[]) if r["id"]==selection.get("review_id")),None)
    candidate=next((c for c in session["candidates"] if c["id"]==selection.get("candidate_id")),None)
    pixel=candidate["pixel"] if candidate else selection.get("pixel")
    if review and pixel and str(selection.get("review_note","")).strip():
        x0,y0,x1,y1=review["bounds_original"]
        if x0<=pixel[0]<x1 and y0<=pixel[1]<y1:
            return "visually_selected",None,alternatives
    return "unresolved","label association needs local image review; supply review_id and review_note, or mark unresolved",alternatives


def measure_plot(session_id: str, x_axis: dict[str, Any], selections: list[dict[str, Any]], y_axis: dict[str, Any] | None = None, reference_label: str | None = None, baseline_pixel: float | None = None, use_label_proposals: bool = False) -> dict[str, Any]:
    folder, session, image = _read_session(session_id)
    analysis=_session_analysis(folder,session)
    selections=list(selections)
    if use_label_proposals:
        given={str(s["label"]).casefold() for s in selections}
        selections=[{"label":label,"candidate_id":a["candidate_ids"][0]} for label,a in session.get("label_assignments",{}).items() if a["status"]=="supported" and label.casefold() not in given]+selections
    if not selections and not session.get("requested_labels"):
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
        if selection.get("status")=="unresolved":
            rows.append(_unresolved_row(label,selection.get("reason","image does not resolve this feature")))
            continue
        assignment_status,assignment_error,alternatives=_assignment_evidence(session,selection)
        if assignment_error:
            rows.append(_unresolved_row(label,assignment_error,alternatives))
            continue
        if selection.get("candidate_id") and len(label) == 1 and label.isalpha():
            selected_candidate = next((c for c in session["candidates"] if c["id"] == selection["candidate_id"]), None)
            if selected_candidate and selected_candidate.get("label_hint") and selected_candidate["label_hint"] != label.upper() and selected_candidate.get("label_hint_confidence", 0) >= 75:
                raise ValueError(f"selected {selection['candidate_id']} is near printed label {selected_candidate['label_hint']}, not {label}; verify image label assignment")
        if selection.get("mode") == "curve" and "x_value" in selection:
            x=xmap.pixel_for(float(selection["x_value"]))
            hint_y=float(selection["pixel"][1]) if "pixel" in selection else None
            try:
                y,pixel_uncertainty=_curve_at_x(image,session["bounds"],session["words"],x,hint_y,analysis["curve_mask"])
            except UnresolvedFeature as error:
                rows.append(_unresolved_row(label,str(error)))
                continue
            method="curve_trace_at_x"
        else:
            try:
                x, y, method, pixel_uncertainty = _resolve_point(selection, session["candidates"], image, session["bounds"], session["words"],analysis)
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
        rows.append({"label": label, "status":"resolved", "detection_status":"validated_apex" if "peak" in method else "selected_feature", "assignment_status":assignment_status, "review_id":selection.get("review_id"), "review_note":selection.get("review_note"), "coordinate_uncertainty_note":"conditional on correct feature assignment", "x": xp, "y": yp, "pixel": [x, y], "x_uncertainty_transformed": xe, "y_uncertainty_transformed": ye, "pixel_uncertainty": pixel_uncertainty, "method": method})
    for requested in session.get("requested_labels",[]):
        if requested.casefold() not in {r["label"].casefold() for r in rows}:
            rows.append(_unresolved_row(requested,"requested label was omitted from selections"))
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
    unresolved=[r["label"] for r in rows if r.get("status")=="unresolved"]
    if unresolved:
        draw.rectangle((0,0,min(image.width,800),30),fill="white")
        draw.text((8,8),"Unresolved: "+", ".join(unresolved),fill="#a00000")
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
