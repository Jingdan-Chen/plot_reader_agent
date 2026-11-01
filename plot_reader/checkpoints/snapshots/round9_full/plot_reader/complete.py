"""Experimental complete-figure extraction. Legacy measurements remain unchanged.

Coordinates in every interface are original-image pixels. Model decisions are
kept separate from raster measurements and axis verification.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from . import core

SCHEMA = "complete-plot-v2"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def bounds_checked(bounds: list, image: Image.Image) -> list[int]:
    if len(bounds) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds):
        raise ValueError("bounds must contain four finite original-image coordinates")
    x0, y0, x1, y1 = map(round, bounds)
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("bounds must lie inside the source image")
    return [x0, y0, x1, y1]


def validate_plan(image_path: str, plan: dict) -> list[str]:
    image=core._load_image(image_path)
    errors=[]
    ids=set()
    if not plan.get("panels"):
        return ["no Cartesian panel was inventoried"]
    for pi,panel in enumerate(plan["panels"]):
        loc=f"panels[{pi}]"
        try:
            pb=bounds_checked(panel["bounds"],image)
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(f"{loc}: {exc}; bounds order is [left,top,right,bottom]");continue
        if panel.get("id") in ids:errors.append(loc+": duplicate panel ID")
        ids.add(panel.get("id"))
        for si,series in enumerate(panel.get("series",[])):
            sloc=f"{loc}.series[{si}]"
            try:
                sb=bounds_checked(series.get("bounds",pb),image)
            except (ValueError,KeyError,TypeError) as exc:
                errors.append(f"{sloc}: {exc}");continue
            if not(pb[0]<=sb[0]<sb[2]<=pb[2] and pb[1]<=sb[1]<sb[3]<=pb[3]):errors.append(sloc+": bounds outside parent panel")
            for seed in series.get("seeds",[]):
                if len(seed)!=2 or not sb[0]<=seed[0]<sb[2] or not sb[1]<=seed[1]<sb[3]:
                    errors.append(f"{sloc}: seed {seed} outside bounds {sb}; choose interior actual ink")
                    continue
                if series.get('kind','curve')=='curve':
                    mask=_series_mask(image,sb,[],series)
                    ys,xs=np.where(mask>0)
                    if not len(xs):errors.append(sloc+': selected bounds/style contain no trace ink');continue
                    distance=np.hypot(xs-seed[0],ys-seed[1]);at=int(distance.argmin())
                    if distance[at]>8:
                        errors.append(f"{sloc}: seed {seed} is {distance[at]:.1f}px away from selected ink; nearest observed pixel [{int(xs[at])},{int(ys[at])}]. Verify series identity on image before choosing a replacement.")
        for name,limits in (("x_axis",(pb[0],pb[2])),("y_axis",(pb[1],pb[3]))):
            spec=panel.get(name)
            if spec:
                cal=Calibration(spec,limits)
                if not cal.evidence["verified"] and not cal.evidence.get("partially_verified"):errors.append(f"{loc}.{name}: {cal.evidence['reason']}; set null if evidence is insufficient")
        for si,series in enumerate(panel.get('series',[])):
            if series.get('baseline'):
                sb=series.get('bounds',pb)
                for anchor in series['baseline'].get('anchors',[]):
                    if len(anchor)!=2 or not sb[0]<=anchor[0]<=sb[2] or not sb[1]<=anchor[1]<=sb[3]:
                        errors.append(f'{loc}.series[{si}]: baseline anchor {anchor} outside series bounds; verify anchors or extend the reviewed series bounds')
    return errors


def axis_hypotheses(ticks: list[dict], tolerance: float = 3.0) -> dict:
    """Fit two end anchors and verify ALL remaining observed ticks in pixels."""
    usable = sorted([t for t in ticks if all(isinstance(t.get(k), (int, float)) and math.isfinite(t[k])
                                           for k in ("pixel", "value"))], key=lambda t: t["pixel"])
    usable = [t for i, t in enumerate(usable) if i == 0 or abs(t["pixel"] - usable[i-1]["pixel"]) > 2]
    proposals = []
    if len(usable) >= 3:
        for scale in ("linear", "log10"):
            if scale == "log10" and any(t["value"] <= 0 for t in usable):
                continue
            spec = {"anchors": [{k: t[k] for k in ("pixel", "value")} for t in (usable[0], usable[-1])],
                    "checks": [{k: t[k] for k in ("pixel", "value")} for t in usable[1:-1]], "scale": scale}
            try:
                axis = core.Axis(**spec)
                residual = max(abs(axis.pixel_for(t["value"]) - t["pixel"]) for t in usable[1:-1])
            except (ValueError, OverflowError):
                continue
            proposals.append({**spec, "pixel_residual": residual, "verified": residual <= tolerance})
    proposals.sort(key=lambda p: p["pixel_residual"])
    good = [p for p in proposals if p["verified"]]
    # Both mappings can fit a tiny numeric range; do not pretend this identifies scale.
    unique = len(good) == 1 or (len(good) == 2 and good[1]["pixel_residual"] - good[0]["pixel_residual"] > 1)
    return {"selected": good[0] if good and unique else None, "hypotheses": proposals,
            "status": "verified" if good and unique else "needs_review",
            "reason": None if good and unique else "insufficient independent ticks or competing axis mappings"}


def segmented_axis_proposals(ticks: list[dict]) -> list[dict]:
    """Numeric discontinuity proposals; a visible break still needs image review."""
    ticks=sorted(ticks,key=lambda t:t['pixel'])
    proposals=[]
    if axis_hypotheses(ticks)['selected'] is not None:return proposals
    for at in range(3,len(ticks)-2):
        groups=(ticks[:at],ticks[at:]);parts=[]
        for i,group in enumerate(groups):
            hypothesis=axis_hypotheses(group)['selected']
            if hypothesis is None:break
            parts.append({'id':f'S{i+1}','pixel_bounds':[group[0]['pixel'],group[-1]['pixel']],
                          'anchors':hypothesis['anchors'],'checks':hypothesis['checks'],'scale':hypothesis['scale']})
        if len(parts)==2:
            proposals.append({'segments':parts,'omitted_pixel_interval':[ticks[at-1]['pixel'],ticks[at]['pixel']],
                              'status':'needs_review','evidence':'each tick group fits independently; whole axis fails',
                              'requires':'verify visible axis break; never infer a break from absent tick labels alone'})
    return proposals


def series_proposals(image: Image.Image,bounds: list[int],words: list[dict]) -> list[dict]:
    """Style-separated inventories to review, without treating text as data."""
    x0,y0,x1,y1=bounds;rgb=np.asarray(image);roi=rgb[y0:y1,x0:x1]
    hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
    colored=(hsv[:,:,1]>55)&(hsv[:,:,2]>55)
    quantized=(roi//32).astype(int);unique,counts=np.unique(quantized[colored],axis=0,return_counts=True)
    colors=[]
    for at in np.argsort(counts)[::-1]:
        if counts[at]<10:continue
        color=(unique[at]*32+16).clip(0,255)
        if not any(np.linalg.norm(color-c)<90 for c in colors):colors.append(color)
    proposals=[]
    styles=[None]+[c.tolist() for c in colors]
    for color in styles:
        mask=_series_mask(image,bounds,[],{'color_rgb':color})
        if color is None:
            full_hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask[full_hsv[:,:,1]>55]=0
        for word in words:
            if word.get('confidence',0)>=70:
                a,b,c,d=word['box'];mask[max(y0,b):min(y1,d),max(x0,a):min(x1,c)]=0
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        eligible=[i for i,s in enumerate(stats[1:],1) if s[4]>=4 and s[2]>=3]
        if not eligible:continue
        curve=[i for i in eligible if stats[i][2]>(x1-x0)*.12 or stats[i][3]>30]
        markers=[i for i in eligible if i not in curve and 3<=stats[i][2]<=20 and 3<=stats[i][3]<=20]
        for kind,components in (('curve',curve),('markers',markers)):
            if not components:continue
            ys,xs=np.where(np.isin(labels,components))
            seeds=[]
            for offset in (.2,.8):
                near=np.argmin(abs(xs-np.quantile(xs,offset)));seeds.append([int(xs[near]),int(ys[near])])
            proposals.append({'id':f'T{len(proposals)+1}','kind':kind,'color_rgb':color,
                              'bounds':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                              'representative_ink_seeds':seeds,'visible_components':len(components),'status':'needs_review',
                              'reason':'style grouping may contain multiple same-color or overlapping series'})
    return proposals


class Calibration:
    def __init__(self, spec: dict | None, bounds: tuple[float, float]):
        self.parts: list[tuple[str, tuple[float, float], core.Axis]] = []
        self.unit = (spec or {}).get("unit")
        self.evidence: dict = {"verified": False, "reason": "axis not supplied"}
        if not spec:return
        segments=spec.get("segments") or [{"id":"S1","pixel_bounds":list(bounds),**spec}]
        records=[];ids=set();geometry=[]
        try:
            for i,segment in enumerate(segments):
                sid=str(segment.get('id',f'S{i+1}'))
                if sid in ids:raise ValueError('axis segment IDs must be unique')
                ids.add(sid)
                lo,hi=map(float,segment['pixel_bounds'])
                if not(bounds[0]<=lo<hi<=bounds[1]):raise ValueError('segments must lie inside panel bounds')
                geometry.append((sid,(lo,hi),segment))
            if any(geometry[i][1][0]<=geometry[i-1][1][1] for i in range(1,len(geometry))):
                raise ValueError('segments must be ordered and disjoint')
        except (KeyError,TypeError,ValueError) as exc:
            self.evidence={'verified':False,'reason':str(exc),'unit':self.unit};return
        for sid,(lo,hi),segment in geometry:
            try:
                axis=core.Axis(anchors=segment['anchors'],scale=segment.get('scale','linear'),checks=segment.get('checks'))
                if not axis.checks:raise ValueError('each segment needs an independent check tick')
                if any(not lo<=float(t['pixel'])<=hi for t in axis.anchors+axis.checks):raise ValueError('tick lies outside its segment')
                if any(abs(c['pixel']-a['pixel'])<2 for c in axis.checks for a in axis.anchors):raise ValueError('check tick must be independent of anchors')
                residual=max(abs(axis.pixel_for(c['value'])-c['pixel']) for c in axis.checks)
                if residual>3:raise ValueError('independent tick disagrees by more than three pixels')
                self.parts.append((sid,(lo,hi),axis))
                records.append({'id':sid,'pixel_bounds':[lo,hi],'anchors':axis.anchors,'checks':axis.checks,'scale':axis.scale,'pixel_residual':residual,'verified':True})
            except (KeyError,TypeError,ValueError,OverflowError) as exc:
                records.append({'id':sid,'pixel_bounds':[lo,hi],'verified':False,'reason':str(exc)})
        failed=[r for r in records if not r['verified']]
        self.evidence={'verified':not failed,'partially_verified':bool(self.parts) and bool(failed),
                       'reason':'; '.join(r['id']+': '+r['reason'] for r in failed) or None,
                       'segments':records,'unit':self.unit,
                       'unresolved_pixel_ranges':[r['pixel_bounds'] for r in failed]}

    def convert(self, pixel: float) -> tuple[float | None, str | None]:
        for sid, (lo, hi), axis in self.parts:
            if lo <= pixel <= hi:
                return axis.convert(pixel), sid
        return None, None

    def pixel_for(self, value: float) -> float | None:
        pixels = []
        for _, (lo, hi), axis in self.parts:
            try:
                p = axis.pixel_for(value)
            except (ValueError, OverflowError):
                continue
            if lo <= p <= hi:
                pixels.append(p)
        return pixels[0] if len(pixels) == 1 else None


def panel_proposals(image: Image.Image) -> list[dict]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    ink = (gray < 160).astype(np.uint8)*255
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, image.width//5), 1)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, 8)
    lines = sorted([tuple(map(int, s[:4])) for s in stats[1:]
                    if s[2] >= image.width*.22 and s[3] <= 8], key=lambda s: s[1])
    proposals = []
    for x, y, width, _ in lines:
        if y < image.height*.12:
            continue
        # A horizontal stroke is a proposal, not a verified Cartesian panel.
        prior = [p for p in lines if p[1] < y-20 and abs(p[0]-x) < 30 and abs(p[2]-width) < 40]
        top = max(0, max((p[1]+5 for p in prior), default=y-max(80, round(width*.65))))
        if y-top < 40:
            continue
        if any(abs(p["bounds"][3]-y) < 12 and abs(p["bounds"][0]-x) < 12 for p in proposals):
            continue
        proposals.append({"id": f"P{len(proposals)+1}", "bounds": [x, top, min(image.width,x+width), y],
                          "status": "needs_review", "evidence": "long horizontal axis-like stroke"})
    return proposals


def discover_plot(image_path: str, output_dir: str) -> dict:
    image = core._load_image(image_path)
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    words = core._ocr(image)
    panels = panel_proposals(image)
    for panel in panels:
        x0,y0,x1,y1 = panel["bounds"]
        region = image.crop((max(0,x0-55), max(0,y0-10), min(image.width,x1+20), min(image.height,y1+60)))
        ox, oy = max(0,x0-55), max(0,y0-10)
        local = [{**w,"box":[w["box"][0]-ox,w["box"][1]-oy,w["box"][2]-ox,w["box"][3]-oy]}
                 for w in words if ox<=w["box"][0]<x1+20 and oy<=w["box"][1]<y1+60]
        ticks = core._tick_candidates(local, region.width, region.height)
        core._snap_ticks_to_marks(region,ticks)
        for axis, offset in (("x",ox),("y",oy)):
            for tick in ticks[axis]:
                tick["pixel"] += offset
                tick["box"] = [tick["box"][0]+ox,tick["box"][1]+oy,tick["box"][2]+ox,tick["box"][3]+oy]
        panel["ticks"] = ticks
        panel["axis_proposals"] = {axis:axis_hypotheses(ticks[axis]) for axis in ("x","y")}
        panel['segmented_axis_proposals']={axis:segmented_axis_proposals(ticks[axis]) for axis in ('x','y')}
        panel['series_proposals']=series_proposals(image,panel['bounds'],words)
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    for panel in panels:
        draw.rectangle(panel["bounds"],outline="red",width=2)
        draw.text(tuple(panel["bounds"][:2]),panel["id"],fill="red",stroke_width=1,stroke_fill="white")
    overview.save(folder/"overview.png")
    result = {"schema_version":SCHEMA,"source_image":str(Path(image_path).resolve()),
              "source_sha256":hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
              "size":list(image.size),"panels":panels,"ocr":words,
              "review_image_paths":[str(folder/"overview.png")],
              "discovery_complete":False,"note":"Model must verify panels, recover missed panels and inventory every series."}
    write_json(folder/"discovery.json",result)
    return result


def _series_mask(image: Image.Image, bounds: list[int], words: list[dict], spec: dict) -> np.ndarray:
    # Whole-image OCR can label trace strokes as confident text. Reviewed
    # exclusions and component/style ownership remove text without punching
    # inferred holes through an unreviewed curve.
    mask = core._ink_mask(image,bounds,[])
    rgb = np.asarray(image).astype(float)
    color = spec.get("color_rgb")
    if color is not None:
        if len(color)!=3 or any(not 0<=float(c)<=255 for c in color):
            raise ValueError("color_rgb must contain three channels between 0 and 255")
        distance = np.linalg.norm(rgb-np.array(color),axis=2)
        mask = ((distance < float(spec.get("color_tolerance",100))) & (mask>0)).astype(np.uint8)*255
    if spec.get("exclude_bounds"):
        for b in spec["exclude_bounds"]:
            a,c,d,e = bounds_checked(b,image)
            mask[c:e,a:d] = 0
    return mask


def trace_path(mask: np.ndarray, bounds: list[int], seeds: list[list[float]], legacy: bool = False, evidence: dict | None = None) -> tuple[np.ndarray,np.ndarray]:
    """Track raster run centers without interpolating absent columns."""
    x0,y0,x1,y1=bounds
    if legacy:
        profile=core._profile(core._trace_mask(mask,bounds),bounds)
        return profile,np.zeros(len(profile),dtype=bool)
    selected = mask.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    if seeds:
        keep=set()
        for sx,sy in seeds:
            candidates=[(abs(sx-(s[0]+s[2]/2))*.05+abs(sy-(s[1]+s[3]/2)),i)
                        for i,s in enumerate(stats[1:],1) if s[4]>=4 and s[2]>=3]
            # Distance to actual ink, not component bounding-box center.
            distances=[]
            for _,i in candidates:
                s=stats[i]
                if s[0]-8<=sx<=s[0]+s[2]+8 and s[1]-8<=sy<=s[1]+s[3]+8:
                    ys,xs=np.where(labels[max(y0,round(sy)-12):min(y1,round(sy)+13),
                                           max(x0,round(sx)-12):min(x1,round(sx)+13)]==i)
                    if len(xs):
                        distances.append((float(np.min(np.hypot(xs+max(x0,round(sx)-12)-sx,ys+max(y0,round(sy)-12)-sy))),i))
            if distances:
                keep.add(min(distances)[1])
        if keep:
            # A dashed or occluded series can own several components. Extend
            # ownership by local endpoint geometry, never filling their gaps.
            candidates={i for i,s in enumerate(stats[1:],1) if s[4]>=4 and s[2]>=3}
            changed=True
            while changed:
                changed=False
                for i in candidates-keep:
                    a=stats[i]
                    for j in list(keep):
                        b=stats[j]
                        left,right=(a,b) if a[0]<b[0] else (b,a)
                        gap=right[0]-(left[0]+left[2])
                        vertical=max(0,max(left[1],right[1])-min(left[1]+left[3],right[1]+right[3]))
                        if 0<=gap<=max(16,(x1-x0)*.03) and vertical<=max(8,gap*1.5):
                            keep.add(i);changed=True;break
            selected=(np.isin(labels,list(keep))*255).astype(np.uint8)
        else:
            return np.full(x1-x0,np.nan),np.ones(x1-x0,dtype=bool)
    if evidence is not None:
        evidence.update({"source_components":count-1,"retained_components":len(np.unique(labels[selected>0]))-int(0 in labels[selected>0]),
                         "identity_method":"seed anchors and local component continuation" if seeds else "reviewed style/bounds",
                         "gap_policy":"absent ink remains unmeasured","run_bounds":{}})
    rows=[]
    spans=[]
    for x in range(x0,x1):
        ys=np.flatnonzero(selected[y0:y1,x])+y0
        groups=np.split(ys,np.flatnonzero(np.diff(ys)>1)+1) if len(ys) else []
        rows.append(np.array([np.median(g) for g in groups],dtype=float))
        spans.append([[int(g[0]),int(g[-1])] for g in groups])
    path=np.full(x1-x0,np.nan)
    ambiguous=np.zeros(x1-x0,dtype=bool)
    observed=[i for i,r in enumerate(rows) if len(r)]
    if not observed:
        return path,ambiguous
    groups=np.split(observed,np.flatnonzero(np.diff(observed)>12)+1)
    for group in groups:
        costs=[];backs=[]
        for n,i in enumerate(group):
            values=rows[i]
            seed_cost=np.zeros(len(values))
            for sx,sy in seeds:
                if abs((x0+i)-sx)<=3:
                    seed_cost+=abs(values-sy)*2
            if n==0:
                if seeds:
                    sy=min(seeds,key=lambda s:abs(s[0]-(x0+i)))[1]
                    seed_cost+=abs(values-sy)*.05
                costs.append(seed_cost);backs.append(np.zeros(len(values),dtype=int))
            else:
                j=group[n-1]
                transition=costs[-1][:,None]+abs(rows[j][:,None]-values[None,:])*.04/max(1,i-j)
                back=transition.argmin(axis=0)
                costs.append(transition[back,np.arange(len(values))]+seed_cost);backs.append(back)
        at=int(costs[-1].argmin())
        for n in range(len(group)-1,-1,-1):
            i=group[n]
            path[i]=rows[i][at]
            if evidence is not None:evidence["run_bounds"][int(i)]=spans[i][at]
            # Multiple separated paths need evidence; nearby stroke runs are one trace.
            ambiguous[i]=len(rows[i])>1 and np.sum(abs(rows[i]-path[i])<6)>1
            at=int(backs[n][at])
    return path,ambiguous


def baseline_model(path: np.ndarray, bounds: list[int], spec: dict, ambiguous: np.ndarray, reference_features: list[dict] | None = None, clip_bounds: list[int] | None = None) -> dict:
    valid=np.isfinite(path)&~ambiguous
    if valid.sum()<20:
        return {"valid":False,"reason":"insufficient observed trace for normalization"}
    supplied=spec.get("baseline")
    xs=np.flatnonzero(valid)
    if supplied:
        anchors=supplied.get("anchors",[])
        if len(anchors)<2:
            return {"valid":False,"reason":"baseline needs two image-supported anchors"}
        xx=np.array([a[0]-bounds[0] for a in anchors]);yy=np.array([a[1] for a in anchors])
        if any(not bounds[0]<=a[0]<=bounds[2] or not bounds[1]<=a[1]<=bounds[3] for a in anchors):
            return {"valid":False,"reason":"baseline anchors outside panel"}
        if np.ptp(xx)<10:
            return {"valid":False,"reason":"baseline anchors too close"}
        for x,y in zip(xx,yy):
            nearby=np.flatnonzero(valid & (abs(np.arange(len(path))-x)<=4))
            if not len(nearby) or np.min(abs(path[nearby]-y))>4:
                evidence=spec.get('_baseline_stroke_evidence',[])
                if not any(abs(e['pixel_xy'][0]-(bounds[0]+x))<=1 and abs(e['pixel_xy'][1]-y)<=1 for e in evidence):
                    return {'valid':False,'reason':'baseline anchor lacks nearby observed trace or independently visible baseline stroke support'}
        coef=np.polyfit(xx,yy,1)
        source="reviewed_baseline_anchors"
    else:
        bins=np.array_split(xs,min(12,max(3,len(xs)//20)))
        bx=np.array([np.median(b) for b in bins]);by=np.array([np.quantile(path[b],.9) for b in bins])
        coef=np.polyfit(bx,by,1)
        # Robustly refit baseline-supported bins only; broad peaks are not baseline.
        residual=by-np.polyval(coef,bx)
        keep=residual>=np.quantile(residual,.3)
        coef=np.polyfit(bx[keep],by[keep],1) if keep.sum()>=3 else coef
        source="robust_lower_envelope"
    baseline=np.polyval(coef,np.arange(len(path)))
    heights=baseline-path
    apex=int(np.nanargmax(np.where(valid,heights,np.nan)))
    reference=[bounds[0]+apex,float(path[apex])]
    if reference_features is not None:
        supported=[f for f in reference_features if f['feature_type']=='peak' and f['status']=='resolved' and f.get('prominence_pixels',0)>=3]
        if not supported:return {'valid':False,'reason':'no supported spectral reference apex','baseline_model':coef.tolist(),'baseline_origin_pixel_x':bounds[0]}
        reference=max(supported,key=lambda f:np.polyval(coef,f['original_pixel_xy'][0]-bounds[0])-f['original_pixel_xy'][1])['original_pixel_xy']
    amplitude=float(np.polyval(coef,reference[0]-bounds[0])-reference[1])
    if clip_bounds and reference[1]<=clip_bounds[1]+2:
        return {'valid':False,'reason':'reference apex clipped at panel boundary','baseline_model':coef.tolist(),'baseline_origin_pixel_x':bounds[0]}
    near=valid&(abs(heights)<max(2,amplitude*.04))
    noise=float(np.median(abs(heights[near]))) if near.sum()>=5 else None
    quiet_windows=[]
    for window in np.array_split(np.arange(len(path)),12):
        quiet=window[near[window]]
        if len(quiet)>=5:quiet_windows.append({'pixel_x_bounds':[int(bounds[0]+quiet.min()),int(bounds[0]+quiet.max())],
                                              'observed_samples':len(quiet),'median_residual_pixels':float(np.median(abs(heights[quiet])))})
    independently_supported_flat_line=len(spec.get('_baseline_stroke_evidence',[]))>=3 and abs(coef[0])<.005
    required_windows=2 if independently_supported_flat_line else 3
    if reference_features is not None and (len(quiet_windows)<required_windows or
            max(w['pixel_x_bounds'][1] for w in quiet_windows)-min(w['pixel_x_bounds'][0] for w in quiet_windows)<len(path)*.6):
        return {'valid':False,'reason':'insufficient separated baseline support across series domain',
                'baseline_model':coef.tolist(),'baseline_origin_pixel_x':bounds[0],'quiet_windows':quiet_windows}
    if amplitude<=3 or noise is None or near.sum()<max(5,valid.sum()*.03) or noise>amplitude*.05:
        return {"valid":False,"reason":"baseline or dominant apex lacks precision","baseline_model":coef.tolist(),"source":source}
    return {"valid":True,"baseline_model":coef.tolist(),"baseline_noise_pixels":noise,
            "baseline_origin_pixel_x":bounds[0],"series_domain_original_pixel_x":[bounds[0],bounds[2]],
            "coefficient_order":["slope","intercept"],"coordinate_system":"original-image pixels","y_direction":"down",
            "coefficient_units":["pixel_y/pixel_x","pixel_y"],"quiet_windows":quiet_windows,
            "independent_baseline_stroke_support":spec.get('_baseline_stroke_evidence',[]),
            "dominant_peak_original_pixel_xy":reference,
            "baseline_original_pixel_y":float(np.polyval(coef,reference[0]-bounds[0])),"dominant_peak_pixel":reference[1],
            "apex_validation":{"validated":True,"method":"multiscale_trace" if reference_features is not None else "dense_trace","pixel_y":reference[1]},
            "amplitude_pixels":amplitude,"source":source,
            "formula":"baseline evaluated at each original sample x; reference amplitude at reference x",
            "scope":"all visible components of this series; omitted axis ranges excluded"}


def normalize_observation(px: float,py: float,evidence: dict,ycal: Calibration) -> dict:
    if not evidence.get('valid'):return {'normalization_status':'unresolved'}
    baseline=float(np.polyval(evidence['baseline_model'],px-evidence['baseline_origin_pixel_x']))
    contrast=(baseline-py)/evidence['amplitude_pixels']
    values={'normalized_pixel_height':contrast,'normalization_status':'verified_visible_pixel_contrast'}
    if not ycal.parts:
        values.update({'normalized_height':contrast,'normalization_basis':'visible_pixel_amplitude','physical_normalization_verified':False})
        return values
    ref=evidence['dominant_peak_original_pixel_xy'];ref_base=float(np.polyval(evidence['baseline_model'],ref[0]-evidence['baseline_origin_pixel_x']))
    yy,_=ycal.convert(py);base,_=ycal.convert(baseline);ry,_=ycal.convert(ref[1]);rb,_=ycal.convert(ref_base)
    if any(v is None for v in (yy,base,ry,rb)) or abs(ry-rb)<1e-12:
        values.update({'normalization_status':'unresolved_physical_intensity','normalization_reason':'baseline/sample/reference lacks common verified calibration'})
        return values
    values.update({'normalized_height':(yy-base)/(ry-rb),'normalization_basis':'calibrated_linear_intensity',
                   'physical_normalization_verified':True,'normalization_status':'verified_physical_intensity'})
    return values


def exhaustive_peaks(path: np.ndarray, bounds: list[int], ambiguous: np.ndarray, geometry: dict | None = None) -> list[dict]:
    observed=np.flatnonzero(np.isfinite(path))
    features=[]
    for group in np.split(observed,np.flatnonzero(np.diff(observed)>1)+1):
        if len(group)<5:
            continue
        signal=-path[group]
        proposals=[]
        for sigma in (.4,.8,1.6,3.2,6.4,12.8):
            if sigma>len(group)/5:
                continue
            smooth=gaussian_filter1d(signal,sigma)
            peaks, props=find_peaks(smooth,prominence=max(1.5,sigma*.2),distance=2)
            for j,prominence in zip(peaks,props["prominences"]):
                # Refine in raw pixels, preserving narrow apex location.
                lo=max(0,j-max(1,round(sigma)));hi=min(len(group),j+max(1,round(sigma))+1)
                raw=lo+int(np.argmax(signal[lo:hi]))
                proposals.append((int(group[raw]),float(prominence)))
        for at,prom in sorted(proposals,key=lambda p:-p[1]):
            if any(abs(at-f["original_pixel_xy"][0]+bounds[0])<=2 for f in features):
                continue
            if not (0<at<len(path)-1 and path[at]<=path[at-1]+.25 and path[at]<=path[at+1]+.25):
                continue
            py=float(path[at]);method="selected_column_center";span=(geometry or {}).get("run_bounds",{}).get(at)
            # At a narrow turn the column run contains ink from both flanks.
            # Its center is a curve sample; the supported upper tip is the apex.
            if span and prom>=3 and at>=3 and at+3<len(path) and all(np.isfinite(path[j]) for j in (at-3,at+3)):
                tip=float(span[0])
                if min(path[at-3],path[at+3])-tip>=1.5:
                    py=tip;method="selected_run_tip_with_both_flanks"
            features.append({"feature_type":"peak","original_pixel_xy":[bounds[0]+at,py],
                             "column_center_pixel_xy":[bounds[0]+at,float(path[at])],"apex_readout":method,
                             "selected_run_pixel_y_bounds":span,
                             "prominence_pixels":prom,"status":"unresolved" if ambiguous[at] else "resolved",
                             "reason":"competing trace paths" if ambiguous[at] else None})
    return sorted(features,key=lambda p:p["original_pixel_xy"][0])


def extract_plot(image_path: str, plan: dict, output_dir: str, engine: str = "candidate") -> dict:
    if engine not in ("candidate","legacy"):
        raise ValueError("engine must be candidate or legacy")
    image=core._load_image(image_path)
    words=core._ocr(image)
    folder=Path(output_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
    overlay=image.copy();draw=ImageDraw.Draw(overlay)
    panels=[];all_rows=[];identities=set()
    for pi,panel in enumerate(plan.get("panels",[])):
        pid=str(panel.get("id",f"P{pi+1}"))
        if pid in identities:
            raise ValueError("panel IDs must be unique")
        identities.add(pid)
        bounds=bounds_checked(panel["bounds"],image)
        xcal=Calibration(panel.get("x_axis"),(bounds[0],bounds[2]))
        ycal=Calibration(panel.get("y_axis"),(bounds[1],bounds[3]))
        series=[];series_ids=set()
        draw.rectangle(bounds,outline="#dd6600",width=1)
        for si,spec in enumerate(panel.get("series",[])):
            sid=str(spec.get("id",f"T{si+1}"))
            if sid in series_ids:
                raise ValueError("series IDs must be unique within a panel")
            series_ids.add(sid)
            mode=spec.get("kind","curve")
            if mode not in ("curve","markers"):
                raise ValueError("series kind must be curve or markers")
            sbounds=bounds_checked(spec.get("bounds",bounds),image)
            if not(bounds[0]<=sbounds[0]<sbounds[2]<=bounds[2] and bounds[1]<=sbounds[1]<sbounds[3]<=bounds[3]):
                raise ValueError("series bounds must lie within its panel")
            mask=_series_mask(image,sbounds,words,spec)
            if mode=="markers":
                markers=core._marker_candidates(image,sbounds,words)
                features=[{"feature_type":"marker","original_pixel_xy":m["pixel"],"status":"resolved","shape":m["shape"]}
                          for m in markers if np.any(mask[m["bounds"][1]:m["bounds"][3], m["bounds"][0]:m["bounds"][2]])]
                curves=[];normalization={"valid":False,"reason":"marker series"}
            else:
                seeds=spec.get("seeds",[])
                if any(len(s)!=2 or not sbounds[0]<=s[0]<sbounds[2] or not sbounds[1]<=s[1]<sbounds[3] for s in seeds):
                    raise ValueError("trace seeds must lie inside series bounds")
                geometry={}
                path,ambiguous=trace_path(mask,sbounds,seeds,legacy=engine=="legacy",evidence=geometry)
                features=exhaustive_peaks(path,sbounds,ambiguous,geometry) if engine=="candidate" else [
                    {"feature_type":"peak","original_pixel_xy":[sbounds[0]+x,float(path[x])],"status":"resolved","prominence_pixels":prom}
                    for x,prom in core._peak_locations(path,limit=24)]
                baseline_spec=dict(spec);support=[]
                anchors=(spec.get('baseline') or {}).get('anchors',[])
                checks=list(anchors)
                if len(anchors)>=2:
                    checks.append([(anchors[0][0]+anchors[-1][0])/2,(anchors[0][1]+anchors[-1][1])/2])
                for ax,ay in checks:
                    ax,ay=round(ax),round(ay)
                    crop=np.asarray(image.convert('L'))[max(0,ay-2):min(image.height,ay+3),max(0,ax-25):min(image.width,ax+26)]
                    if crop.size and max(np.sum(crop<160,axis=1),default=0)>=15:
                        support.append({'pixel_xy':[ax,ay],'method':'reviewed baseline anchor on independently observed horizontal stroke',
                                        'semantics':'visible pixel baseline; physical meaning conditional on verified y mapping'})
                baseline_spec['_baseline_stroke_evidence']=support
                clip_bounds=bounds if ycal.parts else [0,0,image.width,image.height]
                normalization=baseline_model(path,sbounds,baseline_spec,ambiguous,features if engine=='candidate' else None,clip_bounds)
                if panel.get('y_axis') and not ycal.parts:
                    normalization={**normalization,'valid':False,'reason':'requested y calibration could not be verified; physical normalization unresolved'}
                curves=[]
                # One sample per observed source column: no sparse grid misses narrow peaks.
                for at in range(len(path)):
                    px=sbounds[0]+at
                    py=float(path[at]) if np.isfinite(path[at]) else None
                    status="resolved" if py is not None and not ambiguous[at] else "unresolved"
                    row={"feature_type":"curve","original_pixel_xy":[px,py],"status":status,
                         "reason":None if status=="resolved" else "competing paths" if py is not None else "no observed ink"}
                    if py is not None:row.update(normalize_observation(px,py,normalization,ycal))
                    curves.append(row)
            for fi,row in enumerate(features+curves):
                px,py=row["original_pixel_xy"]
                if row['feature_type']!='curve' and mode=='curve' and py is not None:row.update(normalize_observation(px,py,normalization,ycal))
                xv,segment=xcal.convert(px);yv,ysegment=ycal.convert(py) if py is not None else (None,None)
                if engine=="legacy" and (panel.get("x_axis") or {}).get("segments"):
                    xv,segment=None,None
                row.update({"id":f"{pid}:{sid}:{fi+1}","panel_id":pid,"series_id":sid,
                            "calibrated_x":xv,"calibrated_y":yv,"x_segment_id":segment,
                            "y_segment_id":ysegment,"pixel_status":row['status'],
                            "x_calibration_status":"verified" if xv is not None else "unresolved",
                            "y_calibration_status":"verified" if yv is not None else "unresolved",
                            "x_unit":xcal.unit,"y_unit":ycal.unit,"pixel_uncertainty":1.5 if py is not None else None,
                            "calibration_status":"verified" if xv is not None else "unresolved",
                            "coordinate_uncertainty_note":"conditional on correct panel and series assignment"})
                if xcal.parts and segment is None:
                    row["status"]="unresolved";row["reason"]="axis break or outside calibrated segment"
                    row.pop("normalized_height",None)
                if py is not None:
                    row['calibrated_uncertainty']={'x_interval':[xcal.convert(px-1.5)[0],xcal.convert(px+1.5)[0]],
                                                   'y_interval':[ycal.convert(py-1.5)[0],ycal.convert(py+1.5)[0]],
                                                   'includes_tick_fit_error':False,'conditional_on_identity':True}
                if py is not None:
                    if row["feature_type"]!="curve":
                        draw.ellipse((px-3,py-3,px+3,py+3),outline="red",width=1)
                    elif row["status"]=="resolved":draw.point((round(px),round(py)),fill="#0088cc")
                all_rows.append(row)
            queries=[]
            for query in spec.get("queries",[]):
                px=query.get("pixel_x") if "pixel_x" in query else xcal.pixel_for(query["x_value"])
                pool=curves if query.get("mode","curve")=="curve" else features
                nearest=min(pool,key=lambda p:abs(p["original_pixel_xy"][0]-(px or 0)),default=None)
                if px is None or nearest is None or abs(nearest["original_pixel_xy"][0]-px)>2:
                    queries.append({"label":query["label"],"status":"unresolved","reason":"no source measurement at requested position"})
                else:
                    queries.append({**nearest,"label":query["label"]})
            normalization['id']=f'{pid}:{sid}:N1'
            normalization['panel_id']=pid;normalization['series_id']=sid
            if normalization.get('valid'):
                ref=normalization['dominant_peak_original_pixel_xy']
                normalization['reference_feature_id']=next((f['id'] for f in features if f['original_pixel_xy']==ref),None)
                normalization['mode']='highest_supported_apex' if engine=='candidate' else 'highest_dense_column_center'
                normalization['reference_x_segment_id']=xcal.convert(ref[0])[1]
            for row in features+curves:
                if row.get('normalized_height') is not None or row.get('normalized_pixel_height') is not None:
                    row['normalization_id']=normalization['id']
            series.append({"id":sid,"kind":mode,"description":spec.get("description"),"requested_target_ids":spec.get('requested_target_ids',[]),"features":features,
                           "curve":curves,"queries":queries,"normalization_evidence":normalization,
                           "bounds":sbounds,"trace_evidence":{k:v for k,v in geometry.items() if k!='run_bounds'} if mode=='curve' else None,
                           "observed_columns":sum(c["original_pixel_xy"][1] is not None for c in curves)})
        panels.append({"id":pid,"bounds":bounds,"x_calibration":xcal.evidence,"y_calibration":ycal.evidence,"series":series})
    overlay.save(folder/"overlay.png")
    fields=["id","panel_id","series_id","feature_type","status","reason","calibrated_x","calibrated_y","x_unit","y_unit",
            "x_segment_id","y_segment_id","pixel_x","pixel_y","pixel_status","pixel_uncertainty","normalized_height","normalized_pixel_height","normalization_id","normalization_basis","normalization_status","calibration_status"]
    with (folder/"measurements.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for row in all_rows:
            writer.writerow({**row,"pixel_x":row["original_pixel_xy"][0],"pixel_y":row["original_pixel_xy"][1]})
    result={"schema_version":SCHEMA,"engine":engine,"source_image":str(Path(image_path).resolve()),
            "source_sha256":hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),"panels":panels,
            "inventory_complete":bool(plan.get("inventory_complete",False)),
            "unresolved_inventory":plan.get("unresolved_inventory",[]),
            "export_manifest":{"rows":len(all_rows),"csv_sha256":hashlib.sha256((folder/"measurements.csv").read_bytes()).hexdigest()},
            "overlay_path":str(folder/"overlay.png"),"csv_path":str(folder/"measurements.csv")}
    write_json(folder/"result.json",result)
    return {"schema_version":SCHEMA,"result_path":str(folder/"result.json"),"overlay_path":result["overlay_path"],
            "panels":len(panels),"series":sum(len(p["series"]) for p in panels),
            "features":sum(len(s["features"]) for p in panels for s in p["series"]),
            "resolved_rows":sum(r["status"]=="resolved" for r in all_rows),"rows":len(all_rows)}
