"""Raster-grounded scatter inventory. LUNA selects IDs, never estimates pixels.

Candidates are image evidence, not measurements or series assignments. Small
holes and open symbols need separate detectors. Every selected ID is bound to
its native source coordinate before the frozen engine sees the plan.
"""
from __future__ import annotations

import copy
import asyncio
import json
import math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from .timing import run_local, local_work
from .luna_runtime import LunaError, _json_schema, worker_dir
from .scatter_template_tool import request_template_evidence, tile_response_evidence


def marker_candidates(source: Path) -> list[dict]:
    gray = np.asarray(Image.open(source).convert("L"))
    # Native pixels are never resized for measurement. Detection dimensions
    # scale with the source, including the tiny holes in low-resolution scans.
    scale = max(.6, max(gray.shape) / 650)
    raw = []
    contours, _ = cv2.findContours((gray < 180).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (2 <= w <= 15*scale and 2 <= h <= 15*scale and .3 < w/h < 3 and area >= max(1, 2*scale**2)):
            continue
        hull = cv2.contourArea(cv2.convexHull(contour))
        if not hull or area/hull < .65:
            continue
        m = cv2.moments(contour)
        raw.append((.86, [m['m10']/area, m['m01']/area], "bounded contour"))
    # Templates supplement holes: plus/cross symbols usually join a fitted
    # line and never have a bounded contour. These scores are not confidence.
    for radius in sorted({max(2, round(n*scale)) for n in (2, 3, 4, 5)}):
        pad = max(3, round(3*scale)); c = radius+pad; size = 2*c+1
        for shape in ("square", "triangle", "inverted triangle", "diamond", "plus", "cross", "circle"):
            t = np.full((size, size), 255, np.uint8)
            r = radius; thickness = max(1, round(scale))
            if shape == "square": cv2.rectangle(t, (c-r,c-r), (c+r,c+r), 0, thickness)
            elif shape == "circle": cv2.circle(t, (c,c), r, 0, thickness)
            elif shape in ("triangle", "inverted triangle", "diamond"):
                vertices = {"triangle": [(c,c-r),(c-r,c+r),(c+r,c+r)],
                            "inverted triangle": [(c-r,c-r),(c+r,c-r),(c,c+r)],
                            "diamond": [(c,c-r),(c-r,c),(c,c+r),(c+r,c)]}[shape]
                cv2.polylines(t, [np.array(vertices)], True, 0, thickness)
            elif shape == "plus":
                cv2.line(t, (c-r,c), (c+r,c), 0, thickness); cv2.line(t, (c,c-r), (c,c+r), 0, thickness)
            else:
                cv2.line(t, (c-r,c-r), (c+r,c+r), 0, thickness); cv2.line(t, (c-r,c+r), (c+r,c-r), 0, thickness)
            if size >= min(gray.shape): continue
            response = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            yy, xx = np.where((response == maximum_filter(response, size=max(3, round(4*scale)))) & (response > .49))
            raw.extend((float(response[y,x]), [float(x+c),float(y+c)], shape+" template") for y,x in zip(yy,xx))
    # Corners provide local search seeds where overlapping markers or a
    # border prevent contours/templates. Review may move at most four native
    # pixels (scaled with resolution), never an unrestricted image coordinate.
    corners=cv2.goodFeaturesToTrack(gray, maxCorners=0, qualityLevel=.035,
                                   minDistance=max(3,3*scale), blockSize=3)
    if corners is not None:
        raw.extend((.48,[float(x),float(y)],'ink corner') for [[x,y]] in corners)
    kept = []
    for score, pixel, kind in sorted(raw, key=lambda p:p[0], reverse=True):
        px,py=map(round,pixel); r=max(4,round(5*scale))
        patch=gray[max(0,py-r):py+r+1,max(0,px-r):px+r+1]
        yy,xx=np.where(patch < 180)
        if len(xx)<5: continue
        eigen=np.linalg.eigvalsh(np.cov(np.array([xx,yy])))
        # Reject essentially one-dimensional ink, including smooth fitted
        # lines and frame edges that correlate with one arm of a plus.
        if eigen[-1]<=0 or eigen[0]/eigen[-1]<.09: continue
        if all(math.dist(pixel, p['pixel']) > max(1.5, (4 if kind=='ink corner' else 1.8)*scale) for p in kept):
            kept.append({'pixel':[round(v,2) for v in pixel], 'detector':kind, 'score':round(score,3), 'offset_limit':max(3,round(4*scale))})
    kept.sort(key=lambda p:(p['pixel'][0],p['pixel'][1]))
    return [dict(p, id=f"M{i}") for i,p in enumerate(kept,1)]


def candidate_tiles(source: Path, candidates: list[dict], folder: Path, plan: dict) -> list[dict]:
    image = Image.open(source).convert('RGB'); w,h = image.size
    scale = max(.6,max(w,h)/650); folder.mkdir(parents=True,exist_ok=True)
    # Anchor span is a guide only; retain a margin for ticks inside the plot.
    xs = [a['pixel'] for a in (plan.get('x_axis') or {}).get('anchors',[])]
    ys = [a['pixel'] for a in (plan.get('y_axis') or {}).get('anchors',[])]
    left,right = (max(0,min(xs)-w*.035),min(w,max(xs)+w*.035)) if xs else (0,w)
    top,bottom = (max(0,min(ys)-h*.035),min(h,max(ys)+h*.035)) if ys else (0,h)
    layout=plan.get('marker_layout',{})
    if layout.get('plot_bounds'):
        left,top,right,bottom=layout['plot_bounds']
    exclusions=layout.get('exclude_regions',[])
    points = [p for p in candidates if left <= p['pixel'][0] <= right and top <= p['pixel'][1] <= bottom
              and not any(a <= p['pixel'][0] <= c and b <= p['pixel'][1] <= d for a,b,c,d in exclusions)]
    # No candidate truncation: split populated regions until each is legible.
    regions = [(left,top,right,bottom,points)]
    bounded = []
    while regions:
        x0,y0,x1,y1,pts = regions.pop(0)
        if not pts: continue
        if len(pts) > 42 or x1-x0 > 190*scale:
            # Vertical strips preserve neighbouring traces and series context.
            axis = 0 if (x1-x0) > 95*scale else 1
            mid = ((x0+x1)/2 if axis==0 else (y0+y1)/2)
            low = [p for p in pts if p['pixel'][axis] < mid]; high = [p for p in pts if p['pixel'][axis] >= mid]
            if axis==0: regions.extend([(x0,y0,mid,y1,low),(mid,y0,x1,y1,high)])
            else: regions.extend([(x0,y0,x1,mid,low),(x0,mid,x1,y1,high)])
        else: bounded.append((x0,y0,x1,y1,pts))
    records=[]
    font=ImageFont.load_default(size=15)
    for i,(x0,y0,x1,y1,pts) in enumerate(bounded):
        # Original unmarked crop plus catalogue: labels cannot hide source ink.
        bounds=[max(0,math.floor(x0-10*scale)),max(0,math.floor(y0-10*scale)),
                min(w,math.ceil(x1+10*scale)),min(h,math.ceil(y1+10*scale))]
        zoom=min(6,max(2,round(4/scale)))
        crop=image.crop(bounds);crop=crop.resize((crop.width*zoom,crop.height*zoom))
        context=folder/f'tile-{i}-source.png';crop.save(context)
        cell=116; cols=6; atlas=Image.new('RGB',(cols*cell,math.ceil(len(pts)/cols)*(cell+26)),'#eef2f6')
        draw=ImageDraw.Draw(atlas)
        for j,p in enumerate(pts):
            px,py=p['pixel'];radius=max(7,round(9*scale));box=(round(px)-radius,round(py)-radius,round(px)+radius+1,round(py)+radius+1)
            patch=image.crop(box).resize((cell-8,cell-8),Image.Resampling.NEAREST)
            dx=(j%cols)*cell+4;dy=(j//cols)*(cell+26)+24;atlas.paste(patch,(dx,dy))
            draw.text((dx,dy-22),p['id'],fill='#063d75',font=font)
            # Tick marks outside the center keep the symbol fully visible.
            cx=dx+(px-box[0])*(cell-8)/(box[2]-box[0]);cy=dy+(py-box[1])*(cell-8)/(box[3]-box[1])
            for line in ((cx-8,cy,cx-4,cy),(cx+4,cy,cx+8,cy),(cx,cy-8,cx,cy-4),(cx,cy+4,cx,cy+8)):
                draw.line(line,fill='#e00085',width=1)
        catalogue=folder/f'tile-{i}-candidates.png';atlas.save(catalogue)
        records.append({'bounds':bounds,'ownership_bounds':[x0,y0,x1,y1], 'source_path':str(context),
                        'catalogue_path':str(catalogue),'candidates':pts})
    return records


def bind_inventory(plan: dict, candidates: list[dict], records: list[dict], series: list[str]) -> dict:
    """Reject foreign IDs; match old labels without ever reusing old pixels."""
    updated=copy.deepcopy(plan);by_id={p['id']:p for p in candidates};selected=[];seen=set()
    for rec in records:
        for item in rec['items']:
            key=(item['series_id'],item['candidate_id'])
            if key in seen: raise ValueError('Duplicate marker/series assignment')
            seen.add(key)
            if item['candidate_id'] not in by_id or item['series_id'] not in {f'S{i}' for i in range(len(series))}:
                raise ValueError('Unknown grounded candidate or series')
            p=by_id[item['candidate_id']];name=series[int(item['series_id'][1:])]
            if p.get('template_series') is not None and p['template_series'] != item['series_id']:
                raise ValueError('Template candidate cannot be assigned to a different series; use a supported alternative')
            offset=item.get('offset',[0,0]);limit=p.get('offset_limit',0)
            if len(offset)!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit for v in offset):
                raise ValueError('Marker center adjustment exceeds its local patch')
            pixel=[round(a+b,2) for a,b in zip(p['pixel'],offset)]
            selected.append({'label':f"{name} · {p['id']}",'series':name,'mode':'exact','pixel':pixel,
                'candidate_id':None,'status':'selected','reason':None,'x_value':None,'review_id':None,
                'review_note':f"Source candidate {p['id']}: {item['reason']}",
                'coordinate_review':'uncertain' if item['verdict']=='uncertain' or p.get('center_stable') is False else 'corrected',
                'coordinate_review_note':f"Source candidate {p['id']} ({p['detector']}): {item['reason']}",
                'grounding':dict(p, offset=offset, version='template-tool-v1' if p.get('template_series') else 'local-candidates-v1')})
    counts=Counter(tuple(p['pixel']) for p in selected)
    for p in selected:
        if counts[tuple(p['pixel'])]>1:
            p['coordinate_review']='uncertain';p['coordinate_review_note']+=' Shared raster center; overlapping series cannot be separated.'
        if p['grounding'].get('center_stable') is False:
            p['coordinate_review_note']+=' Template response has an unstable center.'
    for i,p in enumerate(selected):
        for q in selected[i+1:]:
            if p['series']==q['series'] and math.dist(p['pixel'],q['pixel'])<=1.5 and (
                    p['grounding'].get('template_series') or q['grounding'].get('template_series')):
                for item in (p,q):
                    item['coordinate_review']='uncertain'
                    item['coordinate_review_note']+=' Possible duplicate from template and local candidates; review source.'
    # Assign prior labels within each series; missing detections stay explicit
    # unresolved rows, with no unsupported ring or invented measured number.
    missing=[]
    for name in series:
        old=[p for p in plan['selections'] if p.get('series','Series')==name]
        new=[p for p in selected if p['series']==name]
        matched=set()
        if old and new:
            costs=np.array([[abs((o.get('pixel') or [1e6,1e6])[0]-n['pixel'][0]) + .15*abs((o.get('pixel') or [1e6,1e6])[1]-n['pixel'][1]) for n in new] for o in old])
            # Dummy unmatched columns avoid forcing a missing seed to steal
            # the next feature and shifting every subsequent label.
            oi,ni=linear_sum_assignment(np.hstack([costs,np.full((len(old),len(old)),35.)]))
            for a,b in zip(oi,ni):
                if b < len(new) and costs[a,b] < 35:
                    new[b]['label']=old[a]['label'];matched.add(a)
        for i,o in enumerate(old):
            if i not in matched:
                p=copy.deepcopy(o);p.update(pixel=None,candidate_id=None,status='unresolved',
                    reason='No independently identified source marker; previous approximate location was discarded.',
                    coordinate_review='uncertain',coordinate_review_note='No locally grounded marker assignment. Review this feature in the source.')
                missing.append(p)
    updated['selections']=sorted(selected,key=lambda p:(series.index(p['series']),p['pixel'][0],p['pixel'][1]))+missing
    updated.setdefault('notes',[]).append('Marker inventory rebuilt from native raster candidate IDs in enlarged local crops. Missing/overlapping assignments remain explicit; candidate detection alone does not prove series identity.')
    return updated


async def ground_scatter(runner, plan: dict, source: Path, folder: Path, target: str) -> dict:
    candidates=await run_local(marker_candidates,source)
    series=list(dict.fromkeys(p.get('series','Series') for p in plan['selections']))
    layout_schema=_json_schema({'plot_bounds':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},
        'exclude_regions':{'type':'array','items':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4}},
        'reason':{'type':'string'},'series_guidance':{'type':'string'}})
    width,height=Image.open(source).size
    layout_prompt=f"Identify the Cartesian DATA rectangle and any legend/text-only rectangles inside it. This is layout segmentation, not data measurement. Source image is {width} by {height} pixels, x right y down. Return native-pixel [left,top,right,bottom] bounds enclosing the axes and all data markers, including boundary points. Exclude only clearly text/legend regions with no plotted data. Never exclude a crowded data region. Target: {target}. Series: {json.dumps({f'S{j}':s for j,s in enumerate(series)})}. Describe each series symbol and its broad location in the plot in series_guidance, using visible evidence. This is an identity guide only, not a source of numbers."
    local_plan=copy.deepcopy(plan)
    for attempt in range(2):
        try:
            layout=await runner.call(prompt=layout_prompt,schema=layout_schema,images=[str(source)],
                                    workdir=worker_dir(str(folder),'marker-layout'+('-repair' if attempt else '')))
            for box in [layout['plot_bounds'],*layout['exclude_regions']]:
                if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                    raise ValueError('Invalid layout rectangle')
            local_plan['marker_layout']=layout
            (folder/'marker-layout.json').write_text(json.dumps(layout,indent=2))
            break
        except (LunaError,ValueError,KeyError,TypeError) as exc:
            layout_prompt+=' Return valid native-image rectangles. Previous response: '+str(exc)
    template_result=None
    if local_plan.get('marker_layout'):
        template_result=await request_template_evidence(runner,source,local_plan['marker_layout'],series,
                                                        folder/'scatter-template-tool',target)
    if template_result:
        candidates=candidates+template_result['candidates']
    tiles=await run_local(candidate_tiles,source,candidates,folder/'marker-evidence',local_plan)
    series=list(dict.fromkeys(p.get('series','Series') for p in plan['selections']))
    legends=[]
    for i,box in enumerate(local_plan.get('marker_layout',{}).get('exclude_regions',[])):
        # Enlarging the actual legend is critical: tiny V/^ symbols are easy
        # to swap when only the full-image context is attached.
        crop=Image.open(source).convert('RGB').crop(tuple(round(v) for v in box))
        factor=min(8,max(2,round(650/max(crop.size))))
        path=folder/f'marker-legend-{i}.png';crop.resize((crop.width*factor,crop.height*factor)).save(path)
        legends.append(str(path))
    records=[]
    manifest={'version':'local-candidates-v1','candidates':candidates,'tiles':tiles,'series':series}
    if template_result:
        manifest['template_tool']={'scan_id':template_result['scan_id'],'source_sha256':template_result['source_sha256'],
                                   'arguments':template_result['arguments'],'warnings':template_result['warnings']}
    (folder/'marker-grounding.json').write_text(json.dumps(manifest,indent=2))
    for i,tile in enumerate(tiles):
        await runner.emit({'type':'stage','stage':'coordinate_review','message':f'Identifying source markers in enlarged region {i+1} of {len(tiles)}'})
        ids=[p['id'] for p in tile['candidates']]
        template_images=[]
        if template_result:
            template_images=[await run_local(tile_response_evidence,template_result,tile['bounds'],
                                                      folder/'marker-evidence'/f'tile-{i}-responses.png')]
        schema=_json_schema({'items':{'type':'array','items':_json_schema({
            'candidate_id':{'type':'string','enum':ids},'series_id':{'type':'string','enum':[f'S{j}' for j in range(len(series))]},
            'verdict':{'type':'string','enum':['confirmed','uncertain']},'offset':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},'reason':{'type':'string'}})},'notes':{'type':'array','items':{'type':'string'}}})
        prompt=f'''Identify ALL experimental markers represented by the candidate catalogue in this local region. The first image is the full unannotated source for legend and series context, the second an enlarged unmarked local crop, the third a catalogue of native raster candidates. Each catalogue patch is centered at the pink crosshair. Choose IDs; NEVER invent absolute pixel coordinates. Return offset [dx,dy] from the candidate crosshair to the actual marker center, in NATIVE source pixels (x right, y down). Use [0,0] for a centered candidate. Each offset component must stay within the supplied offset_limit. A catalogue patch covers about 18 native pixels across; its magnification is display only. A small bounded offset can center a corner seed or a partial contour. If the marker center is outside the limit, omit that candidate and mention it in notes. Select only candidates whose CENTER is the marker's center, not a corner or adjacent fitted line. Do not select legend symbols, text, ticks, axes, isolated fitted-curve segments, or multiple candidates for the same physical marker. Candidate detector shape names are weak hints, not truth. Match series using visible symbol shapes and the full-source legend. Squares and upright triangles, diamonds and downward triangles are DISTINCT even where their curves almost overlap. Plus and x are distinct. A continuous fit is not an experimental marker. Include visible points missed in any initial inventory. Mark uncertain for inseparable overlapping markers and weak series identity. If no candidate is centered on a particular visible marker, mention it in notes; do not substitute a nearby curve or guess. This request is to analyse all visible data points, with warnings for ambiguity.
Target: {target}
Series identities (must match): {json.dumps({f'S{j}':s for j,s in enumerate(series)})}
Full-chart identity guide (verify against the actual symbol): {local_plan.get('marker_layout',{}).get('series_guidance','')}
Additional images after the catalogue are enlarged source legends. Compare each marker with the legend; never reassign an upside-down triangle to the upright-triangle series. If the symbol orientation or identity is not visibly supported, mark uncertain.
If a final template-response image is attached, it covers the SAME native region below. Each panel is labeled by series ID. White rings highlight correlation peaks, not certainty. K IDs are returned by scan_scatter_templates with offset_limit=0: accept their computed centers only when the source verifies them, otherwise omit them. Their template_series must match the selected series_id. A K ID and an M ID may describe the same physical marker; choose only one. Template agreement does not prove identity; missing response does not prove absence. Continue examining generic candidates for markers missed by the tool. Unstable template centers remain uncertain even after semantic review.
Native region bounds: {tile['bounds']}
Candidate IDs and source coordinates: {json.dumps(tile['candidates'])}
'''
        rec=None
        for attempt in range(2):
            try:
                rec=await runner.call(prompt=prompt,schema=schema,
                    images=[str(source),tile['source_path'],tile['catalogue_path'],*legends,*template_images],
                    workdir=worker_dir(str(folder),f'marker-region-{i}'+('-repair' if attempt else '')))
                pairs=[(p['candidate_id'],p['series_id']) for p in rec['items']]
                if len(pairs)!=len(set(pairs)) or any(p['candidate_id'] not in ids or p['series_id'] not in {f'S{j}' for j in range(len(series))} for p in rec['items']):
                    raise ValueError('Unknown ID or duplicate assignment')
                bound=local_work(bind_inventory,{'selections':[]},tile['candidates'],[rec],series)
                if any(not (0 <= p['pixel'][0] < width and 0 <= p['pixel'][1] < height) for p in bound['selections']):
                    raise ValueError('Adjusted marker lies outside the source')
                break
            except (LunaError,ValueError,KeyError,TypeError) as exc:
                if attempt: rec={'items':[],'notes':[f'Local marker review failed: {exc}']}
                else: prompt+='\nReturn the exact schema with unique candidate/series pairs and only allowed IDs. The last response failed validation: '+str(exc)
        records.append(rec)
        (folder/'marker-inventory.json').write_text(json.dumps(records,indent=2))
    updated=local_work(bind_inventory,plan,candidates,records,series)
    if template_result:
        updated.setdefault('notes',[]).append('LUNA requested source-template correlation evidence. Its fixed-center proposals and local heatmaps supplemented the existing detector; response scores do not certify marker identity or completeness.')
    updated['notes'].extend(note for rec in records for note in rec.get('notes',[]))
    return updated
