"""Beta local evidence: line nuisance fits, centered proposals and duplicate groups.

All scores are image-fit diagnostics, never calibrated probabilities. Rejected
search seeds are retained for coverage review; no trend or sampling grid is used.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


@lru_cache(maxsize=12)
def _line_bank(radius: int, thickness: int):
    size = radius * 2 + 1
    masks = []
    for angle in range(0, 180, 15):
        a = math.radians(angle)
        direction = np.array([math.cos(a), math.sin(a)])
        normal = np.array([-direction[1], direction[0]])
        for offset in (-3, 0, 3):
            center = np.array([radius, radius]) + normal * offset
            ends = [tuple(np.rint(center + sign * size * direction).astype(int)) for sign in (-1, 1)]
            for width in range(1, thickness+3):
                mask = np.zeros((size, size), np.uint8)
                cv2.line(mask, *ends, 1, width)
                masks.append(mask.astype(bool))
    return np.array(masks)


def line_evidence(gray: np.ndarray, pixel: list[float], radius: int) -> dict:
    x, y = map(round, pixel)
    padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=255)
    patch = padded[y:y+2*radius+1, x:x+2*radius+1] < 180
    masks = _line_bank(radius, max(1, round(max(gray.shape)/900)))
    yy0,xx0=np.mgrid[:2*radius+1,:2*radius+1]
    center_disk=(xx0-radius)**2+(yy0-radius)**2 <= max(4,radius*.7)**2
    # Nearby parallel curves outside the proposed center must not make a
    # line-only center look like a marker. Keep the full patch for visual review.
    patch = patch & center_disk
    masks = masks & center_disk
    overlap = (masks & patch).sum(axis=(1, 2))
    best = int(np.argmax(overlap - .25 * (masks & ~patch).sum(axis=(1, 2))))
    remaining = patch & ~masks[best]
    total = int(patch.sum())
    yy, xx = np.where(remaining)
    quadrants = len(set(zip((xx >= radius).tolist(), (yy >= radius).tolist())))
    return {"ink_pixels": total, "line_residual_pixels": int(remaining.sum()),
            "line_residual_fraction": round(float(remaining.sum()/max(total, 1)), 4),
            "residual_quadrants": quadrants,
            "line_only": bool(total > 5 and remaining.sum() <= max(3, total*.10))}


def prepare_candidates(source: Path, candidates: list[dict], layout: dict) -> tuple[list[dict], list[dict]]:
    """Group near-identical seeds; retain weak line/corner seeds in a reserve."""
    gray = np.asarray(Image.open(source).convert('L'))
    scale = max(.6, max(gray.shape)/1432)
    radius = max(5, round(10*scale))
    a,b,c,d = layout['plot_bounds']
    inside = [dict(p) for p in candidates if a <= p['pixel'][0] <= c and b <= p['pixel'][1] <= d
              and not any(x <= p['pixel'][0] <= z and y <= p['pixel'][1] <= w
                          for x,y,z,w in layout.get('exclude_regions', []))]
    for p in inside:
        p['local_fit'] = line_evidence(gray, p['pixel'], radius)
    # Source-derived centers outrank corners; retain competing series separately.
    inside.sort(key=lambda p: (p.get('template_series') is None, p['detector']=='ink corner', -p.get('score',0)))
    kept, reserve = [], []
    for p in inside:
        near = next((q for q in kept if math.dist(p['pixel'],q['pixel']) <= max(1.5, 2.5*scale)
                     and (not p.get('template_series') or not q.get('template_series')
                          or p['template_series']==q['template_series'])), None)
        if near:
            near.setdefault('aliases', []).append(p['id'])
            reserve.append({**p, 'reserve_reason':'near_duplicate', 'representative_id':near['id']})
        elif p['detector']=='ink corner' and (p['local_fit']['line_only'] or any(
                math.dist(p['pixel'],q['pixel']) <= max(3, 6*scale) for q in kept)):
            reserve.append({**p, 'reserve_reason':'line_or_nearby_corner'})
        else:
            kept.append(p)
    return sorted(kept,key=lambda p:(p['pixel'][0],p['pixel'][1])), reserve


def exemplar_proposals(source: Path, seeds: list[dict], layout: dict, limit_per_series: int = 3) -> list[dict]:
    """Use stable isolated source-template proposals as appearance exemplars.

Series remain hypotheses until visual review. Search on untouched source ink;
compare marker support after fitting nuisance line ink, including open symbols.
"""
    gray = np.asarray(Image.open(source).convert('L'))
    ink = 1-gray.astype(np.float32)/255
    by_series = {}
    for p in seeds:
        if p.get('template_series') and p.get('center_stable') and not p.get('alternatives'):
            by_series.setdefault(p['template_series'], []).append(p)
    proposals = []
    a,b,c,d = layout['plot_bounds']
    for sid, choices in by_series.items():
        selected = sorted(choices,key=lambda p:p.get('score',0),reverse=True)[:limit_per_series]
        for seed in selected:
            x,y = map(round,seed['pixel']);radius=max(3,round(seed.get('diameter',12)/2))
            if x-radius<0 or y-radius<0 or x+radius>=gray.shape[1] or y+radius>=gray.shape[0]:continue
            kernel=ink[y-radius:y+radius+1,x-radius:x+radius+1]
            # The untouched patch includes its local fitted line; support is
            # checked against a line-only alternative at every proposed center.
            response=cv2.matchTemplate(ink,kernel,cv2.TM_CCOEFF_NORMED)
            peaks=(response==cv2.dilate(response,np.ones((max(3,radius),)*2,np.uint8))) & (response>=.70)
            yy,xx=np.where(peaks)
            for u,v in zip(xx,yy):
                pixel=[float(u+radius),float(v+radius)]
                if not(a<=pixel[0]<=c and b<=pixel[1]<=d):continue
                if any(l<=pixel[0]<=r and t<=pixel[1]<=bottom for l,t,r,bottom in layout.get('exclude_regions',[])):continue
                # Seed patches cannot validate themselves.
                if math.dist(pixel,seed['pixel'])<=radius:continue
                evidence=line_evidence(gray,pixel,radius)
                if evidence['line_only'] or evidence['line_residual_pixels']<5 or evidence['residual_quadrants']<2:continue
                if any(q['template_series']==sid and math.dist(q['pixel'],pixel)<max(2,radius*.6) for q in proposals):continue
                proposals.append({'id':f'E{len(proposals)+1}','pixel':pixel,'detector':'plotted exemplar with line nuisance',
                    'score':round(float(response[v,u]),4),'template_series':sid,'offset_limit':0,
                    'center_stable':False,'exemplar_id':seed['id'],'local_fit':evidence,
                    'note':'Unverified appearance exemplar; visual identity and center review required.'})
    return proposals


def fit_marker_plus_line(gray: np.ndarray, pixel: list[float], shape: str) -> dict:
    """Fit a small symbol jointly with a locally observed straight-line nuisance.

The local center search is bounded to two pixels. This function reports evidence
and an alternative center; callers do not replace fixed template coordinates.
"""
    scale=max(.6,max(gray.shape)/1432);radius=max(8,round(14*scale));size=2*radius+1
    x,y=map(round,pixel)
    padded=cv2.copyMakeBorder(gray,radius,radius,radius,radius,cv2.BORDER_CONSTANT,value=255)
    ink=padded[y:y+size,x:x+size]<180
    lines=_line_bank(radius,max(1,round(max(gray.shape)/900)))
    errors=(lines & ~ink).sum(axis=(1,2)) + 1.5*(ink & ~lines).sum(axis=(1,2))
    line=lines[int(np.argmin(errors))];baseline=float(np.min(errors))
    yy,xx=np.mgrid[:size,:size];central=(xx-radius)**2+(yy-radius)**2 <= (radius-2)**2
    trials=[]
    radii=sorted({max(2,round(v*scale)) for v in (3,4,5,6,7,8)})
    for r in radii:
        for stroke in (1,max(2,round(2*scale))):
            for dx in (-2,-1,0,1,2):
                for dy in (-2,-1,0,1,2):
                    cx,cy=radius+dx,radius+dy;marker=np.zeros_like(ink,dtype=np.uint8)
                    if shape=='square':cv2.rectangle(marker,(cx-r,cy-r),(cx+r,cy+r),1,stroke)
                    elif shape=='diamond':cv2.polylines(marker,[np.array([(cx,cy-r),(cx+r,cy),(cx,cy+r),(cx-r,cy)])],True,1,stroke)
                    elif shape in {'triangle','inverted triangle'}:
                        vertices=[(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)] if shape=='triangle' else [(cx-r,cy-r),(cx+r,cy-r),(cx,cy+r)]
                        cv2.polylines(marker,[np.array(vertices)],True,1,stroke)
                    elif shape=='plus':
                        cv2.line(marker,(cx-r,cy),(cx+r,cy),1,stroke);cv2.line(marker,(cx,cy-r),(cx,cy+r),1,stroke)
                    elif shape=='cross':
                        cv2.line(marker,(cx-r,cy-r),(cx+r,cy+r),1,stroke);cv2.line(marker,(cx-r,cy+r),(cx+r,cy-r),1,stroke)
                    else:return {'supported':False,'reason':'Unsupported shape'}
                    marker=marker.astype(bool);union=line|marker
                    error=float((union & ~ink).sum()+1.5*(ink & ~union).sum())
                    novel=marker & ~line
                    support=float((novel & ink).sum()/max(1,novel.sum()))
                    # New marker arms must explain ink beyond the line itself.
                    novel_ink=int((novel & ink & central).sum())
                    trials.append((error,-support,-novel_ink,dx,dy,r))
    trials.sort();best=trials[0];error,negative_support,negative_ink,dx,dy,r=best
    alternatives=[t for t in trials if t[0]<=error+max(1,.02*baseline)]
    center_spread=max(math.hypot(t[3]-dx,t[4]-dy) for t in alternatives)
    gain=(baseline-error)/max(1,baseline)
    return {'supported':bool(gain>=.12 and -negative_support>=.6 and -negative_ink>=5),
            'pixel':[float(x+dx),float(y+dy)],'line_error':round(baseline,3),'joint_error':round(error,3),
            'relative_gain':round(gain,4),'off_line_support':round(-negative_support,4),
            'center_spread':round(center_spread,3),'radius':r,'center_stable':center_spread<=1}


def add_joint_fit_evidence(source: Path, candidates: list[dict]) -> list[dict]:
    gray=np.asarray(Image.open(source).convert('L'));result=[]
    for candidate in candidates:
        p=dict(candidate);shape=p['detector'].removesuffix(' template')
        if shape in {'square','diamond','triangle','inverted triangle','plus','cross'}:
            fit=fit_marker_plus_line(gray,p['pixel'],shape);p['joint_fit']=fit
            # Preserve the original proposal, keep this an explicit bounded fit.
            if fit['supported'] and fit['center_stable']:
                p['original_pixel']=p['pixel'];p['pixel']=fit['pixel'];p['offset_limit']=0
                p['center_stable']=True;p['detector']+=' + local joint fit'
        result.append(p)
    return result


def marker_shape(description: str) -> str | None:
    text=description.lower()
    if 'plus' in text or text.strip()=='+':return 'plus'
    if 'cross' in text or text.strip()=='x':return 'cross'
    if 'down' in text or 'inverted' in text:return 'inverted triangle'
    if 'triangle' in text:return 'triangle'
    if 'square' in text:return 'square'
    if 'diamond' in text:return 'diamond'
    return None


def validate_marker_selections(source: Path, candidates: list[dict], records: list[dict], series: list[dict], layout: dict):
    """Require local source support before model selections become measurements.

A model can assign identity but cannot turn a line/corner seed into measured data
without marker support. Rejections remain in an explicit coverage ledger.
"""
    gray=np.asarray(Image.open(source).convert('L'));scale=max(.6,max(gray.shape)/1432)
    by_id={p['id']:p for p in candidates};accepted=[];rejected=[]
    left,top,right,bottom=layout['plot_bounds']
    for record in records:
        items=[]
        for item in record['items']:
            p=by_id[item['candidate_id']];pixel=[a+b for a,b in zip(p['pixel'],item['offset'])]
            evidence=line_evidence(gray,pixel,max(5,round(10*scale)))
            shape=marker_shape(series[int(item['series_id'][1:])]['shape'])
            fit=fit_marker_plus_line(gray,pixel,shape) if shape else {'supported':False}
            reason=None
            if evidence['line_only']:reason='Selected center is explained by line ink without independent marker strokes.'
            elif p['detector']=='ink corner' and not fit.get('supported'):
                reason='Corner seed lacks a supported marker-plus-line fit for the selected shape.'
            elif (abs(pixel[0]-left)<10*scale or abs(pixel[1]-bottom)<10*scale) and not p.get('template_series') and (not fit.get('supported') or fit.get('off_line_support',0)<.9):
                reason='Frame/tick-adjacent proposal lacks an independent marker fit.'
            if reason:
                rejected.append({'candidate_id':p['id'],'series_id':item['series_id'],'pixel':pixel,'reason':reason,'local_fit':evidence,'joint_fit':fit})
            else:
                value=dict(item)
                if p['detector']=='ink corner' and fit.get('supported'):
                    offset=[a-b for a,b in zip(fit['pixel'],p['pixel'])]
                    if fit.get('center_stable') and all(abs(v)<=p['offset_limit'] for v in offset):value['offset']=offset
                    else:value['verdict']='uncertain'
                items.append(value)
        accepted.append({'items':items,'notes':record.get('notes',[])})
    return accepted,rejected
