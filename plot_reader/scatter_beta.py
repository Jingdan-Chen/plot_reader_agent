"""Bounded beta scatter workflow: compact setup, local evidence, concurrent review."""
from __future__ import annotations
import asyncio
import copy
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pytesseract
from .luna_runtime import AXIS, SELECTED_SCHEMA, LunaError, _json_schema, worker_dir
from .timing import run_local
from .browser_grounding import marker_candidates, bind_inventory
from .browser_evidence import selection_overlay
from .scatter_template_tool import request_template_evidence
from .marker_fitting import prepare_candidates, exemplar_proposals, add_joint_fit_evidence, validate_marker_selections
from .jev_vision import classify_candidates

BOX={'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4}
LAYOUT=_json_schema({'plot_bounds':BOX,'exclude_regions':{'type':'array','items':BOX},
                     'reason':{'type':'string'},'series_guidance':{'type':'string'}})


def setup_schema(allow_questions=False):
    axis=copy.deepcopy(AXIS)
    axis['anyOf'][0]['properties']['anchors']['maxItems']=2
    questions=copy.deepcopy(SELECTED_SCHEMA['properties']['clarifications'])
    if not allow_questions:questions['maxItems']=0
    return _json_schema({'x_axis':axis,'y_axis':axis,'marker_layout':LAYOUT,
        'marker_series':{'type':'array','maxItems':20,'items':_json_schema({
            'name':{'type':'string'},'shape':{'type':'string'},'guide':{'type':'string'}})},
        'clarifications':questions,'notes':{'type':'array','items':{'type':'string'}}})


def validate_setup(plan, size):
    layout=plan['marker_layout'];width,height=size
    for box in [layout['plot_bounds'],*layout['exclude_regions']]:
        if len(box)!=4 or not all(type(v) in (float,int) and math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
            raise ValueError('Invalid native-image rectangle')
    names=[p['name'] for p in plan['marker_series']]
    if len(set(names))!=len(names) or any(not s.strip() for s in names):raise ValueError('Series names must be unique and nonempty')


def printed_tick_checks(source, plan):
    """Local strip OCR can challenge an invented check, never invent calibration.

    Only challenge a check bracketed by at least three readable, monotonic label
    observations. Missing OCR at an endpoint does not invalidate a visual anchor.
    """
    image=Image.open(source).convert('RGB');scale=max(.6,max(image.size)/1432)
    left,top,right,bottom=plan['marker_layout']['plot_bounds']
    boxes={'x':[max(0,left-50*scale),min(image.height,bottom+15*scale),image.width,min(image.height,bottom+95*scale)],
           'y':[max(0,left-120*scale),max(0,top-20*scale),max(0,left-20*scale),min(image.height,bottom+30*scale)]}
    evidence={};errors=[]
    for name,box in boxes.items():
        box=list(map(round,box));axis=plan.get(name+'_axis')
        if not axis or axis.get('scale')!='linear' or box[2]<=box[0] or box[3]<=box[1]:continue
        crop=image.crop(box);crop=crop.resize((crop.width*3,crop.height*3))
        data=pytesseract.image_to_data(crop,config='--psm 6',output_type=pytesseract.Output.DICT)
        labels=[]
        for i,word in enumerate(data['text']):
            if float(data['conf'][i])<70 or not re.fullmatch(r'[+−-]?\d+(?:[.,]\d+)?',word):continue
            pixel=(box[0]+(data['left'][i]+data['width'][i]/2)/3 if name=='x'
                   else box[1]+(data['top'][i]+data['height'][i]/2)/3)
            labels.append({'value':float(word.replace('−','-').replace(',','.')),'label_pixel':pixel})
        labels.sort(key=lambda p:p['label_pixel']);evidence[name]=labels
        values=[p['value'] for p in labels]
        monotonic=(all(a<b for a,b in zip(values,values[1:])) or all(a>b for a,b in zip(values,values[1:])))
        if len(labels)<3 or not monotonic:continue
        for check in axis.get('checks',[]):
            if min(values)<check['value']<max(values) and not any(math.isclose(check['value'],v,abs_tol=1e-9) for v in values):
                errors.append(f"{name}-axis check {check['value']} has no matching printed label in local strip OCR. Readable labels: {labels}. Use a visible labeled check, not an inferred minor tick.")
    return evidence,errors


async def plan_scatter(runner, source, folder, request, inspection, followup=''):
    size=Image.open(source).size
    prompt=f'''Plan extraction of all requested experimental scatter markers. Return axes, series identities and layout only; do not enumerate point coordinates. Native source size: {size}; x right, y down.
For each calibrated axis choose exactly two widely separated ticks with readable PRINTED numerical labels and at least one independent labeled check. Never infer an unlabeled minor tick value from spacing; it cannot be an anchor or check. Null axes are allowed when unsupported. Read symbol orientation and legend labels from source. Distinguish markers from fitted lines. Name each series uniquely; describe its shape and visible identity cues. No expected trends or sampling grids may establish points. Identify the complete data rectangle including boundary markers. Exclude only pure legend/text rectangles containing no data. Report ambiguity in notes.
Request: {request.get('targets')}. Channel: {request.get('channel_description')}. Figure: {request.get('target',{}).get('explanation','')}
{'Agent questions are enabled; ask only if a source interpretation prevents extraction.' if request.get('allow_agent_questions') else 'Agent questions are disabled. Do not ask the user questions; return clarifications=[] and preserve uncertainty.'}
Local OCR ticks (fallible): {json.dumps(inspection.get('ticks',{}))}
{followup}'''
    for attempt in range(2):
        try:
            result=await runner.call(prompt=prompt,schema=setup_schema(request.get('allow_agent_questions',False)),images=[str(source)],
                workdir=worker_dir(str(folder),'scatter-setup'+('-repair' if attempt else '')))
            validate_setup(result,size)
            ticks,errors=await run_local(printed_tick_checks,source,result)
            (folder/'scatter-axis-evidence.json').write_text(json.dumps(ticks,indent=2))
            if errors:raise ValueError(' '.join(errors))
            result.update(selections=[],reference_label=None,baseline_pixel=None,scatter_beta=True)
            (folder/'scatter-setup.json').write_text(json.dumps(result,indent=2))
            return result
        except (LunaError,ValueError,KeyError,TypeError) as exc:
            if attempt:raise LunaError('Scatter setup remained invalid after one repair') from exc
            prompt+='\nCorrect this validation error: '+str(exc)


def review_batches(source, primary, reserve, folder):
    """Every native candidate has one owner; sparse regions share a call."""
    image=Image.open(source).convert('RGB');folder.mkdir(parents=True,exist_ok=True)
    records=[];scale=max(.6,max(image.size)/1432);radius=max(8,round(18*scale))
    for kind,points,size in [('primary',primary,40),('reserve',reserve,64)]:
        points=sorted(points,key=lambda p:(p['pixel'][0],p['pixel'][1]))
        for start in range(0,len(points),size):
            batch=points[start:start+size];index=len(records);cell=112;cols=8
            sheet=Image.new('RGB',(cols*cell,math.ceil(len(batch)/cols)*(cell+24)),'#eef2f6');draw=ImageDraw.Draw(sheet)
            font=ImageFont.load_default(size=16)
            for i,p in enumerate(batch):
                x,y=p['pixel'];bounds=(round(x)-radius,round(y)-radius,round(x)+radius+1,round(y)+radius+1)
                patch=image.crop(bounds).resize((cell-8,cell-8),Image.Resampling.NEAREST)
                u=(i%cols)*cell+4;v=(i//cols)*(cell+24)+24
                sheet.paste(patch,(u,v));draw.text((u,v-22),p['id'],fill='#063d75',font=font)
                # Crosshair outside center; source patch remains readable.
                cx=u+(x-bounds[0])*(cell-8)/(2*radius+1);cy=v+(y-bounds[1])*(cell-8)/(2*radius+1)
                for ends in [(cx-9,cy,cx-5,cy),(cx+5,cy,cx+9,cy),(cx,cy-9,cx,cy-5),(cx,cy+5,cx,cy+9)]:draw.line(ends,fill='#d00080')
            atlas=folder/f'batch-{index}-candidates.png';sheet.save(atlas)
            bounds=[max(0,math.floor(min(p['pixel'][0] for p in batch)-radius*2)),max(0,math.floor(min(p['pixel'][1] for p in batch)-radius*2)),
                    min(image.width,math.ceil(max(p['pixel'][0] for p in batch)+radius*2)),min(image.height,math.ceil(max(p['pixel'][1] for p in batch)+radius*2))]
            context=image.crop(bounds);context.thumbnail((1100,1100));path=folder/f'batch-{index}-source.png';context.save(path)
            records.append({'kind':kind,'bounds':bounds,'candidates':batch,'catalogue_path':str(atlas),'source_path':str(path)})
    return records


def reconcile(records,candidates,series):
    """Suppress repeated same-marker selections across adjacent proposal groups."""
    by_id={p['id']:p for p in candidates};kept=[];suppressed=[]
    for record in records:
        for item in record['items']:
            p=by_id[item['candidate_id']];pixel=[a+b for a,b in zip(p['pixel'],item['offset'])]
            duplicate=next((q for q in kept if q['series_id']==item['series_id'] and math.dist(q['_pixel'],pixel)<=2),None)
            if duplicate:
                suppressed.append({'candidate_id':item['candidate_id'],'retained_id':duplicate['candidate_id'],'reason':'Same-series centers within two native pixels'})
                duplicate['verdict']='uncertain';duplicate['reason']+=' Nearby duplicate proposal merged; inspect touching markers.'
            else:kept.append({**item,'_pixel':pixel})
    return [{'items':[{k:v for k,v in x.items() if k!='_pixel'} for x in kept],'notes':[]}],suppressed


async def ground_scatter_beta(runner, plan, source, folder, request, *, concurrency=4):
    series=[s['name'] for s in plan['marker_series']];layout=plan['marker_layout']
    if not series:return plan
    raw=await run_local(marker_candidates,source)
    raw=await run_local(add_joint_fit_evidence,source,raw)
    template=await request_template_evidence(runner,source,layout,series,folder/'scatter-template-tool',request.get('targets',''))
    if template:
        raw+=template['candidates']
        raw+=await run_local(exemplar_proposals,source,template['candidates'],layout)
    primary,reserve=await run_local(prepare_candidates,source,raw,layout)
    candidates=primary+reserve
    batches=await run_local(review_batches,source,primary,reserve,folder/'marker-evidence')
    manifest={'version':'beta-v1','concurrency':concurrency,'series':series,'primary_count':len(primary),'reserve_count':len(reserve),
              'candidates':candidates,'batches':batches,'jev_mode':request.get('jev_mode','off')}
    (folder/'marker-grounding.json').write_text(json.dumps(manifest,indent=2))
    legends=[]
    image=Image.open(source).convert('RGB')
    for i,box in enumerate(layout['exclude_regions']):
        im=image.crop(tuple(round(v) for v in box));factor=min(4,max(1,round(900/max(im.size))));im=im.resize((im.width*factor,im.height*factor));path=folder/f'marker-legend-{i}.png';im.save(path);legends.append(str(path))
    mode=request.get('jev_mode','off');jev_task=None;jev={'answers':{}}
    if mode in {'shadow','assist'}:
        jev_task=asyncio.create_task(classify_candidates(source,primary,plan['marker_series'],folder/'jev-vision',mode=mode,emit=runner.emit))
        if mode=='assist':jev=await jev_task
    sem=asyncio.Semaphore(max(1,min(4,concurrency)));results=[None]*len(batches)
    async def review(index,batch):
        async with sem:
            child=runner.fork();ids=[p['id'] for p in batch['candidates']]
            schema=_json_schema({'items':{'type':'array','items':_json_schema({
                'candidate_id':{'type':'string','enum':ids},'series_id':{'type':'string','enum':[f'S{i}' for i in range(len(series))]},
                'marker_visible':{'type':'boolean'},'center_supported':{'type':'boolean'},'verdict':{'type':'string','enum':['confirmed','uncertain']},'offset':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},
                'reason':{'type':'string'}})},'notes':{'type':'array','items':{'type':'string'}}})
            compact=[{k:p[k] for k in ('id','pixel','offset_limit','detector','template_series','center_stable','reserve_reason','representative_id') if k in p} for p in batch['candidates']]
            hints={k:v['choice'] for k,v in jev['answers'].items() if k in ids}
            prompt=f'''Your task is to classify the CENTER of each labeled candidate patch, not to enumerate points from the full chart. Images: labeled patches FIRST, local source crop SECOND, full source THIRD, then enlarged legend. Inspect every candidate in this {batch['kind']} batch. Return only visibly supported experimental markers centered at the pink crosshair (or within the permitted small offset). Omit line-only/text/axis/blank seeds. A nearby marker elsewhere in a patch does not make its center a marker. Set marker_visible and center_supported true only with visible glyph evidence. Uncertainty is NOT permission to accept a line or an off-center patch. Never infer a point from a trend or regular x spacing. Reserve proposals can contain real missed markers: inspect them too.
Assign series by visible glyph shape and legend. Distinguish + from x, square from upright triangle, diamond from downward triangle. Review intersections and boundary markers carefully. Choose only one ID for each physical marker. A candidate with representative_id is an alternative to that representative; use it only if the source supports a DIFFERENT marker or the representative is not centered. Template/exemplar series are hypotheses, not proof. If a template has wrong identity omit it or use another candidate; do not relabel it.
Return offset [dx,dy] in ORIGINAL source pixels within offset_limit; zero for centered points and fixed templates. A patch spans {2*max(8,round(max(image.size)/1432*18))+1} native pixels. Model coordinates cannot create candidates. Unstable/occluded centers and inseparable series remain uncertain. Brief reasons only; put missed markers or unresolved coverage in notes. Do not manufacture points when no candidate is supported.
Series: {json.dumps({f'S{i}':s for i,s in enumerate(plan['marker_series'])})}
Use symbol geometry for identity. Broad curve position is not evidence of marker shape.
Candidates: {json.dumps(compact)}
Optional experimental JEV hints (fallible; verify source independently): {json.dumps(hints)}'''
            try:
                await runner.emit({'type':'stage','stage':'coordinate_review','message':f'Reviewing candidate batch {index+1}/{len(batches)} · up to {concurrency} concurrent reviews'})
                for attempt in range(2):
                    try:
                        result=await child.call(prompt=prompt,schema=schema,images=[batch['catalogue_path'],batch['source_path'],str(source),*legends],
                            workdir=worker_dir(str(folder),f'marker-batch-{index}'+('-repair' if attempt else '')))
                        result['items']=[item for item in result['items'] if item.get('marker_visible') and item.get('center_supported')]
                        bound=bind_inventory({'selections':[]},batch['candidates'],[result],series)
                        if any(not(0<=p['pixel'][0]<image.width and 0<=p['pixel'][1]<image.height) for p in bound['selections']):raise ValueError('Center outside source')
                        break
                    except (LunaError,ValueError,KeyError,TypeError) as exc:
                        if attempt:result={'items':[],'notes':[f'Batch {index+1} could not be reviewed; coverage unresolved.']}
                        else:prompt+='\nRepair invalid result: '+str(exc)
                results[index]=result
                (folder/f'marker-batch-{index}.json').write_text(json.dumps(result,indent=2))
            finally:await runner.release(child)
    tasks=[asyncio.create_task(review(i,b)) for i,b in enumerate(batches)]
    try:
        await asyncio.gather(*tasks)
        if jev_task:jev=await jev_task
    except BaseException:
        for task in tasks:task.cancel()
        if jev_task:jev_task.cancel()
        await asyncio.gather(*tasks,*([jev_task] if jev_task else []),return_exceptions=True)
        raise
    validated,rejected=await run_local(validate_marker_selections,source,candidates,results,plan['marker_series'],layout)
    (folder/'local-acceptance.json').write_text(json.dumps({'rejected':rejected,'policy':'Model identities require local marker support; rejected claims remain coverage issues.'},indent=2))
    merged,suppressed=reconcile(validated,candidates,series)
    updated=bind_inventory(plan,candidates,merged,series)
    updated['notes'].extend(note for r in results for note in r.get('notes',[]))
    updated['_beta_grounded']=True
    if mode != 'off':
        updated['auxiliary_inference']={k:jev.get(k) for k in ('status','mode','model','batches','failed_batches','policy')}
        if jev.get('status') != 'complete':
            updated.setdefault('notes',[]).append('JEV was unavailable or incomplete; the selected agent reviewed all source candidates.')
            await runner.emit({'type':'review_note','message':'JEV assistance was incomplete; all source review remained with the selected agent.'})
    updated['inventory_complete']=False # No automatic assertion of complete recall.
    (folder/'marker-inventory.json').write_text(json.dumps({'batches':results,'suppressed_duplicates':suppressed},indent=2))
    # Independent coverage audit, including unsupported/missing detections.
    overlay=folder/'beta-inventory-overlay.png'
    await run_local(selection_overlay,source,updated['selections'],overlay)
    audit_schema=_json_schema({'coverage_supported':{'type':'boolean'},'issues':{'type':'array','items':{'type':'string'}}})
    try:
        audit=await runner.call(prompt=f'Compare original source and inventory overlay. Audit whether experimental markers in all requested series are represented, including origin, overlaps and boundary points. Overlay rings are not truth. Report missing markers, ghosts or series confusion in issues. Do not estimate coordinates. Set coverage_supported=false if completeness is uncertain. Series: {json.dumps(series)}',
            schema=audit_schema,images=[str(source),str(overlay)],workdir=worker_dir(str(folder),'scatter-coverage'))
        failures=[note for r in results for note in r.get('notes',[]) if 'could not be reviewed' in note]
        issues=list(audit['issues'])+failures
        unsupported=[r for r in rejected if not r['local_fit']['line_only']]
        if unsupported:
            issues.append(f'Local validation withheld {len(unsupported)} unsupported marker claims; inspect local-acceptance.json and source coverage.')
        updated['inventory_complete']=audit['coverage_supported'] and not issues
        updated['notes'].extend(issues);updated['unresolved_inventory']=issues
        (folder/'scatter-coverage.json').write_text(json.dumps(audit,indent=2))
    except LunaError:
        updated['unresolved_inventory']=['Source-wide coverage audit did not complete.']
    return updated
