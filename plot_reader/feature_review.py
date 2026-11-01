"""Local review proposals; never silently promote a shoulder/noise/clip to a peak."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from PIL import Image,ImageDraw


def review_features(result_path:str,output_dir:str)->dict:
    source=json.loads(Path(result_path).read_text());image=Image.open(source['source_image']).convert('RGB')
    folder=Path(output_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
    overlay=image.copy();draw=ImageDraw.Draw(overlay);records=[]
    for panel in source['panels']:
        for series in panel['series']:
            if series['kind']!='curve':continue
            rows=series['curve'];valid=[i for i,r in enumerate(rows) if r['original_pixel_xy'][1] is not None]
            peaks=[f['original_pixel_xy'][0] for f in series['features'] if f['feature_type']=='peak']
            proposals=[]
            for group in np.split(valid,np.flatnonzero(np.diff(valid)>1)+1):
                if len(group)<5:continue
                x=np.array([rows[i]['original_pixel_xy'][0] for i in group]);y=np.array([rows[i]['original_pixel_xy'][1] for i in group])
                smooth=gaussian_filter1d(y,1.6);slope=np.gradient(smooth)
                troughs,_=find_peaks(smooth,prominence=3)
                proposals.extend(('trough',float(x[i]),float(y[i]),'supported local minimum of visible signal; review spectral meaning') for i in troughs)
                turns,props=find_peaks(-y,prominence=.5)
                proposals.extend(('noise_or_small_feature',float(x[i]),float(y[i]),'sub-threshold raster turn; noise versus feature unresolved')
                                 for i,prom in zip(turns,props['prominences']) if prom<1.5 and all(abs(x[i]-p)>3 for p in peaks))
                shoulders,_=find_peaks(-abs(slope),prominence=.15,distance=5)
                for i in shoulders:
                    if i<4 or i+4>=len(x) or any(abs(x[i]-p)<7 for p in peaks):continue
                    if slope[i-4]*slope[i+4]>0 and abs(slope[i])<min(abs(slope[i-4]),abs(slope[i+4]))*.4 and np.quantile(y,.9)-y[i]>3:
                        proposals.append(('shoulder',float(x[i]),float(y[i]),'same-sign flanks with local slope flattening; inseparable peak count unresolved'))
                boundary=panel['bounds'][1]
                for at in (0,len(x)-1):
                    if y[at]<=boundary+2:
                        proposals.append(('clipped',float(x[at]),float(y[at]),'visible fragment ends at upper data boundary; hidden apex cannot be measured'))
                near=np.flatnonzero(y<=boundary+2)
                for span in np.split(near,np.flatnonzero(np.diff(near)>1)+1):
                    if len(span)>=3:
                        at=int(span[len(span)//2]);proposals.append(('clipped',float(x[at]),float(y[at]),'trace touches upper data boundary; hidden apex cannot be measured'))
            for row in rows:
                if row.get('pixel_status',row['status'])=='unresolved' and row['original_pixel_xy'][1] is not None:
                    px,py=row['original_pixel_xy'];proposals.append(('crossing_or_competing_path',px,py,row.get('reason') or 'multiple source paths'))
            # Group repeated local ambiguity flags; there is no count limit.
            kept=[]
            for kind,x,y,reason in sorted(proposals,key=lambda r:(r[0],r[1])):
                if any(k==kind and abs(x-px)<4 for k,px,_,_ in kept):continue
                kept.append((kind,x,y,reason))
            for kind,x,y,reason in kept:
                key=f"{panel['id']}:{series['id']}:R{len(records)+1}";crop=[max(0,round(x)-24),max(0,round(y)-24),min(image.width,round(x)+25),min(image.height,round(y)+25)]
                path=folder/f"crop-{len(records)+1}.png";image.crop(crop).save(path)
                records.append({'id':key,'panel_id':panel['id'],'series_id':series['id'],'feature_type':kind,
                                'original_pixel_xy':[x,y],'status':'unresolved','reason':reason,
                                'pixel_uncertainty':2,'calibrated_values':None,'review_crop':str(path),
                                'crop_original_pixel_bounds':crop,'crop_to_original_translation':crop[:2]})
                draw.rectangle((x-4,y-4,x+4,y+4),outline='#bf00bf')
    overlay.save(folder/'review-overlay.png')
    result={'schema_version':'feature-review-v1','source_result':str(Path(result_path).resolve()),
            'proposals':records,'overlay_path':str(folder/'review-overlay.png'),
            'policy':'proposals remain unresolved; validated measurements in source_result are unchanged'}
    (folder/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    return {'result_path':str(folder/'review.json'),'overlay_path':result['overlay_path'],'review_proposals':len(records)}
