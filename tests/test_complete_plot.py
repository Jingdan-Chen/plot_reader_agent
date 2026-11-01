import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from plot_reader.complete import Calibration, axis_hypotheses, exhaustive_peaks, extract_plot, trace_path
from plot_reader.complete import normalize_observation,baseline_model,validate_plan


def test_review_proposals_do_not_promote_clipped_reference_or_modify_measurements(tmp_path):
    from plot_reader.feature_review import review_features
    image=Image.new('RGB',(120,100),'white');src=tmp_path/'source.png';image.save(src)
    rows=[{'original_pixel_xy':[x,10 if 40<=x<=50 else 65], 'status':'resolved'} for x in range(15,100)]
    result={'source_image':str(src),'panels':[{'id':'p','bounds':[10,10,110,90],
             'series':[{'id':'s','kind':'curve','curve':rows,'features':[]}]}]}
    path=tmp_path/'result.json';path.write_text(json.dumps(result));before=path.read_bytes()
    out=review_features(str(path),str(tmp_path/'review'))
    review=json.loads(Path(out['result_path']).read_text())
    assert any(p['feature_type']=='clipped' for p in review['proposals'])
    assert all(p['status']=='unresolved' and p['calibrated_values'] is None for p in review['proposals'])
    assert path.read_bytes()==before


def test_ocr_packet_preserves_pixels_and_reports_omissions(tmp_path,monkeypatch):
    from plot_reader.complete import evidence_packet
    monkeypatch.setattr('plot_reader.complete.core._ocr',lambda image:[{'text':'100','box':[5,30,15,40],'confidence':70},
                                                                       {'text':'a','box':[10,3,20,14],'confidence':90}])
    src=tmp_path/'source.png';Image.new('RGB',(100,60),'white').save(src)
    packet=evidence_packet(str(src),1)
    assert packet['records'][0]['text']=='a'
    assert packet['omitted_records']==1 and packet['total_records']==2
    assert packet['records'][0]['original_pixel_bounds']==[10,3,20,14]


def test_experimental_cli_keeps_existing_measurement_calls(monkeypatch):
    import sys
    from plot_reader.complete_cli import main
    captured=[]
    monkeypatch.setattr(sys,'argv',['plot-reader','measure','session','--x-axis','{}'])
    monkeypatch.setattr('plot_reader.cli.main',lambda:captured.append(sys.argv[1:]))
    main()
    assert captured==[['measure','session','--x-axis','{}']]


def test_scale_is_selected_by_independent_ticks():
    log=axis_hypotheses([{"pixel":x,"value":v} for x,v in [(10,1),(110,10),(210,100),(310,1000)]])
    assert log["selected"]["scale"]=="log10"
    linear=axis_hypotheses([{"pixel":x,"value":v} for x,v in [(10,100),(110,70),(210,40),(310,10)]])
    assert linear["selected"]["scale"]=="linear"
    assert axis_hypotheses([{"pixel":10,"value":1},{"pixel":110,"value":10}])["selected"] is None


def test_segmented_axis_rejects_gap_and_duplicate_check():
    spec={"segments":[{"pixel_bounds":[0,100],"anchors":[{"pixel":0,"value":100},{"pixel":100,"value":60}],"checks":[{"pixel":50,"value":80}]},
                      {"pixel_bounds":[130,250],"anchors":[{"pixel":130,"value":30},{"pixel":250,"value":0}],"checks":[{"pixel":190,"value":15}]}]}
    axis=Calibration(spec,(0,250))
    assert axis.evidence["verified"]
    assert axis.convert(120)==(None,None)
    assert axis.pixel_for(45) is None
    assert axis.convert(190)[0]==pytest.approx(15)
    spec["segments"][0]["checks"]=[{"pixel":0,"value":100}]
    assert not Calibration(spec,(0,250)).evidence["verified"]


def test_peak_detection_does_not_truncate_dense_inventory():
    x=np.arange(1200)
    path=100-sum(40*np.exp(-.5*((x-c)/2)**2) for c in range(15,1190,25))
    peaks=exhaustive_peaks(path,[0,0,1200,150],np.zeros(1200,dtype=bool))
    assert len(peaks)==47


def test_path_keeps_missing_ink_missing():
    mask=np.zeros((120,200),dtype=np.uint8)
    mask[70,10:80]=255;mask[70,110:180]=255
    path,_=trace_path(mask,[0,0,200,120],[[20,70],[120,70]])
    assert np.isnan(path[90])
    assert path[20]==70 and path[120]==70


def test_complete_export_preserves_unverified_pixels_and_hollow_markers(tmp_path,monkeypatch):
    monkeypatch.setattr("plot_reader.complete.core._ocr",lambda image:[])
    image=Image.new("RGB",(220,150),"white");draw=ImageDraw.Draw(image)
    for x,y in [(40,40),(100,65),(180,90)]:draw.ellipse((x-4,y-4,x+4,y+4),outline=(30,70,210),width=2)
    src=tmp_path/"plot.png";image.save(src)
    plan={"inventory_complete":True,"panels":[{"bounds":[10,10,210,130],"series":[{"kind":"markers","color_rgb":[30,70,210]}]}]}
    summary=extract_plot(str(src),plan,str(tmp_path/"out"))
    result=json.loads(Path(summary["result_path"]).read_text())
    points=result["panels"][0]["series"][0]["features"]
    assert len(points)==3
    assert all(p["calibrated_x"] is None and p["original_pixel_xy"] for p in points)
    assert result["export_manifest"]["rows"]==3


def test_wrong_check_cannot_export_calibrated_values():
    spec={"anchors":[{"pixel":0,"value":0},{"pixel":100,"value":10}],"checks":[{"pixel":50,"value":8}]}
    axis=Calibration(spec,(0,100))
    assert not axis.evidence["verified"]
    assert axis.convert(50)==(None,None)


def test_narrow_turn_apex_uses_owned_run_tip_without_changing_curve():
    image=Image.new('L',(100,220),0);draw=ImageDraw.Draw(image)
    draw.line([(5,190),(50,30),(95,190)],fill=255,width=3)
    mask=np.asarray(image)
    geometry={}
    path,ambiguous=trace_path(mask,[0,0,100,220],[[20,137],[80,137]],evidence=geometry)
    before=path.copy()
    peaks=exhaustive_peaks(path,[0,0,100,220],ambiguous,geometry)
    assert len(peaks)==1
    assert abs(peaks[0]['original_pixel_xy'][0]-50)<=1
    assert abs(peaks[0]['original_pixel_xy'][1]-30)<=2
    np.testing.assert_equal(path,before)
    assert peaks[0]['apex_readout']=='selected_run_tip_with_both_flanks'


def test_one_seed_admits_visible_dashes_without_filling_blank_gaps():
    mask=np.zeros((100,180),dtype=np.uint8)
    for x in range(10,165,20):mask[40:42,x:x+12]=255
    path,_=trace_path(mask,[0,0,180,100],[[15,40]])
    assert np.isfinite(path).sum()==8*12
    assert np.isnan(path[25])


def test_log_y_normalization_uses_intensity_not_log_contrast():
    cal=Calibration({'scale':'log10','anchors':[{'pixel':0,'value':1000},{'pixel':300,'value':1}],
                     'checks':[{'pixel':100,'value':100}]},(0,300))
    evidence={'valid':True,'baseline_model':[0,300],'baseline_origin_pixel_x':80,
              'amplitude_pixels':300,'dominant_peak_original_pixel_xy':[200,0]}
    result=normalize_observation(100,100,evidence,cal)
    assert result['normalized_height']==pytest.approx(99/999)
    assert result['normalized_pixel_height']==pytest.approx(2/3)
    assert result['physical_normalization_verified']


def test_drift_origin_translation_preserves_normalized_values():
    a={'valid':True,'baseline_model':[.1,120],'baseline_origin_pixel_x':80,
       'amplitude_pixels':50,'dominant_peak_original_pixel_xy':[180,80]}
    b={**a,'baseline_model':[.1,122],'baseline_origin_pixel_x':100}
    assert normalize_observation(140,101,a,Calibration(None,(0,150)))==normalize_observation(140,101,b,Calibration(None,(0,150)))


def test_unobserved_baseline_anchor_preserves_raw_without_normalizing():
    x=np.arange(200);path=100-50*np.exp(-.5*((x-100)/10)**2)
    result=baseline_model(path,[0,0,200,140],{'baseline':{'anchors':[[10,130],[190,130]]}},np.zeros(200,bool))
    assert not result['valid']


def test_seed_validation_reports_observed_source_pixel(tmp_path):
    im=Image.new('RGB',(100,100),'white');ImageDraw.Draw(im).line([(10,60),(90,40)],fill='black',width=2)
    path=tmp_path/'line.png';im.save(path)
    errors=validate_plan(str(path),{'panels':[{'id':'P','bounds':[5,5,95,95], 'series':[{'kind':'curve','bounds':[5,5,95,95],'seeds':[[50,10]]}]}]})
    assert any('nearest observed pixel' in e for e in errors)


def test_one_unverified_segment_does_not_erase_verified_segment():
    axis=Calibration({'segments':[
        {'id':'left','pixel_bounds':[0,100],'anchors':[{'pixel':0,'value':100},{'pixel':100,'value':60}],'checks':[]},
        {'id':'right','pixel_bounds':[130,250],'anchors':[{'pixel':130,'value':30},{'pixel':250,'value':0}],
         'checks':[{'pixel':190,'value':15}]}]},(0,250))
    assert not axis.evidence['verified'] and axis.evidence['partially_verified']
    assert axis.convert(50)==(None,None)
    assert axis.convert(120)==(None,None)
    assert axis.convert(190)==(15,'right')
