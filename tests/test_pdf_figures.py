from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from PIL import Image
import pytest
from starlette.testclient import TestClient

from plot_reader import browser, pdf_figures as intake


def materialized_run(tmp_path, pages=1):
    run=tmp_path/"run"; (run/"pictures").mkdir(parents=True); (run/"md").mkdir(); (run/"bibtex").mkdir()
    for i in range(pages):
        Image.new("RGB",(200,120),(i*40,80,100)).save(run/f"pictures/plot-{i}.png")
        (run/f"md/doc_{i:04d}.md").write_text(f"![plot](../pictures/plot-{i}.png)\n\nFigure {i+1}. Measured sample.\n")
    Image.new("RGB",(500,800),"white").save(run/"pictures/layout-preview.png")
    (run/"bibtex/citation.bib").write_text('@misc{example, title={Test}, author={Test author}, year={2026}}')
    (run/"benchmark.txt").write_text(f"Detected pages: {pages}\nAPI processing seconds: 1.1\nMaterialization seconds: 0.2\nTotal seconds: 1.3\n")
    return run


@pytest.mark.parametrize("count,start,end,expected", [(1,None,None,(1,1)),(10,None,None,(1,10)),(11,2,4,(2,4)),(11,1,11,(1,11))])
def test_page_range(count,start,end,expected):
    assert intake.page_range(count,start,end)==expected


@pytest.mark.parametrize("start,end",[(None,None),(0,3),(4,2),(1,12),(True,3),(1.5,3)])
def test_long_pdf_requires_valid_explicit_range(start,end):
    with pytest.raises(intake.FigureIntakeError): intake.page_range(11,start,end)


def test_only_markdown_pictures_become_choices_with_original_page_numbers(tmp_path):
    run=materialized_run(tmp_path,2)
    figures,report=intake.index_figures(run,5,6)
    assert [f["page"] for f in figures]==[5,6]
    assert [f["title"] for f in figures]==["Figure 1","Figure 2"]
    assert len(figures)==2 and report["picture_count"]==3  # Output visualization is not a figure.
    assert report["markdown_count"]==2 and report["page_range"]==[5,6]
    assert report["timings"]["Total seconds"]=="1.3"
    assert any("doi" in warning for warning in report["warnings"])


def test_missing_or_outside_picture_link_rejects_gallery(tmp_path):
    run=materialized_run(tmp_path)
    (run/"md/doc_0000.md").write_text('![unsafe](../../outside.png)')
    Image.new("RGB",(20,20)).save(tmp_path/"outside.png")
    with pytest.raises(intake.FigureIntakeError,match="escapes"):
        intake.index_figures(run,1,1)


def test_paddleocr_html_images_and_captions_are_indexed(tmp_path):
    run=materialized_run(tmp_path)
    (run/"md/doc_0000.md").write_text('<div><img src="../pictures/plot-0.png" alt="Image" width="77%" /></div>\n\n<div style="text-align:center;">Figure 7. Blue &amp; red.</div>')
    figures,_=intake.index_figures(run,2,2)
    assert len(figures)==1 and figures[0]["title"]=="Figure 7"
    assert figures[0]["caption"]=="Figure 7. Blue & red."
    (run/"md/doc_0000.md").write_text('<img src="../../outside.png">')
    with pytest.raises(intake.FigureIntakeError,match="escapes"):
        intake.index_figures(run,2,2)


def test_backend_validation_and_token_preflight_preserve_source(tmp_path,monkeypatch):
    monkeypatch.setenv("PADDLEOCR_BACKEND","MCP")
    assert intake.selected_backend()=="mcp"
    monkeypatch.setenv("PADDLEOCR_BACKEND","invalid")
    with pytest.raises(intake.FigureIntakeError,match="no fallback"):
        intake.selected_backend()
    monkeypatch.setenv("PADDLEOCR_BACKEND","api");monkeypatch.delenv("PADDLEOCR_API_TOKEN",raising=False)
    monkeypatch.setattr(intake,"skill_directory",lambda:tmp_path)
    source=tmp_path/"source.pdf";source.write_bytes(b"preserved")
    async def progress(*args):pytest.fail("Missing-token preflight cannot submit")
    with pytest.raises(intake.FigureIntakeError,match="PADDLEOCR_API_TOKEN"):
        asyncio.run(intake.extract_figures(source,2,1,2,tmp_path,progress))
    assert source.read_bytes()==b"preserved"


def test_page_subset_is_built_before_skill_script_is_called(tmp_path,monkeypatch):
    source=tmp_path/"source.pdf";source.write_bytes(b"original")
    calls=[]
    monkeypatch.setattr(intake.shutil,"which",lambda tool:"/bin/"+tool)
    async def run(command,stage):calls.append((command,stage));return ""
    monkeypatch.setattr(intake,"run_script",run)
    selected=asyncio.run(intake.select_pages(source,12,3,4,tmp_path))
    assert selected.name=="pages-3-4.pdf"
    assert calls[0][0][1:5]==["-f","3","-l","4"]
    assert [Path(p).name for p in calls[1][0][1:-1]]==["page-3.pdf","page-4.pdf"]
    assert source.read_bytes()==b"original"


def client_setup(tmp_path,monkeypatch,count):
    monkeypatch.setattr(browser,"DATA_ROOT",tmp_path/"data")
    monkeypatch.setattr(browser,"_pdf_page_count",lambda _:count)
    monkeypatch.setattr(browser.app.state,"token","test")
    return TestClient(browser.app,base_url="http://127.0.0.1")


def upload_pdf(client):
    r=client.post("/api/upload",files={"file":("paper.pdf",b"%PDF test","application/pdf")},headers={"x-plot-reader-token":"test"})
    assert r.status_code==200,r.text
    return r.json()


def poll(client,source):
    for _ in range(30):
        result=client.get(f"/api/source/{source['id']}/figures",headers={"x-plot-reader-token":"test"}).json()
        if result["status"]!="processing":return result
    pytest.fail("Fake OCR task did not finish")


def test_image_upload_is_already_selected_without_explanation(tmp_path,monkeypatch):
    with client_setup(tmp_path,monkeypatch,1) as client:
        image=io.BytesIO();Image.new("RGB",(400,300)).save(image,format="PNG")
        r=client.post("/api/upload",files={"file":("plot.png",image.getvalue(),"image/png")},headers={"x-plot-reader-token":"test"})
        assert r.status_code==200
        source=r.json();target=browser._read(browser._upload_dir(source["id"])/"target.json")
        assert target["crop"]==[0,0,400,300]
        assert source["target"]==target


def test_long_pdf_does_not_start_ocr_until_range_is_selected(tmp_path,monkeypatch):
    called=[]
    async def extract(*args):called.append(args);raise intake.FigureIntakeError("polling","provider failed")
    monkeypatch.setattr(browser,"extract_figures",extract)
    with client_setup(tmp_path,monkeypatch,11) as client:
        source=upload_pdf(client);assert source["requires_page_range"]
        assert called==[]
        url=f"/api/source/{source['id']}/figures";headers={"x-plot-reader-token":"test"}
        assert client.post(url,json={},headers=headers).status_code==400
        assert called==[]
        assert client.post(url,json={"page_start":2,"page_end":3},headers=headers).status_code==202
        result=poll(client,source)
        assert result["status"]=="error" and result["stage"]=="polling"
        assert called[0][2:4]==(2,3)
        assert (browser._upload_dir(source["id"])/"source.pdf").read_bytes()==b"%PDF test"
        assert client.post(url,json={"page_start":2,"page_end":3},headers=headers).status_code==429


def test_short_pdf_gallery_selection_uses_the_exact_extracted_picture(tmp_path,monkeypatch):
    calls=[]
    async def extract(source,count,start,end,folder,progress):
        calls.append((start,end))
        (folder/"pictures").mkdir();Image.new("RGB",(200,120),"red").save(folder/"pictures/one.png")
        return [{"id":"F001","page":1,"title":"Figure 1","caption":"Figure 1. Red plot", "image_size":[200,120],"path":"pictures/one.png"}],{"backend":"api"}
    monkeypatch.setattr(browser,"extract_figures",extract)
    with client_setup(tmp_path,monkeypatch,10) as client:
        source=upload_pdf(client);assert not source["requires_page_range"]
        headers={"x-plot-reader-token":"test"};url=f"/api/source/{source['id']}/figures"
        assert client.post(url,json={},headers=headers).status_code==202
        result=poll(client,source);assert result["status"]=="ready"
        assert calls==[(1,10)]
        assert "path" not in result["figures"][0]
        assert client.post(url,json={},headers=headers).status_code==200
        assert len(calls)==1
        target_body={"upload_id":source["id"],"figure_id":"F001","figure_run_id":result["run_id"]}
        target=client.post("/api/target",json=target_body,headers=headers)
        assert target.status_code==200,target.text
        assert target.json()["crop"]==[0,0,200,120]
        assert target.json()["explanation"]=="Figure 1. Red plot"
        output=tmp_path/"measured-source.png"
        metadata=browser._crop_source(source,target.json(),output)
        assert metadata["source_kind"]=="paddleocr_picture"
        assert Image.open(output).getpixel((50,50))==(255,0,0)
        preview=url+f"/{result['run_id']}/F001"
        assert client.get(preview,headers=headers).status_code==200
        assert client.get(preview).status_code==403
        target_body["figure_id"]="F999"
        assert client.post("/api/target",json=target_body,headers=headers).status_code==400


def test_sensitive_token_is_redacted_from_backend_failure(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_API_TOKEN","test-secret-value")
    assert "test-secret-value" not in intake.sanitize("submission failed test-secret-value")
    assert intake.failure_stage("Job polling failed")=="polling"
    assert intake.failure_stage("Result download failed")=="download"
    assert intake.failure_stage("PADDLEOCR_STAGE=submission\nConnection failed")=="submission"


def test_stage_annotation_preserves_inner_failure():
    from plot_reader.paddleocr_skill_runner import annotate,StageFailure
    def download():raise OSError("network failed")
    inner=annotate(download,"download")
    outer=annotate(inner,"materialization")
    with pytest.raises(StageFailure) as caught:outer()
    assert caught.value.stage=="download"


def test_three_failed_runs_cannot_be_bypassed_by_changing_range(tmp_path,monkeypatch):
    with client_setup(tmp_path,monkeypatch,11) as client:
        source=upload_pdf(client);folder=browser._upload_dir(source['id'])
        for i in range(3):
            browser._json(folder/'figure-intake'/f'{i:032x}'/'intake.json',{
                'status':'error','stage':'polling','page_start':2,'page_end':2,'finished_at':0})
        browser._json(folder/'figure-intake.json',{'status':'ready','page_start':3,'page_end':3})
        response=client.post(f"/api/source/{source['id']}/figures",json={'page_start':2,'page_end':2},headers={'x-plot-reader-token':'test'})
        assert response.status_code==409 and 'three times' in response.json()['error']
