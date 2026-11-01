"""Optional text-only JEV diagnostics. Does not manufacture raster evidence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.request import Request,urlopen

ENDPOINT="https://api.typesafe.ai/v1/systemone"
SOURCE_THREAD="01a0f0ab-a790-7b20-a944-84450bb7e7a9"


def authorized_key():
    key=os.environ.get("TYPESAFE_API_KEY")
    if key:return key
    # The human explicitly supplied this chat as the credential source. Never
    # search other conversations, print credentials, or include them in logs.
    codex_dir=Path(os.environ.get("CODEX_HOME",str(Path.home()/".codex")))
    try:
        con=sqlite3.connect(f"file:{codex_dir/'state_5.sqlite'}?mode=ro",uri=True)
        row=con.execute("select rollout_path from threads where id=?",(SOURCE_THREAD,)).fetchone();con.close()
        if row:
            for line in Path(row[0]).read_text().splitlines():
                record=json.loads(line);p=record.get("payload",{})
                if record.get("type") in ("response_item","retained_context","world_state"):
                    match=re.search(r"apikey_[a-zA-Z0-9_]+",json.dumps(record))
                    if match:return match.group(0)
    except (sqlite3.Error,OSError,ValueError):pass
    raise RuntimeError("JEV credential unavailable in environment or explicitly authorized source chat")


def classify(state:dict,cache_dir:str,model="jev-latest"):
    payload={"model":model,"state":state,"questions":{
        "axis_scale":{"type":"choice","instructions":"Using supplied OCR ticks, geometric residuals, and image-derived axis evidence ONLY, which axis structure is best supported? Do not invent missing ticks. Choose uncertain if evidence is insufficient.",
                      "criteria":{"linear":"Linear axis","log10":"Logarithmic axis","segmented":"Broken/segmented axis","uncertain":"Insufficient or conflicting evidence"}},
        "target_type":{"type":"choice","instructions":"What measurement route does the supplied target and style evidence support?", 
                       "criteria":{"curve":"Continuous or dashed fitted/experimental curve","markers":"Experimental marker centers","mixed":"Both separate markers and curves","uncertain":"Insufficient evidence"}},
        "needs_review":{"type":"noul","instructions":"Does supplied evidence indicate unresolved panel/series identity, invalid calibration or an incomplete inventory needing image review?",
                        "criteria":{"true":"Needs image review","false":"Supported by supplied evidence"}}}}
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    folder=Path(cache_dir);folder.mkdir(parents=True,exist_ok=True)
    path=folder/(signature+".json")
    if path.exists():return {**json.loads(path.read_text()),"cache_hit":True}
    req=Request(ENDPOINT,data=json.dumps(payload).encode(),headers={"Authorization":"Bearer "+authorized_key(),"Content-Type":"application/json","Accept":"application/json"},method="POST")
    with urlopen(req,timeout=60) as response:body=json.load(response)
    answers=body.get("answers")
    if not isinstance(answers,dict):raise RuntimeError("JEV response missing typed answers")
    usage=body.get("usage") or {};tokens=usage.get("input_tokens")
    result={"input_hash":signature,"model":body.get("model",model),"state":state,"answers":answers,"usage":usage,
            "api_equivalent_usd":tokens*.042/1e6 if isinstance(tokens,int) else None,
            "price_source":"https://typesafe.ai/","price_checked":"2026-10-02","cache_hit":False,
            "recognition_enabled":False,"note":"Ablation evidence required before enabling; no direct image support."}
    path.write_text(json.dumps(result,indent=2)+"\n")
    return result
