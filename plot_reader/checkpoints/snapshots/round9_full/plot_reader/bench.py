"""Reproducible fixture generation and independent coordinate grading.

Ground truth is written outside the image directory. Give Luna only a case's
image and request text; never pass the gold JSON to a benchmark task.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _axis_mapping(ax, height):
    box=ax.get_window_extent()
    return {"x":{"pixels":[float(box.x0),float(box.x1)],"values":list(ax.get_xlim()),"scale":"log10" if ax.get_xscale()=="log" else "linear"},"y":{"pixels":[float(height-box.y0),float(height-box.y1)],"values":list(ax.get_ylim()),"scale":"log10" if ax.get_yscale()=="log" else "linear"}}


def _mapped_pixel(value, spec):
    a,b=spec["values"];p,q=spec["pixels"]
    if spec.get("scale")=="log10":
        value,a,b=map(math.log10,(value,a,b))
    return p+(value-a)*(q-p)/(b-a)


def _mpl_case(path: Path, seed: int, kind: str) -> dict:
    rng = random.Random(seed)
    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafbfc" if seed % 2 else "white")
    if seed % 3 == 0:
        ax.grid(True, color="#cfd4dc", linewidth=0.7)
    ax.set_xlabel("Position (arb. units)" if kind != "scatter_log" else "Concentration")
    ax.set_ylabel("Response")
    color = ["#2445a8", "#aa4937", "#357547"][seed % 3]
    if kind in {"peaks","lorentzian"}:
        centers = [2.1 + rng.uniform(-0.15, 0.15), 5.4 + rng.uniform(-0.15, 0.15), 8.0 + rng.uniform(-0.15, 0.15)]
        x = np.linspace(0, 10, 1200)
        if kind=="lorentzian":
            centers=[1.6+rng.uniform(-.1,.1),4.7+rng.uniform(-.1,.1),8.4+rng.uniform(-.1,.1)]
            y=.06+sum((.85+i*.23)/(1+((x-c)/(.12+i*.05))**2) for i,c in enumerate(centers))
        else:
            y = 0.08 + sum((1.1 + i * .25) * np.exp(-((x-c)/(0.12+i*.04))**2) for i,c in enumerate(centers))
        ax.plot(x, y, color=color, linewidth=1.8)
        for label, center in zip("ABC", centers):
            ax.text(center + .12, float(np.interp(center, x, y)) + .08, label, fontsize=11)
        truth = {label: [center, float(np.interp(center, x, y))] for label, center in zip("ABC", centers)}
        prompt = "Read the x and y coordinates of the peaks labeled A, B, and C from this plot. Use only the image."
        ax.set_ylim(0, 2.2)
        if seed % 2:
            ax.set_xlim(10, 0)
        else:
            ax.set_xlim(0, 10)
    elif kind in {"scatter_log","scatter_log2"}:
        ax.set_xscale("log")
        xs = [1.6, 7.0, 34.0, 250.0] if kind=="scatter_log" else [2.2,17.0,135.0,630.0]
        ys = [0.3, 0.9, 1.5, 0.7] if kind=="scatter_log" else [.55,1.6,.85,1.25]
        ax.scatter(xs, ys, c=color, s=70 if kind=="scatter_log" else 90, marker="o" if kind=="scatter_log" else "D")
        for label,x,y in zip("PQRS",xs,ys):
            ax.text(x*1.1,y+.06,label,fontsize=11)
        ax.set_xlim(1, 1000)
        ax.set_ylim(0, 2)
        truth = {label:[x,y] for label,x,y in zip("PQRS",xs,ys)}
        prompt = "Read the x and y coordinates of scatter markers Q and S from this logarithmic-x plot. Use only the image."
        truth = {k:v for k,v in truth.items() if k in "QS"}
    else:
        x = np.linspace(0, 10, 300)
        y = .3 + .11*x + .08*np.sin(x*.65) if kind=="curve" else .2+1.55*np.exp(-x/3.2)
        ax.plot(x,y,color=color,linewidth=2)
        ax.set_xlim(10,0) if seed%2 else ax.set_xlim(0,10)
        ax.set_ylim(0,2)
        queries = [2.5,7.5] if kind=="curve" else [1.5,6.5]
        truth = {f"at {q}":[q,float(np.interp(q,x,y))] for q in queries}
        prompt = f"Read the curve's y values at x={queries[0]} and x={queries[1]}. Use only the image."
    fig.canvas.draw()
    width,height = fig.canvas.get_width_height()
    pixel_truth={}
    for label,(x,y) in truth.items():
        px,py = ax.transData.transform((x,y))
        pixel_truth[label] = [float(px),float(height-py)]
    fig.savefig(path,dpi=120)
    plt.close(fig)
    return {"kind":kind,"prompt":prompt,"truth":truth,"pixel_truth":pixel_truth,"mapping":_axis_mapping(ax,height),"image_size":[width,height]}


def _pillow_case(path:Path, seed:int, kind:str) -> dict:
    rng = random.Random(seed)
    width,height=900,560
    image=Image.new("RGB",(width,height),"white")
    draw=ImageDraw.Draw(image)
    left,right,top,bottom=105,830,55,475
    draw.line([(left,top),(left,bottom),(right,bottom)],fill="black",width=2)
    reversed_axis=seed%2==1
    for i in range(6):
        x=left+i*(right-left)/5
        value=10-i*2 if reversed_axis else i*2
        draw.line((x,bottom,x,bottom+7),fill="black",width=2)
        draw.text((x-8,bottom+11),str(value),fill="black")
    for i in range(5):
        y=bottom-i*(bottom-top)/4
        draw.line((left-7,y,left,y),fill="black",width=2)
        draw.text((left-38,y-7),f"{i/2:g}",fill="black")
    def pix(x,y):
        xp=left+((10-x) if reversed_axis else x)*(right-left)/10
        return (xp,bottom-y*(bottom-top)/2)
    if kind in {"scatter","scatter_alt"}:
        points={"A":(2.3,.5),"B":(5.7,1.4),"C":(8.1,1.0)} if kind=="scatter" else {"D":(1.4,.4),"E":(4.9,1.2),"F":(8.7,1.7)}
        for label,(x,y) in points.items():
            px,py=pix(x,y)
            if kind=="scatter":draw.ellipse((px-6,py-6,px+6,py+6),fill="#155b9d")
            else:draw.rectangle((px-6,py-6,px+6,py+6),fill="#9a421c")
            draw.text((px+9,py-18),label,fill="black")
        prompt=f"Measure x and y for labeled scatter points {', '.join(points)}."
    else:
        xs=np.linspace(0,10,900)
        centers=[2.4,6.7] if kind=="peaks" else [3.3,8.2]
        ys=.1+sum((1.3+i*.35)*np.exp(-((xs-c)/(.2+i*.05))**2) for i,c in enumerate(centers)) if kind=="peaks" else .08+sum((1.1+i*.28)/(1+((xs-c)/(.11+i*.05))**2) for i,c in enumerate(centers))
        draw.line([pix(float(x),float(y)) for x,y in zip(xs,ys)],fill="#176094",width=3)
        points={"A":(centers[0],float(np.interp(centers[0],xs,ys))),"B":(centers[1],float(np.interp(centers[1],xs,ys)))}
        for label,(x,y) in points.items():
            px,py=pix(x,y);draw.text((px+9,py-25),label,fill="black")
        prompt="Read the x coordinates of the two peaks labeled A and B."
    if seed%3==0:
        image=image.convert("RGBA")
        image.putalpha(245)
    image.save(path)
    mapping={"x":{"pixels":[left,right],"values":[10,0] if reversed_axis else [0,10],"scale":"linear"},"y":{"pixels":[bottom,top],"values":[0,2],"scale":"linear"}}
    return {"kind":kind,"prompt":prompt,"truth":points,"pixel_truth":{k:list(pix(*v)) for k,v in points.items()},"mapping":mapping,"requested_axes":["x","y"] if kind in {"scatter","scatter_alt"} else ["x"],"image_size":[width,height]}


def _source_case(path: Path, kind: str) -> dict:
    """Render fixed-data cases adapted from the official Matplotlib gallery."""
    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=120)
    if kind == "gallery_sine":
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html
        x = np.arange(0.0, 2.0, 0.01)
        y = 1 + np.sin(2 * np.pi * x)
        ax.plot(x, y, color="#8a2568", linewidth=2.1)
        ax.set(xlim=(0, 2), ylim=(0, 2.2), xlabel="time (s)", ylabel="voltage (mV)")
        ax.grid(True, color="#dddddd")
        truth = {f"at {q}": [q, float(np.interp(q, x, y))] for q in (0.35, 1.4)}
        prompt = "Read the line's y values at x=0.35 and x=1.4."
        source = "https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html"
    else:
        # https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html
        rng = np.random.RandomState(19680801)
        x, y = rng.rand(50), rng.rand(50)
        colors = rng.rand(50)
        area = (30 * rng.rand(50)) ** 2
        ax.scatter(x, y, s=area, c=colors, alpha=0.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y")
        # Select visible, separated points deterministically before evaluation.
        chosen = [int(i) for i in np.argsort(-area) if .1 < x[i] < .9 and .1 < y[i] < .9 and all(np.hypot(x[i]-x[j], y[i]-y[j]) > .12 for j in range(50) if j != i)][:2]
        if len(chosen) < 2:
            raise ValueError("gallery scatter lacks two isolated markers")
        truth = {}
        for label, i in zip("UV", chosen):
            ax.text(float(x[i])+.018, float(y[i])+.018, label, fontsize=10, weight="bold")
            truth[label] = [float(x[i]), float(y[i])]
        prompt = "Read x and y for the scatter markers labeled U and V."
        source = "https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html"
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    pixel_truth = {}
    for label, (vx, vy) in truth.items():
        px, py = ax.transData.transform((vx, vy))
        pixel_truth[label] = [float(px), float(height-py)]
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"kind": kind, "prompt": prompt, "truth": truth, "pixel_truth": pixel_truth, "mapping":_axis_mapping(ax,height), "image_size": [width, height], "source": source}


def generate(out_dir:Path, gold_path:Path, count:int=24, seed:int=40929) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    gold_path.parent.mkdir(parents=True,exist_ok=True)
    cases=[]
    dev_kinds=[("mpl","peaks"),("mpl","scatter_log"),("mpl","curve"),("pillow","scatter"),("pillow","peaks")]
    holdout_kinds=[("mpl","lorentzian"),("mpl","scatter_log2"),("mpl","decay"),("pillow","scatter_alt"),("pillow","peaks_alt")]
    for i in range(count):
        stage="development" if i<count//2 else "holdout"
        available=dev_kinds if stage=="development" else holdout_kinds
        renderer,kind=available[(i if stage=="development" else i-count//2)%len(available)]
        case_id=uuid.uuid5(uuid.NAMESPACE_DNS,f"plot-reader-{seed}-{i}").hex[:12]
        path=out_dir/f"{case_id}.png"
        if stage == "holdout" and i >= count-2:
            kind = "gallery_sine" if i == count-2 else "gallery_scatter"
            renderer = "official_gallery"
            case = _source_case(path, kind)
        else:
            case=_mpl_case(path,seed+i*13,kind) if renderer=="mpl" else _pillow_case(path,seed+i*13,kind)
        cases.append({"id":case_id,"image":str(path),"stage":stage,"renderer":renderer,**case})
        (out_dir/f"{case_id}.txt").write_text(case["prompt"])
    gold_path.write_text(json.dumps({"seed":seed,"cases":cases},indent=2))
    return {"images":str(out_dir),"gold_path":str(gold_path),"count":count,"development":count//2,"holdout":count-count//2}


def grade(gold_path:Path, results_path:Path) -> dict:
    gold=json.loads(gold_path.read_text())
    results=json.loads(results_path.read_text())
    by_id={c["id"]:c for c in gold["cases"]}
    rows=[]
    for item in results:
        case=by_id[item["id"]]
        def canonical(label:str)->str:
            label=label.strip().lower()
            query=re.fullmatch(r"(?:at\s*)?(?:x\s*=\s*)?([-+]?\d+(?:\.\d+)?)",label) if label.startswith(("at ","x=")) else None
            return f"at {float(query.group(1)):g}" if query else label
        found={canonical(str(p["label"])):p for p in item.get("points",[])}
        label_rows=[]
        for label,expected in case["truth"].items():
            got=found.get(canonical(label))
            if got is None or got.get("status")=="unresolved" or got.get("pixel") is None:
                label_rows.append({"label":label,"status":"missing"})
                continue
            px,py=case["pixel_truth"][label]
            # Compare in image pixels; this works for linear and log axes.
            pixel_error=math.hypot(got["pixel"][0]-px,got["pixel"][1]-py)
            numeric_error=None
            if case.get("mapping"):
                try:
                    numeric_error=math.sqrt(sum((_mapped_pixel(got[axis],case["mapping"][axis])-case["pixel_truth"][label][index])**2 for index,axis in enumerate(("x","y")) if axis in case.get("requested_axes",["x","y"])))
                except (TypeError,ValueError,KeyError,OverflowError):numeric_error=math.inf
            passed=pixel_error<=3 and (numeric_error is None or numeric_error<=3)
            label_rows.append({"label":label,"status":"pass" if passed else "fail","pixel_error":pixel_error,"numeric_error_pixels":numeric_error if numeric_error is None or math.isfinite(numeric_error) else None,"grading_mode":"pixels_and_values" if case.get("mapping") else "legacy_pixels_only"})
        rows.append({"id":item["id"],"stage":case["stage"],"kind":case["kind"],"complete":all(r["status"]=="pass" for r in label_rows),"labels":label_rows,"seconds":item.get("seconds"),"cost_usd":item.get("cost_usd")})
    return {"cases":rows,"complete_count":sum(r["complete"] for r in rows),"total":len(rows)}


def main()->None:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    gen=sub.add_parser("generate")
    gen.add_argument("--images",type=Path,required=True)
    gen.add_argument("--gold",type=Path,required=True)
    gen.add_argument("--count",type=int,default=24)
    gen.add_argument("--seed",type=int,default=40929)
    check=sub.add_parser("grade")
    check.add_argument("--gold",type=Path,required=True)
    check.add_argument("--results",type=Path,required=True)
    args=parser.parse_args()
    output=generate(args.images,args.gold,args.count,args.seed) if args.command=="generate" else grade(args.gold,args.results)
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
