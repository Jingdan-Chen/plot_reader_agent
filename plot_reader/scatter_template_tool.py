"""Job-scoped internal tool that LUNA may request through a bounded dispatcher.

The browser's image-only runner has no arbitrary filesystem or shell tools. A
structured call here is validated and executed by the host; results and images
are supplied to LUNA's next decision. This is not a native MCP transport.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .timing import run_local, local_work
from .luna_runtime import LunaError, _json_schema, worker_dir
from .scatter_templates import component_atlas, legend_components, save_evidence, scan_templates, validate_profile


SIZE_BANKS = {"legend_size": [.85, 1., 1.15], "smaller_markers": [.55, .65, .75, .85, 1.]}
SENSITIVITY = {"balanced": (.57, .06), "conservative": (.64, .08)}
TOOL_NAME = "scan_scatter_templates"
TOOL_DESCRIPTION = """Find repeated dark scatter glyphs by source-template correlation.
Use when several plotted markers share a clear shape and similar size and the
catalogue contains a complete, isolated matching legend glyph. You may scan only
the supported series. Skip for curves/spectra without markers, varying symbols,
unusable legend glyphs or inseparable shapes. Legend and plotted sizes may differ;
smaller_markers tests smaller copies. balanced favors more candidates;
conservative requires stronger evidence. Returns native-pixel candidate IDs,
per-series response maps, conflicts, and fixed centers. Scores are NOT confidence;
line crossings and occlusions can produce misses or false matches. The method
supplements the existing detector. Never treat missing responses as proof that
there is no marker. Do not lower thresholds or rescan to achieve an expected count.
"""


def _public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


class ScatterTemplateTool:
    """One source/layout/series binding, no model-controlled paths or kernels."""

    def __init__(self, source: Path, layout: dict, series: list[str], folder: Path):
        self.source, self.layout, self.folder = source, layout, folder
        self.series = {f"S{i}": name for i, name in enumerate(series)}
        self.source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        self.components = legend_components(source, layout.get("exclude_regions", []))
        self.atlas = folder/"legend-catalogue.png"
        self.scans = 0
        self.cache = {}
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"legend-components.json").write_text(json.dumps(self.components, indent=2))
        if self.components:
            component_atlas(source, self.components, self.atlas)

    @property
    def argument_schema(self):
        return _json_schema({
            "templates": {"type": "array", "minItems": 1, "maxItems": len(self.series),
                "items": _json_schema({"series_id": {"type": "string", "enum": list(self.series)},
                                       "glyph_id": {"type": "string", "enum": [c["id"] for c in self.components]}})},
            "size_search": {"type": "string", "enum": list(SIZE_BANKS)},
            "sensitivity": {"type": "string", "enum": list(SENSITIVITY)},
        })

    def definition(self):
        return {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": self.argument_schema}

    def execute(self, name: str, arguments: dict) -> dict:
        if name != TOOL_NAME:
            raise ValueError("Unknown internal tool")
        if not isinstance(arguments, dict) or set(arguments) != {"templates", "size_search", "sensitivity"}:
            raise ValueError("Invalid scan arguments; paths, coordinates and kernels are not accepted")
        if arguments["size_search"] not in SIZE_BANKS or arguments["sensitivity"] not in SENSITIVITY:
            raise ValueError("Unknown size or sensitivity preset")
        selected = arguments["templates"]
        if not isinstance(selected, list) or not 1 <= len(selected) <= len(self.series):
            raise ValueError("Select one or more supported series")
        if any(not isinstance(p, dict) or set(p) != {"series_id", "glyph_id"} or p["series_id"] not in self.series for p in selected):
            raise ValueError("Unknown series or malformed template choice")
        if len({p["series_id"] for p in selected}) != len(selected):
            raise ValueError("Duplicate series template")
        profile = {"applicable": True, "templates": [
            {"series": self.series[p["series_id"]], "component_id": p["glyph_id"]} for p in selected]}
        validate_profile(profile, self.components, [self.series[p["series_id"]] for p in selected])
        if hashlib.sha256(self.source.read_bytes()).hexdigest() != self.source_hash:
            raise ValueError("Source image changed during template analysis")
        key = json.dumps(arguments, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if self.scans >= 2:
            raise ValueError("Template scan budget exhausted")
        self.scans += 1
        output = self.folder/f"scan-{self.scans}"
        output.mkdir(exist_ok=False)
        threshold, margin = SENSITIVITY[arguments["sensitivity"]]
        candidates, maps = scan_templates(self.source, self.components, profile, self.layout,
            scales=SIZE_BANKS[arguments["size_search"]], threshold=threshold, line_margin=margin)
        remap = {c["id"]: f"K{self.scans}_{i}" for i, c in enumerate(candidates, 1)}
        names = {name: ident for ident, name in self.series.items()}
        for candidate in candidates:
            candidate["id"] = remap[candidate["id"]]
            candidate["alternatives"] = [remap[k] for k in candidate["alternatives"]]
            candidate.update(detector="source template correlation", template_series=names[candidate["series"]])
        paths = save_evidence(self.source, candidates, maps, output)
        result = {"tool": TOOL_NAME, "scan_id": f"scan-{self.scans}", "source_sha256": self.source_hash,
            "arguments": arguments, "candidates": candidates, "review_image_paths": paths,
            "counts": dict(Counter(p["series"] for p in candidates)),
            "ambiguous_or_unstable": sum(p["status"] == "needs_review" for p in candidates),
            "warnings": ["Response maxima are proposals, not confirmed data points.",
                         "An absent response does not establish an absent marker.",
                         "Legend size/appearance, crossing lines, and occlusion can defeat the template."],
            "_maps": maps, "_series_ids": names}
        (output/"tool-result.json").write_text(json.dumps(_public(result), indent=2, allow_nan=False))
        self.cache[key] = result
        return result


def _decision_schema(tool: ScatterTemplateTool, has_result: bool):
    actions = ["skip"]
    if tool.scans < 2:
        actions.append("call_tool")
    if has_result:
        actions.append("use_result")
    return _json_schema({
        "action": {"type": "string", "enum": actions},
        "tool_name": {"anyOf": [{"type": "string", "enum": [TOOL_NAME]}, {"type": "null"}]},
        "arguments": {"anyOf": [tool.argument_schema, {"type": "null"}]},
        "reason": {"type": "string"},
    })


async def request_template_evidence(runner, source: Path, layout: dict, series: list[str], folder: Path, target: str):
    """At most two actual scans/four decisions; all failure paths retain fallback."""
    if os.environ.get("PLOT_READER_SCATTER_TEMPLATES", "auto") == "off":
        return None
    trace, latest = [], None
    folder.mkdir(parents=True, exist_ok=True)

    def record(item):
        trace.append(item)
        (folder/"tool-trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False))

    try:
        tool = await run_local(ScatterTemplateTool, source, layout, series, folder)
        if not tool.components or not tool.series:
            record({"event": "unavailable", "reason": "No isolated legend glyphs or no series; use existing local review."})
            return None
        for turn in range(4):
            await runner.emit({"type": "stage", "stage": "coordinate_review",
                               "message": "LUNA is choosing whether to use scatter template evidence"})
            prompt = f"""You can request the internal tool defined below. The host executes the
structured request and returns its results. General shell/filesystem tools remain
unavailable. Decide from visible evidence whether this method helps the target.
Use action=call_tool with the tool name and arguments, or skip with both null.
After a result, use_result with both null makes its candidates/heatmaps available
to the later LOCAL source review. It does not accept any data points. You may make
one revised scan when evidence demonstrates a wrong size or poor template choice.
At most two scans are allowed. Pick only complete glyph IDs for supported series.
If only some series fit, scan that subset and retain the others for the existing
detector. Skip when unsuitable or unhelpful. Never guess coordinates or fabricate
kernels. Review source alongside heatmaps. Do not assume bright peaks are real.
Your reason must explain suitability, the evidence for a revision, or why to skip.
Target: {target}
Series: {json.dumps(tool.series, ensure_ascii=False)}
Source catalogue: {json.dumps(tool.components)}
Tool definition: {json.dumps(tool.definition())}
Prior decisions and tool results: {json.dumps(trace, ensure_ascii=False)}
Remaining scans: {2-tool.scans}.
Images: unannotated source, legend glyph catalogue, then latest per-series response
panels if any. Native response centers are fixed; matching scores are not probability.
"""
            images = [str(source), str(tool.atlas)] + (latest["review_image_paths"] if latest else [])
            try:
                decision = await runner.call(prompt=prompt, schema=_decision_schema(tool, latest is not None),
                    images=images, workdir=worker_dir(str(folder), f"decision-{turn+1}"))
                if not isinstance(decision, dict) or set(decision) != {"action", "tool_name", "arguments", "reason"}:
                    raise ValueError("Malformed tool decision")
                if not isinstance(decision["reason"], str) or not decision["reason"].strip():
                    raise ValueError("Explain the decision using visible evidence")
                action = decision["action"]
                if action not in _decision_schema(tool, latest is not None)["properties"]["action"]["enum"]:
                    raise ValueError("Action is not available at this stage")
                record({"event": "decision", "value": decision})
                if action != "call_tool":
                    if decision["tool_name"] is not None or decision["arguments"] is not None:
                        raise ValueError("Finishing decisions cannot contain tool arguments")
                    record({"event": "finish", "action": action, "reason": decision["reason"]})
                    return latest if action == "use_result" else None
                await runner.emit({"type": "stage", "stage": "coordinate_review", "message": "Running source-template scatter scan"})
                latest = await run_local(tool.execute, decision["tool_name"], decision["arguments"])
                record({"event": "tool_result", "value": _public(latest)})
            except LunaError as exc:
                record({"event": "fallback", "reason": str(exc)})
                return None
            except (ValueError, KeyError, TypeError) as exc:
                record({"event": "error", "reason": str(exc)})
        record({"event": "fallback", "reason": "Decision budget exhausted; use existing local review."})
    except (ValueError, OSError, cv2.error) as exc:
        record({"event": "fallback", "reason": str(exc)})
    return None


def tile_response_evidence(result: dict, bounds: list, output: Path) -> str:
    """Show each series' response in the same crop as the local source review."""
    left, top, right, bottom = map(int, bounds)
    maps = result["_maps"]
    cell_w, cell_h, cols = 260, 380, min(3, len(maps))
    panel = Image.new("RGB", (cols*cell_w, 40+((len(maps)+cols-1)//cols)*cell_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), "Template responses: rings highlight proposals, not confidence", fill="black")
    for i, (name, response) in enumerate(maps.items()):
        crop = response[top:bottom, left:right]
        heat = cv2.applyColorMap(np.uint8(np.clip(crop, 0, 1)*255), cv2.COLORMAP_INFERNO)
        im = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        im.thumbnail((cell_w-12, cell_h-32))
        factor_x, factor_y = im.width/(right-left), im.height/(bottom-top)
        d = ImageDraw.Draw(im)
        for candidate in result["candidates"]:
            x, y = candidate["pixel"]
            if candidate["series"] == name and left <= x < right and top <= y < bottom:
                x, y = (x-left)*factor_x, (y-top)*factor_y
                d.ellipse((x-4, y-4, x+4, y+4), outline="white")
        x, y = (i % cols)*cell_w+6, 40+(i // cols)*cell_h
        draw.text((x, y), result["_series_ids"][name], fill="black")
        panel.paste(im, (x, y+22))
    panel.save(output)
    return str(output)


def main():
    """Exercise the real internal loop on a saved job without changing its data."""
    import argparse
    from .luna_runtime import LunaRunner

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job, output = args.job_dir.resolve(), args.output.resolve()
    if output == job or job in output.parents or (output.exists() and any(output.iterdir())):
        parser.error("Use a fresh output folder outside the saved job")
    output.mkdir(parents=True, exist_ok=True)
    layout = json.loads((job/"marker-layout.json").read_text())
    plan = json.loads((job/"plan.json").read_text())
    series = list(dict.fromkeys(p.get("series", "Series") for p in plan["selections"]))

    async def emit(event):
        with (output/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(event)+"\n")
        if event.get("type") == "stage":
            print(event["message"], flush=True)

    result = asyncio.run(request_template_evidence(LunaRunner(emit, timeout=240), job/"source-image.png",
        layout, series, output, "Identify all experimental scatter markers; separate the fitted curves."))
    summary = {"status": "evidence_ready" if result else "existing_detector", "result": _public(result) if result else None}
    (output/"workflow-result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "counts": result["counts"] if result else {},
                      "scan_id": result["scan_id"] if result else None}, indent=2))


if __name__ == "__main__":
    main()
