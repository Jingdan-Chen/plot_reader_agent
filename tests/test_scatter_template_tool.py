from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from plot_reader import scatter_template_tool as module
from plot_reader.browser_grounding import bind_inventory
from plot_reader.luna_runtime import LunaError


def source_fixture(tmp_path):
    source = tmp_path/"source.png"
    im = Image.new("L", (320, 240), "white")
    d = ImageDraw.Draw(im)
    d.line((15, 80, 300, 80), fill=0)
    for x, y in [(70, 80), (150, 80), (40, 200)]:
        d.rectangle((x-5, y-5, x+5, y+5), outline=0, width=2)
    im.save(source)
    layout = {"plot_bounds": [10, 10, 310, 165], "exclude_regions": [[15, 175, 65, 225]],
              "reason": "plot and legend", "series_guidance": "S0 square"}
    args = {"templates": [{"series_id": "S0", "glyph_id": "G1"}],
            "size_search": "legend_size", "sensitivity": "balanced"}
    return source, layout, args


def decision(action, arguments=None):
    return {"action": action, "tool_name": module.TOOL_NAME if action == "call_tool" else None,
            "arguments": arguments, "reason": "Visible repeated square glyphs"}


class Runner:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls, self.events = [], []

    async def emit(self, event):
        self.events.append(event)

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.decisions)
        if isinstance(value, Exception):
            raise value
        return value


def test_model_requests_real_tool_and_receives_response_images(tmp_path):
    source, layout, args = source_fixture(tmp_path)
    runner = Runner([decision("call_tool", args), decision("use_result")])
    result = asyncio.run(module.request_template_evidence(runner, source, layout, ["square"], tmp_path/"tool", "all points"))
    assert result["counts"] == {"square": 2}
    assert all(c["offset_limit"] == 0 and c["template_series"] == "S0" for c in result["candidates"])
    assert result["review_image_paths"] == runner.calls[1]["images"][2:]
    assert '"tool_result"' in runner.calls[1]["prompt"]
    assert all(Path(p).is_file() for p in runner.calls[1]["images"])
    trace = json.loads((tmp_path/"tool/tool-trace.json").read_text())
    assert [p["event"] for p in trace] == ["decision", "tool_result", "decision", "finish"]
    image = module.tile_response_evidence(result, [10, 10, 180, 150], tmp_path/"tile.png")
    assert Image.open(image).width == 260


def test_skip_off_and_missing_glyphs_do_not_scan(tmp_path, monkeypatch):
    source, layout, _ = source_fixture(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail("No scan should execute")
    monkeypatch.setattr(module, "scan_templates", unexpected)
    runner = Runner([decision("skip")])
    assert asyncio.run(module.request_template_evidence(runner, source, layout, ["square"], tmp_path/"skip", "points")) is None
    assert len(runner.calls) == 1
    runner = Runner([])
    no_legend = {**layout, "exclude_regions": []}
    assert asyncio.run(module.request_template_evidence(runner, source, no_legend, ["square"], tmp_path/"none", "points")) is None
    monkeypatch.setenv("PLOT_READER_SCATTER_TEMPLATES", "off")
    assert asyncio.run(module.request_template_evidence(runner, source, layout, ["square"], tmp_path/"off", "points")) is None
    assert not runner.calls


def test_invalid_parameters_never_dispatch(tmp_path):
    source, layout, args = source_fixture(tmp_path)
    tool = module.ScatterTemplateTool(source, layout, ["square"], tmp_path/"tool")
    for invalid in [dict(args, source_path="/tmp/foreign.png"), dict(args, size_search="unlimited"),
                    dict(args, templates=[{"series_id": "S9", "glyph_id": "G1"}]),
                    dict(args, templates=[{"series_id": "S0", "glyph_id": "G9"}]),
                    dict(args, templates=args["templates"]*2)]:
        with pytest.raises(ValueError):
            tool.execute(module.TOOL_NAME, invalid)
    with pytest.raises(ValueError, match="Unknown internal"):
        tool.execute("shell", args)
    assert tool.scans == 0


def test_duplicate_scan_cached_but_changed_source_rejected(tmp_path):
    source, layout, args = source_fixture(tmp_path)
    tool = module.ScatterTemplateTool(source, layout, ["square"], tmp_path/"tool")
    first = tool.execute(module.TOOL_NAME, args)
    assert tool.execute(module.TOOL_NAME, copy.deepcopy(args)) is first
    assert tool.scans == 1
    Image.new("L", (320, 240), "white").save(source)
    with pytest.raises(ValueError, match="Source image changed"):
        tool.execute(module.TOOL_NAME, args)


def test_model_can_revize_once_but_cannot_exceed_scan_budget(tmp_path):
    source, layout, args = source_fixture(tmp_path)
    revised = dict(args, size_search="smaller_markers")
    runner = Runner([decision("call_tool", args), decision("call_tool", revised),
                     decision("call_tool", args), decision("use_result")])
    result = asyncio.run(module.request_template_evidence(runner, source, layout, ["square"], tmp_path/"tool", "points"))
    assert result["scan_id"] == "scan-2"
    assert "call_tool" not in runner.calls[2]["schema"]["properties"]["action"]["enum"]
    assert len(list((tmp_path/"tool").glob("scan-*"))) == 2
    trace = json.loads((tmp_path/"tool/tool-trace.json").read_text())
    assert any(p["event"] == "error" and "not available" in p["reason"] for p in trace)


def test_failure_falls_back_without_repeated_model_launches(tmp_path):
    source, layout, _ = source_fixture(tmp_path)
    runner = Runner([LunaError("model unavailable")])
    assert asyncio.run(module.request_template_evidence(runner, source, layout, ["square"], tmp_path/"tool", "points")) is None
    assert len(runner.calls) == 1
    assert json.loads((tmp_path/"tool/tool-trace.json").read_text())[-1]["event"] == "fallback"


def test_partial_series_and_locked_coordinates(tmp_path):
    source, layout, args = source_fixture(tmp_path)
    tool = module.ScatterTemplateTool(source, layout, ["square", "unknown"], tmp_path/"tool")
    result = tool.execute(module.TOOL_NAME, args)
    candidate = result["candidates"][0]
    item = {"candidate_id": candidate["id"], "series_id": "S0", "offset": [0, 0],
            "verdict": "confirmed", "reason": "source verifies square"}
    record = {"items": [item]}
    bound = bind_inventory({"selections": []}, [candidate], [record], ["square", "unknown"])
    assert bound["selections"][0]["pixel"] == [round(v, 2) for v in candidate["pixel"]]
    item["offset"] = [.1, 0]
    with pytest.raises(ValueError, match="local patch"):
        bind_inventory({"selections": []}, [candidate], [record], ["square", "unknown"])
    item.update(offset=[0, 0], series_id="S1")
    with pytest.raises(ValueError, match="different series"):
        bind_inventory({"selections": []}, [candidate], [record], ["square", "unknown"])


def test_uncertain_centers_and_cross_detector_duplicates_stay_flagged():
    candidates = [{"id": "K1_1", "pixel": [30, 30], "offset_limit": 0, "template_series": "S0",
                   "detector": "template", "center_stable": False},
                  {"id": "M1", "pixel": [30.2, 30.2], "offset_limit": 2, "detector": "contour"}]
    record = {"items": [{"candidate_id": c["id"], "series_id": "S0", "offset": [0, 0],
                         "verdict": "confirmed", "reason": "visible square"} for c in candidates]}
    bound = bind_inventory({"selections": []}, candidates, [record], ["square"])
    assert all(p["coordinate_review"] == "uncertain" for p in bound["selections"])
    assert all("Possible duplicate" in p["coordinate_review_note"] for p in bound["selections"])


def test_browser_grounding_consumes_tool_ids_and_local_heatmaps(tmp_path, monkeypatch):
    from plot_reader import browser_grounding as grounding
    source, layout, args = source_fixture(tmp_path)
    monkeypatch.setattr(grounding, "marker_candidates", lambda _: [])

    class BrowserRunner(Runner):
        async def call(self, **kwargs):
            self.calls.append(kwargs)
            stage = Path(kwargs["workdir"]).name
            if stage == "marker-layout":
                return layout
            if stage == "decision-1":
                return decision("call_tool", args)
            if stage == "decision-2":
                return decision("use_result")
            assert stage.startswith("marker-region-")
            assert kwargs["images"][-1].endswith("-responses.png")
            ids = kwargs["schema"]["properties"]["items"]["items"]["properties"]["candidate_id"]["enum"]
            return {"items": [{"candidate_id": ident, "series_id": "S0", "offset": [0, 0],
                              "verdict": "confirmed", "reason": "visible square"} for ident in ids], "notes": []}

    runner = BrowserRunner([])
    plan = {"selections": [{"label": "a", "series": "square", "pixel": [70, 80]},
                            {"label": "b", "series": "square", "pixel": [150, 80]}]}
    result = asyncio.run(grounding.ground_scatter(runner, plan, source, tmp_path, "all points"))
    assert len(result["selections"]) == 2
    assert all(p["grounding"]["version"] == "template-tool-v1" for p in result["selections"])
    assert result["selections"][0]["pixel"] == pytest.approx([70, 80], abs=.1)
    assert json.loads((tmp_path/"marker-grounding.json").read_text())["template_tool"]["scan_id"] == "scan-1"
