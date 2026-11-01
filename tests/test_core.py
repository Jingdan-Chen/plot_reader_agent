from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image, ImageDraw

from plot_reader.bench import _mpl_case, _pillow_case, grade
from plot_reader.core import Axis, _axis_from_selection, inspect_plot, measure_plot, refine_region


def test_axis_reversed_log_and_check() -> None:
    linear = Axis([{"pixel": 100, "value": 130}, {"pixel": 900, "value": 50}], checks=[{"pixel": 500, "value": 90}])
    assert linear.convert(500) == pytest.approx(90)
    assert linear.residual() == pytest.approx(0)
    logarithmic = Axis([{"pixel": 50, "value": 1}, {"pixel": 650, "value": 1000}], "log10")
    assert logarithmic.convert(250) == pytest.approx(10)
    with pytest.raises(ValueError, match="positive"):
        Axis([{"pixel": 0, "value": 0}, {"pixel": 100, "value": 10}], "log10")


def test_tick_id_axis_selection_checks_independent_tick() -> None:
    ticks = [{"id": "X1", "pixel": 100, "value": 1}, {"id": "X2", "pixel": 300, "value": 10}, {"id": "X3", "pixel": 500, "value": 100}]
    axis = _axis_from_selection({"anchor_ids": ["X1", "X3"], "check_ids": ["X2"], "scale": "log10"}, ticks)
    assert axis.convert(300) == pytest.approx(10)
    assert axis.residual() == pytest.approx(0)
    with pytest.raises(ValueError, match="independent"):
        _axis_from_selection({"anchor_ids": ["X1", "X3"], "check_ids": ["X3"]}, ticks)
    with pytest.raises(ValueError, match="not found"):
        _axis_from_selection({"anchor_ids": ["X1", "X4"]}, ticks)


def _chart(path: Path) -> tuple[dict, dict]:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    xs = np.linspace(0, 10, 700)
    ys = 0.2 + np.exp(-((xs-3)/0.2)**2) + 1.6 * np.exp(-((xs-7)/0.25)**2)
    ax.plot(xs, ys, color="black", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    def px(x, y):
        a, b = ax.transData.transform((x, y))
        return [a, height - b]
    axis = {"anchors": [{"pixel": px(0, 0)[0], "value": 0}, {"pixel": px(10, 0)[0], "value": 10}], "checks": [{"pixel": px(5, 0)[0], "value": 5}]}
    y_axis = {"anchors": [{"pixel": px(0, 0)[1], "value": 0}, {"pixel": px(0, 2)[1], "value": 2}], "checks": [{"pixel": px(0, 1)[1], "value": 1}]}
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return axis, y_axis


def test_inspect_measure_export_and_crop_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "plot.png"
    x_axis, y_axis = _chart(path)
    session = inspect_plot(str(path), "two peaks")
    assert session["image_size"] == [960, 600]
    assert Path(session["overview_path"]).is_file()
    crop = refine_region(session["session_id"], [100, 100, 400, 350])
    assert crop["bounds_original"] == [100, 100, 400, 350]
    assert Image.open(crop["crop_path"]).width == 300 * crop["display_scale"]
    x0 = x_axis["anchors"][0]["pixel"]
    slope = (x_axis["anchors"][1]["pixel"] - x0) / 10
    y0 = y_axis["anchors"][0]["pixel"]
    yrange = y0 - y_axis["anchors"][1]["pixel"]
    result = measure_plot(session["session_id"], x_axis, [
        {"label": "A", "pixel": [x0 + 3*slope, y0 - 1.2*yrange/2]},
        {"label": "B", "pixel": [x0 + 7*slope, y0 - 1.8*yrange/2]},
    ], y_axis, "A", y0)
    assert [round(r["x"], 3) for r in result["points"]] == [3, 7]
    assert result["points"][0]["normalized_height"] == 1
    assert result["points"][1]["normalized_height"] == pytest.approx(1.5)
    assert result["points"][0]["normalized_height_uncertainty"] == 0
    assert result["points"][1]["normalized_height_uncertainty"] > 0
    saved = json.loads(Path(result["json_path"]).read_text())
    assert saved["points"] == result["points"]
    assert Path(result["csv_path"]).is_file()
    overlay = Image.open(result["overlay_path"]).convert("RGB")
    px, py = (round(v) for v in result["points"][0]["pixel"])
    assert overlay.getpixel((px+7, py))[0] > 180  # red ring at exported pixel


def test_reject_wrong_check_and_normalization_without_baseline(tmp_path: Path) -> None:
    path = tmp_path / "plot.png"
    x_axis, _ = _chart(path)
    session = inspect_plot(str(path))
    x_axis["checks"][0]["value"] = 7
    with pytest.raises(ValueError, match="check tick"):
        measure_plot(session["session_id"], x_axis, [{"label": "a", "pixel": [200, 200]}])
    x_axis["checks"][0]["value"] = 5
    with pytest.raises(ValueError, match="require a baseline"):
        measure_plot(session["session_id"], x_axis, [{"label": "a", "pixel": [200, 200]}], reference_label="a")


def test_normalized_heights_use_baseline_without_y_axis(tmp_path: Path) -> None:
    path = tmp_path / "plot.png"
    x_axis, _ = _chart(path)
    session = inspect_plot(str(path))
    result = measure_plot(session["session_id"], x_axis, [
        {"label": "reference", "pixel": [300, 300]},
        {"label": "other", "pixel": [500, 400]},
    ], reference_label="reference", baseline_pixel=500)
    assert result["points"][0]["y"] is None
    assert result["points"][0]["normalized_height"] == 1
    assert result["points"][1]["normalized_height"] == pytest.approx(.5)
    assert result["points"][1]["normalized_height_uncertainty"] > 0


def test_curve_query_uses_x_value_and_visible_trace(tmp_path: Path) -> None:
    path = tmp_path / "plot.png"
    x_axis, y_axis = _chart(path)
    session = inspect_plot(str(path))
    x=x_axis["anchors"][0]["pixel"] + .5*(x_axis["anchors"][1]["pixel"]-x_axis["anchors"][0]["pixel"])
    hint_y=y_axis["anchors"][0]["pixel"]-.1*(y_axis["anchors"][0]["pixel"]-y_axis["anchors"][1]["pixel"])
    result=measure_plot(session["session_id"],x_axis,[{"label":"at 5","mode":"curve","x_value":5,"pixel":[x,hint_y]}],y_axis)
    assert result["points"][0]["x"] == pytest.approx(5)
    assert result["points"][0]["method"] == "curve_trace_at_x"


def test_local_refinement_candidate_is_measurable(tmp_path: Path) -> None:
    path=tmp_path/"plot.png"
    x_axis,_=_chart(path)
    session=inspect_plot(str(path))
    px=x_axis["anchors"][0]["pixel"]+.3*(x_axis["anchors"][1]["pixel"]-x_axis["anchors"][0]["pixel"])
    region=refine_region(session["session_id"],[round(px)-24,145,round(px)+24,480])
    local=[c for c in region["candidates"] if c["id"].startswith("R")]
    assert local
    nearest=min(local,key=lambda c:abs(c["pixel"][0]-px))
    result=measure_plot(session["session_id"],x_axis,[{"label":"peak","candidate_id":nearest["id"],"mode":"peak"}])
    assert result["points"][0]["x"]==pytest.approx(3,abs=.1)


def test_transparent_input_is_composited(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (500, 400), (0, 0, 0, 0))
    image.save(path)
    session = inspect_plot(str(path))
    assert Image.open(session["overview_path"]).convert("RGB").getpixel((250, 200)) == (255, 255, 255)


def test_scatter_centers_and_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "scatter.png"
    image = Image.new("RGB", (500, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(60, 40), (60, 350), (460, 350)], fill="black", width=2)
    draw.ellipse((194, 144, 206, 156), fill="#215a9b")
    draw.ellipse((294, 194, 306, 206), fill="#215a9b")
    image.save(path)
    session = inspect_plot(str(path))
    markers = [c for c in session["candidates"] if c["kind"] == "marker"]
    chosen = min(markers, key=lambda c: abs(c["pixel"][0]-200)+abs(c["pixel"][1]-150))
    assert np.linalg.norm(np.asarray(chosen["pixel"])-[200, 150]) <= 2
    x_axis = {"anchors": [{"pixel": 60, "value": 0}, {"pixel": 460, "value": 10}]}
    y_axis = {"anchors": [{"pixel": 350, "value": 0}, {"pixel": 50, "value": 3}]}
    point = measure_plot(session["session_id"], x_axis, [{"label": "A", "candidate_id": chosen["id"], "mode": "scatter"}], y_axis)["points"][0]
    assert point["pixel"] == pytest.approx([200, 150], abs=2)
    assert point["x_uncertainty_transformed"] > 0
    assert point["y_uncertainty_transformed"] > 0


def test_peak_refinement_stays_in_selected_neighborhood(tmp_path: Path) -> None:
    path = tmp_path / "close-peaks.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    xs = np.linspace(0, 10, 1500)
    ys = .15 + .7*np.exp(-((xs-4)/.07)**2) + 1.4*np.exp(-((xs-4.5)/.07)**2)
    ax.plot(xs, ys, color="black", linewidth=2)
    ax.set(xlim=(0, 10), ylim=(0, 2))
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    first = ax.transData.transform((4, .85))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    session = inspect_plot(str(path))
    x_axis = {"anchors": [{"pixel": 120, "value": 0}, {"pixel": 864, "value": 10}]}
    point = measure_plot(session["session_id"], x_axis, [{"label": "first", "pixel": [first[0], height-first[1]], "mode": "peak", "radius": 10}])["points"][0]
    assert point["x"] == pytest.approx(4, abs=.08)


def test_printed_labels_guard_reversed_peak_assignment(tmp_path: Path) -> None:
    path = tmp_path / "labeled-peaks.png"
    truth = _pillow_case(path, 40981, "peaks")
    session = inspect_plot(str(path))
    hinted = {c.get("label_hint"): c for c in session["candidates"] if c["kind"] == "peak" and c.get("label_hint")}
    assert "A" in hinted and "B" in hinted
    assert [t["pixel"] for t in session["ticks"]["x"]] == [105, 250, 395, 540, 685]
    x_axis = {"anchors": [{"pixel": 105, "value": 10}, {"pixel": 830, "value": 0}]}
    with pytest.raises(ValueError, match="printed label B"):
        measure_plot(session["session_id"], x_axis, [{"label": "A", "candidate_id": hinted["B"]["id"]}])
    measured = measure_plot(session["session_id"], {"anchor_ids": ["X1", "X5"], "check_ids": ["X3"]}, [
        {"label": "A", "candidate_id": hinted["A"]["id"]},
        {"label": "B", "candidate_id": hinted["B"]["id"]},
    ])
    for point in measured["points"]:
        assert np.linalg.norm(np.asarray(point["pixel"])-truth["pixel_truth"][point["label"]]) <= 3
        assert point["x"] == pytest.approx(truth["truth"][point["label"]][0], abs=.04)


def test_ocr_recovers_logarithmic_power_ticks(tmp_path: Path) -> None:
    path = tmp_path / "log-scatter.png"
    _mpl_case(path, 40942, "scatter_log")
    session = inspect_plot(str(path))
    assert [t["value"] for t in session["ticks"]["x"]] == [1, 10, 100, 1000]


def test_scatter_marker_survives_ocr_symbol(tmp_path: Path) -> None:
    path = tmp_path / "scatter-symbol.png"
    truth = _pillow_case(path, 41033, "scatter")
    session = inspect_plot(str(path))
    target = np.asarray(truth["pixel_truth"]["A"])
    markers = [np.asarray(c["pixel"]) for c in session["candidates"] if c["kind"] == "marker"]
    assert min(np.linalg.norm(candidate-target) for candidate in markers) <= 3


def test_curve_grading_accepts_equivalent_query_label(tmp_path: Path) -> None:
    gold = tmp_path / "gold.json"
    results = tmp_path / "results.json"
    gold.write_text(json.dumps({"cases": [{"id": "curve", "stage": "development", "kind": "curve", "truth": {"at 2.5": [2.5, .7]}, "pixel_truth": {"at 2.5": [100, 200]}}]}))
    results.write_text(json.dumps([{"id": "curve", "points": [{"label": "x=2.5", "pixel": [101, 201]}]}]))
    assert grade(gold, results)["complete_count"] == 1
