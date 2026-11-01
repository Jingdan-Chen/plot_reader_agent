from __future__ import annotations

import io
import csv
import json

from PIL import Image, ImageDraw
from starlette.testclient import TestClient

from plot_reader import browser


def test_bulk_point_request_and_aggregate_detection():
    request = {"category": "plot", "mode": "selected", "targets": "all data points"}
    assert browser._is_bulk_point_request(request)
    assert browser._is_aggregate_unresolved({"selections": [{"label": "all data points", "status": "unresolved"}]})
    assert not browser._is_bulk_point_request({**request, "targets": "peak A"})


def test_repeated_labels_are_unique_across_series_without_inventing_a_reference():
    from plot_reader.browser_evidence import unique_selection_labels

    plan = {"selections": [{"label": "t=0", "series": "square"},
                           {"label": "t=0", "series": "triangle"}], "reference_label": "t=0"}
    updated = unique_selection_labels(plan)
    assert [item["label"] for item in updated["selections"]] == ["square · t=0", "triangle · t=0"]
    assert [item["label"] for item in plan["selections"]] == ["t=0", "t=0"]
    assert updated["reference_label"] is None
    assert updated["notes"]


def test_distant_contour_cannot_silently_move_a_marker_to_another_series():
    from plot_reader.browser_evidence import bind_review_candidates

    result = bind_review_candidates({"items": [{"selection_id": "1", "candidate_id": "C2",
        "pixel": None, "verdict": "confirmed", "reason": "Diamond"}]},
        [{"id": "C1", "pixel": [338, 398]}, {"id": "C2", "pixel": [338, 645]}],
        [{"series": "diamond", "pixel": [338, 410]}], max_shift=40)
    assert result["items"][0]["pixel"] == [338, 398]
    assert result["items"][0]["verdict"] == "uncertain"


def test_reviewed_source_pixel_is_remeasured_and_exported(tmp_path, monkeypatch):
    import pytest
    from plot_reader import core
    from plot_reader.browser_evidence import apply_selection_review, selection_overlay, unsupported_centers

    monkeypatch.setattr(core, "SESSION_ROOT", tmp_path / "sessions")
    source = tmp_path / "source.png"
    image = Image.new("RGB", (600, 400), "white")
    ImageDraw.Draw(image).ellipse((294, 114, 306, 126), outline="black", width=2)
    image.save(source)
    assert unsupported_centers({"selections": [{"label": "bad", "pixel": [300, 300]}]}, source)[0]["ink_distance_pixels"] > 100
    assert not unsupported_centers({"selections": [{"label": "hollow", "pixel": [300, 120]}]}, source)
    inspected = core.inspect_plot(str(source))
    plan = {"selections": [{"label": "square 1", "series": "square", "pixel": [300, 300], "mode": "exact"}]}
    reviewed, warnings = apply_selection_review(plan, {"items": [{"selection_id": "1",
        "verdict": "corrected", "pixel": [300, 120], "reason": "Visible square center"}]}, image.size)
    assert plan["selections"][0]["pixel"] == [300, 300]
    measured = core.measure_plot(inspected["session_id"],
        {"anchors": [{"pixel": 50, "value": 0}, {"pixel": 550, "value": 10}]},
        reviewed["selections"],
        {"anchors": [{"pixel": 350, "value": 0}, {"pixel": 50, "value": 1}]})
    rows = browser._flatten_selected(measured, {"x_unit": "min"}, {}, "")
    browser._write_outputs(tmp_path, rows, {})
    exported = list(csv.DictReader((tmp_path / "table.csv").open(encoding="utf-8-sig")))[0]
    assert float(exported["x"]) == pytest.approx(5)
    assert float(exported["y"]) == pytest.approx(230/300)
    saved = json.loads((tmp_path / "result.json").read_text())["rows"][0]
    assert saved["pixel"] == [300, 120]
    assert saved["x_unit"] == "min"
    assert not warnings
    overlay = tmp_path / "overlay.png"
    selection_overlay(source, measured["points"], overlay)
    assert Image.open(overlay).size == image.size
    assert Image.open(overlay).getpixel((307, 120))[0] > 180


def test_coordinate_review_retains_uncertain_values_and_rejects_invalid_mapping():
    import pytest
    from plot_reader.browser_evidence import apply_selection_review

    plan = {"selections": [{"label": "A", "mode": "curve", "x_value": 5, "pixel": [50, 60]}]}
    review = {"items": [{"selection_id": "1", "verdict": "uncertain", "pixel": None, "reason": "Overlap"}]}
    updated, warnings = apply_selection_review(plan, review, (100, 100))
    assert updated["selections"][0]["pixel"] == [50, 60]
    assert updated["selections"][0]["coordinate_review"] == "uncertain"
    assert warnings == ["A: Overlap"]
    review["items"][0].update(verdict="corrected", pixel=[50, 40])
    updated, _ = apply_selection_review(plan, review, (100, 100))
    assert updated["selections"][0]["mode"] == "curve"
    assert updated["selections"][0]["x_value"] == 5
    review["items"][0]["pixel"] = [50, 140]
    with pytest.raises(ValueError, match="outside"):
        apply_selection_review(plan, review, (100, 100))
    with pytest.raises(ValueError, match="exactly once"):
        apply_selection_review(plan, {"items": []}, (100, 100))


def test_skipped_clarification_resumes_shared_measurement_pipeline(tmp_path, monkeypatch):
    import asyncio
    from starlette.applications import Starlette

    monkeypatch.setattr(browser, "DATA_ROOT", tmp_path)
    app = Starlette()
    app.state.runners, app.state.tasks, app.state.subscribers = {}, {}, {}
    job_id = "a" * 32
    folder = tmp_path / "jobs" / job_id
    browser._json(folder / "job.json", {"id": job_id, "plan": {"selections": [], "clarifications": []}})
    browser._json(folder / "request.json", {"normalization": True})
    browser._json(folder / "inspection.json", {})
    resumed = []

    async def shared_pipeline(app, ident, plan):
        resumed.append((ident, plan))

    monkeypatch.setattr(browser, "_run_job", shared_pipeline)
    asyncio.run(browser._resume_selected(app, job_id, {"id": "normalization-reference"}, "", True))
    assert resumed == [(job_id, {"selections": [], "clarifications": []})]
    assert not json.loads((folder / "request.json").read_text())["normalization"]


def test_focused_review_replaces_only_the_corresponding_source_selection(tmp_path):
    import asyncio

    source = tmp_path / "source.png"
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    for x in (50, 200):
        draw.ellipse((x-4, 96, x+4, 104), outline="black", width=2)
    image.save(source)
    plan = {"selections": [
        {"label": "A", "pixel": [100, 100], "status": "selected", "coordinate_review": "uncertain"},
        {"label": "B", "pixel": [200, 100], "status": "selected", "coordinate_review": "confirmed"}]}

    class Runner:
        calls = 0

        async def emit(self, event):
            pass

        async def call(self, **kwargs):
            self.calls += 1
            return {"items": [{"selection_id": "1", "verdict": "corrected",
                               "candidate_id": "C1", "pixel": [9999, 9999], "reason": "Visible marker center C1"}]}

    runner = Runner()
    updated = asyncio.run(browser._refine_unsupported_centers(runner, plan, source, tmp_path))
    assert runner.calls == 1
    assert updated["selections"][0]["pixel"] == [50, 100]
    assert updated["selections"][1]["pixel"] == [200, 100]
    assert json.loads((tmp_path / "pixel-support-final.json").read_text()) == []


def test_browser_retains_failed_axis_checks_as_warnings_and_uncertainty(tmp_path, monkeypatch):
    import pytest
    from plot_reader import core
    from plot_reader.browser_measurement import measure_selected

    monkeypatch.setattr(core, "SESSION_ROOT", tmp_path / "sessions")
    source = tmp_path / "source.png"
    Image.new("RGB", (600, 400), "white").save(source)
    inspected = core.inspect_plot(str(source))
    x_axis = {"anchors": [{"pixel": 50, "value": 0}, {"pixel": 550, "value": 10}],
              "checks": [{"pixel": 300, "value": 5.2}]}
    y_axis = {"anchors": [{"pixel": 350, "value": 0}, {"pixel": 50, "value": 1}],
              "checks": [{"pixel": 200, "value": 0.6}]}
    selections = [{"label": "A", "pixel": [300, 120]}]
    with pytest.raises(ValueError, match="check tick"):
        core.measure_plot(inspected["session_id"], x_axis, selections, y_axis)
    measured = measure_selected(inspected["session_id"], x_axis, selections, y_axis)
    assert len(measured["calibration_warnings"]) == 2
    assert measured["x_axis"]["checks"] == x_axis["checks"]
    assert measured["points"][0]["x"] == pytest.approx(5)
    assert measured["points"][0]["y"] == pytest.approx(230/300)
    assert measured["points"][0]["x_uncertainty_transformed"] > 0.2
    assert measured["points"][0]["y_uncertainty_transformed"] > 0.1


def test_local_wizard_upload_preview_and_confirm_target(tmp_path, monkeypatch):
    monkeypatch.setattr(browser, "DATA_ROOT", tmp_path / "browser-data")
    monkeypatch.setattr(browser.app.state, "token", "test-local-token")
    async def skip_measurement(app, job_id):
        pass
    monkeypatch.setattr(browser, "_run_job", skip_measurement)
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.line((45, 240, 350, 65), fill="black", width=3)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    with TestClient(browser.app, base_url="http://127.0.0.1") as client:
        assert client.get("/?token=test-local-token").status_code == 200
        assert client.get("/").status_code == 403
        headers = {"X-Plot-Reader-Token": "test-local-token"}
        uploaded = client.post("/api/upload", data={"explanation": "red scatter series"},
                               files={"file": ("plot.png", encoded.getvalue(), "image/png")},
                               headers=headers)
        assert uploaded.status_code == 200, uploaded.text
        source = uploaded.json()
        assert source["image_size"] == [400, 300]
        assert client.get(f"/api/source/{source['id']}/preview?token=test-local-token").status_code == 200
        confirmed = client.post("/api/target", json={"upload_id": source["id"],
                                  "crop": [0, 0, 400, 300]}, headers=headers)
        assert confirmed.status_code == 200, confirmed.text
        accepted = client.post("/api/jobs", json={"upload_id": source["id"],
                                 "category": "plot", "mode": "selected", "description": "Scatter plot"}, headers=headers)
        assert accepted.status_code == 202, accepted.text
        saved = browser._read(browser._job_dir(accepted.json()["id"]) / "request.json")
        assert saved["targets"] == "All data points"
        assert browser._is_bulk_point_request(saved)


def test_pdf_text_target_ranking_prefers_figure_number_and_terms():
    pages = ["Methods and sample preparation", "Figure 2. 1H NMR spectrum of compound A",
             "Figure 3. UV spectrum of compound B"]
    ranked = browser._match_pdf_pages(pages, "Figure 2 NMR compound A")
    assert ranked[0]["page"] == 2
    assert ranked[0]["score"] > ranked[-1]["score"]


def test_complete_plan_schema_is_strict_and_accepts_unscaled_axes():
    from jsonschema import Draft202012Validator

    schema = browser._complete_schema()
    Draft202012Validator.check_schema(schema)
    plan = {"inventory_complete": True, "unresolved_inventory": [], "panels": [
        {"id": "P1", "bounds": [10, 10, 290, 190], "x_axis": None, "y_axis": None,
         "series": [{"id": "S1", "kind": "curve", "description": "black trace",
                     "requested_target_ids": [], "bounds": [10, 10, 290, 190],
                     "seeds": [[80, 100]], "color_rgb": None,
                     "exclude_bounds": [], "baseline": None}]}]}
    Draft202012Validator(schema).validate(plan)
    plan["unexpected"] = True
    assert list(Draft202012Validator(schema).iter_errors(plan))


def test_luna_schemas_are_strict_for_codex_response_format():
    from jsonschema import Draft202012Validator
    from plot_reader.luna_runtime import COMMENT_SCHEMA, SELECTED_SCHEMA, SELECTION_REVIEW_SCHEMA, FOCUSED_REVIEW_SCHEMA

    def audit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert set(node.get("properties", {})) == set(node.get("required", []))
            for value in node.values():
                audit(value)
        elif isinstance(node, list):
            for value in node:
                audit(value)

    for schema in (COMMENT_SCHEMA, SELECTED_SCHEMA, SELECTION_REVIEW_SCHEMA, FOCUSED_REVIEW_SCHEMA, browser._complete_schema()):
        Draft202012Validator.check_schema(schema)
        audit(schema)


def test_complete_rows_use_calibrated_uncertainty_not_pixels(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"panels": [{"id": "P1", "series": [{"id": "S1",
        "description": "red trace", "features": [{"id": "F1", "status": "resolved",
        "calibrated_x": 2.0, "calibrated_y": 3.0, "normalized_height": 1.0,
        "pixel_uncertainty": 1.5, "calibrated_uncertainty": {
            "x_interval": [1.9, 2.1], "y_interval": [2.8, 3.2]},
        "original_pixel_xy": [100, 200]}], "queries": [], "curve": []}]}]}))

    row = browser._complete_rows(path)[0]
    assert abs(row["x_uncertainty"] - 0.1) < 1e-9
    assert abs(row["y_uncertainty"] - 0.2) < 1e-9
    assert row["height_relative"] is None


def test_csv_export_uses_simplified_report_columns_and_retains_json_evidence(tmp_path):
    fields = ["panel", "series", "feature", "status", "x", "y", "height_relative",
              "reason", "method", "review_note"]
    rows = [{key: None for key in fields}]
    rows[0].update(panel="P1", feature="A", status="resolved", x=2.5, pixel=[20, 30],
                   x_uncertainty=.1, x_unit="min", review_id="R1", reason="Overlap,\nneeds review")
    browser._write_outputs(tmp_path, rows, {})
    with (tmp_path / "table.csv").open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.reader(stream))
    assert exported[0] == fields
    assert exported[1][fields.index("x")] == "2.5"
    assert exported[1][fields.index("reason")] == "Overlap,\nneeds review"
    assert json.loads((tmp_path / "result.json").read_text())["rows"] == rows


def test_saved_run_csv_download_uses_current_report_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(browser, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(browser.app.state, "token", "test-local-token")
    job_id = "c" * 32
    folder = browser._job_dir(job_id)
    row = {"panel":"P1", "feature":"A", "x":2.5, "pixel":[20,30], "x_uncertainty":.1,
           "x_unit":"min", "review_id":"R1", "reason":"Note", "method":"exact", "review_note":"Reviewed"}
    browser._json(folder / "result.json", {"rows":[row]})
    old_csv = "x,pixel,x_uncertainty,x_unit,review_id\n2.5,old,.1,min,R1\n"
    (folder / "table.csv").write_text(old_csv)
    with TestClient(browser.app, base_url="http://127.0.0.1") as client:
        url = f"/api/jobs/{job_id}/artifact/table.csv"
        assert client.get(url).status_code == 403
        response = client.get(url, headers={"X-Plot-Reader-Token":"test-local-token"})
        assert response.status_code == 200
        reader = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
        assert reader.fieldnames == ["panel","series","feature","status","x","y","height_relative","reason","method","review_note"]
        assert next(reader)["review_note"] == "Reviewed"
        evidence = client.get(f"/api/jobs/{job_id}/artifact/result.json", headers={"X-Plot-Reader-Token":"test-local-token"}).json()
        assert evidence["rows"] == [row]
    assert (folder / "table.csv").read_text() == old_csv
