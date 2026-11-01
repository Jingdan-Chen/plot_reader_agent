import asyncio
import json

import pytest
from starlette.applications import Starlette

from plot_reader import browser
from plot_reader.timing import AnalysisTiming, current_timing, run_local, timing_span


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_totals_partition_wall_time_and_stages_include_nested_work(tmp_path):
    clock = Clock()
    timing = AnalysisTiming(tmp_path, clock=clock)
    timing.set_stage("inspect")
    clock.advance(1)
    with timing.span("model_wait", "selected-plan"):
        clock.advance(4)
    with timing.span("local_python", "inspect_plot"):
        clock.advance(2)
    timing.set_stage("review")
    clock.advance(1)
    result = timing.finish()
    assert result["summary"] == {"total_seconds": 8, "model_wait_seconds": 4,
        "local_python_seconds": 2, "user_wait_seconds": 0, "host_overhead_seconds": 2}
    assert sum(row["elapsed_seconds"] for row in result["stages"]) == 8
    assert next(s for s in result["stages"] if s["name"] == "inspect")["elapsed_seconds"] == 7
    assert json.loads((tmp_path / "timing.json").read_text()) == result


def test_clarification_wait_and_prior_calls_survive_resuming(tmp_path):
    clock = Clock()
    timing = AnalysisTiming(tmp_path, clock=clock)
    timing.set_stage("inspect")
    with timing.span("model_wait", "selected-plan"):
        clock.advance(3)
    timing.set_stage("user_wait")
    clock.advance(20)
    resumed = AnalysisTiming(tmp_path, clock=clock)
    resumed.set_stage("clarification")
    with resumed.span("model_wait", "clarification-1"):
        clock.advance(2)
    result = resumed.finish()
    assert result["summary"]["total_seconds"] == 25
    assert result["summary"]["user_wait_seconds"] == 20
    assert result["summary"]["model_wait_seconds"] == 5
    assert len(result["model_calls"]) == 2
    clock.advance(10)
    assert resumed.finish() == result


def test_overlapping_calls_are_counted_once_in_summary(tmp_path):
    clock = Clock()
    timing = AnalysisTiming(tmp_path, clock=clock)
    with timing.span("local_python", "outer"):
        clock.advance(1)
        with timing.span("local_python", "inner"):
            clock.advance(2)
        clock.advance(1)
    result = timing.finish()
    assert result["summary"]["local_python_seconds"] == 4
    assert sum(c["elapsed_seconds"] for c in result["local_python_calls"]) == 6
    assert result["summary"]["total_seconds"] == 4


def test_failed_and_cancelled_response_waits_are_recorded(tmp_path):
    clock = Clock()
    timing = AnalysisTiming(tmp_path, clock=clock)
    with pytest.raises(ValueError):
        with timing.span("model_wait", "invalid-response"):
            clock.advance(2)
            raise ValueError("invalid")
    with pytest.raises(asyncio.CancelledError):
        with timing.span("model_wait", "cancelled-response"):
            clock.advance(1)
            raise asyncio.CancelledError()
    result = timing.finish()
    assert result["summary"]["model_wait_seconds"] == 3
    assert [c["status"] for c in result["model_calls"]] == ["failed", "cancelled"]


def test_worker_inherits_job_timing_and_context_is_isolated(tmp_path):
    clock = Clock()
    timing = AnalysisTiming(tmp_path, clock=clock)
    def measure():
        clock.advance(2)
        return 42
    async def work():
        token = current_timing.set(timing)
        try:
            assert await run_local(measure) == 42
        finally:
            current_timing.reset(token)
        with timing_span("model_wait", "unrelated"):
            clock.advance(1)
    asyncio.run(work())
    result = timing.finish()
    assert result["summary"]["local_python_seconds"] == 2
    assert result["summary"]["model_wait_seconds"] == 0
    assert len(result["local_python_calls"]) == 1


def test_result_export_has_current_timing_instead_of_parent_timing(tmp_path):
    timing = AnalysisTiming(tmp_path)
    token = current_timing.set(timing)
    try:
        timing.set_stage("correct_points")
        result = {"timing": {"summary": {"total_seconds": 9999}}}
        browser._write_outputs(tmp_path, [], result)
    finally:
        current_timing.reset(token)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["timing"] == json.loads((tmp_path / "timing.json").read_text())
    assert saved["timing"]["summary"]["total_seconds"] < 9999
    assert saved["timing"]["summary"]["model_wait_seconds"] == 0


def test_job_scope_resumes_without_losing_wait_or_finishing_nested_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(browser, "DATA_ROOT", tmp_path)
    job_id = "b" * 32
    folder = browser._job_dir(job_id)
    browser._json(folder / "job.json", {"status": "running"})
    app = Starlette()
    calls = []
    @browser._timed_job
    async def pipeline(app, job_id):
        with timing_span("model_wait", "selected-plan"):
            calls.append("plan")
        await browser._job_update(app, job_id, status="needs_input", stage="clarification")
    asyncio.run(pipeline(app, job_id))
    assert not json.loads((folder / "_timing-state.json").read_text())["finished"]
    @browser._timed_job
    async def nested(app, job_id):
        assert not current_timing.get().state["finished"]
        browser._write_outputs(folder, [], {})
        await browser._job_update(app, job_id, status="complete", stage="complete")
    @browser._timed_job
    async def resume(app, job_id):
        await browser._job_update(app, job_id, status="running", stage="clarification")
        with timing_span("model_wait", "clarification-1"):
            calls.append("answer")
        await nested(app, job_id)
    asyncio.run(resume(app, job_id))
    result = json.loads((folder / "result.json").read_text())
    assert len(result["timing"]["model_calls"]) == 2
    assert result["timing"]["summary"]["user_wait_seconds"] > 0
    assert current_timing.get() is None
