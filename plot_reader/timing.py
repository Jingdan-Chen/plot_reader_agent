"""Host-measured wall times for one browser analysis, including resumed work."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
import time


current_timing: ContextVar[AnalysisTiming | None] = ContextVar("analysis_timing", default=None)


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _duration(intervals):
    return sum(end - start for start, end in _union(intervals))


def _intersection(left, right):
    return _union((max(a, c), min(b, d)) for a, b in left for c, d in right)


def _subtract(left, right):
    pieces = _union(left)
    for start, end in _union(right):
        remaining = []
        for a, b in pieces:
            if b <= start or a >= end:
                remaining.append([a, b])
            else:
                if a < start:
                    remaining.append([a, start])
                if b > end:
                    remaining.append([end, b])
        pieces = remaining
    return pieces


class AnalysisTiming:
    def __init__(self, folder: Path, *, clock=time.perf_counter):
        self.folder = Path(folder)
        self.clock = clock
        self.lock = RLock()
        self.path = self.folder / "_timing-state.json"
        if self.path.is_file():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"origin": clock(), "started_at": datetime.now(timezone.utc).isoformat(),
                          "stage": "starting", "stage_start": 0.0, "stages": [], "calls": [],
                          "finished": False}

    def offset(self):
        return max(0.0, self.clock() - self.state["origin"])

    def save(self):
        with self.lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, allow_nan=False))
            temporary.replace(self.path)

    def set_stage(self, stage):
        with self.lock:
            if self.state["finished"] or stage == self.state["stage"]:
                return
            end = self.offset()
            self.state["stages"].append({"name": self.state["stage"],
                "start": self.state["stage_start"], "end": end})
            self.state.update(stage=stage, stage_start=end)
            self.save()

    @contextmanager
    def span(self, kind, name, **metadata):
        start, stage = self.offset(), self.state["stage"]
        status = "completed"
        try:
            yield
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            raise
        finally:
            end = self.offset()
            with self.lock:
                # A cancelled to_thread call can finish after its job has stopped.
                # Its uncompleted work is not charged to the finished analysis.
                if not self.state["finished"]:
                    self.state["calls"].append({"kind": kind, "name": name, "stage": stage,
                        "start": start, "end": end, "status": status, **metadata})
                    self.save()

    def finish(self):
        with self.lock:
            if not self.state["finished"]:
                end = self.offset()
                self.state["stages"].append({"name": self.state["stage"],
                    "start": self.state["stage_start"], "end": end})
                self.state.update(finished=True, total_seconds=end,
                    finished_at=datetime.now(timezone.utc).isoformat())
                self.save()
            report = self.report()
            (self.folder / "timing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            return report

    def report(self):
        total = self.state.get("total_seconds", self.offset())
        calls = self.state["calls"]
        full = [[0, total]]
        model = _intersection(full, _union((c["start"], c["end"]) for c in calls if c["kind"] == "model_wait"))
        local = _subtract(_intersection(full, _union((c["start"], c["end"]) for c in calls if c["kind"] == "local_python")), model)
        stages = list(self.state["stages"])
        if not self.state["finished"]:
            stages.append({"name": self.state["stage"], "start": self.state["stage_start"], "end": total})
        user = _subtract([(s["start"], s["end"]) for s in stages if s["name"] == "user_wait"], model + local)
        other = _subtract(full, model + local + user)
        stage_rows = []
        for name in dict.fromkeys(s["name"] for s in stages):
            intervals = [(s["start"], s["end"]) for s in stages if s["name"] == name]
            stage_rows.append({"name": name, "elapsed_seconds": _duration(intervals),
                "model_wait_seconds": _duration(_intersection(intervals, model)),
                "local_python_seconds": _duration(_intersection(intervals, local))})
        def records(kind):
            return [{**c, "elapsed_seconds": c["end"] - c["start"]}
                    for c in sorted(calls, key=lambda c: c["start"]) if c["kind"] == kind]
        return {"schema_version": 1, "clock": "perf_counter", "unit": "seconds",
            "started_at": self.state["started_at"], "finished_at": self.state.get("finished_at"),
            "summary": {"total_seconds": total, "model_wait_seconds": _duration(model),
                "local_python_seconds": _duration(local), "user_wait_seconds": _duration(user),
                "host_overhead_seconds": _duration(other)},
            "stages": stage_rows, "model_calls": records("model_wait"),
            "local_python_calls": records("local_python"),
            "notes": ["Total runs from analysis start through result preparation, including clarification waits; upload, PDF discovery, browser rendering and download time are excluded.",
                "Model wait is the CLI round trip to a validated response, including CLI startup, transfer, provider queueing and inference; these parts cannot be separated by the host.",
                "Local Python calls measure wall time inside each worker or Python subprocess, including any child tools such as Tesseract. Small host operations are included in host overhead.",
                "Summary categories partition total wall time. Stage times include their model and Python work; call rows are details, not extra time. Concurrent or nested calls can overlap."]}


@contextmanager
def timing_span(kind, name, **metadata):
    recorder = current_timing.get()
    if recorder is None:
        yield
    else:
        with recorder.span(kind, name, **metadata):
            yield


def local_work(function, *args, **kwargs):
    """Measure inside the worker; thread scheduling remains host overhead."""
    name = getattr(function, "__qualname__", type(function).__name__)
    with timing_span("local_python", name):
        return function(*args, **kwargs)


async def run_local(function, *args, **kwargs):
    return await asyncio.to_thread(local_work, function, *args, **kwargs)
