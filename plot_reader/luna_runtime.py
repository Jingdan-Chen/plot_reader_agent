"""Narrow, structured LUNA calls for the local Plot Reader web interface.

The Codex worker receives only attached images and its prompt. All image
analysis, validation, measurements, and file writes stay in the host process.
"""
from __future__ import annotations

import asyncio
import base64
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any, Awaitable, Callable
from PIL import Image
from jsonschema import Draft202012Validator, ValidationError
from .timing import run_local, timing_span


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "view_image", "multi_agent", "apps", "plugins",
    "remote_plugin", "browser_use", "computer_use",
    "image_generation", "workspace_dependencies", "skill_search", "tool_suggest",
)


def _json_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": required or list(properties), "additionalProperties": False}


AXIS = {"anyOf": [
    _json_schema({"anchors": {"type": "array", "items": _json_schema({
        "pixel": {"type": "number"}, "value": {"type": "number"}}), "minItems": 2},
        "checks": {"type": "array", "items": _json_schema({
            "pixel": {"type": "number"}, "value": {"type": "number"}}), "minItems": 1},
        "scale": {"type": "string", "enum": ["linear", "log10"]},
        "unit": {"type": ["string", "null"]}}, ["anchors", "checks", "scale", "unit"]),
    {"type": "null"}]}

SELECTION = _json_schema({
    "label": {"type": "string"},
    "series": {"type": "string"},
    "mode": {"type": "string", "enum": ["exact", "peak", "scatter", "curve"]},
    "candidate_id": {"type": ["string", "null"]},
    "pixel": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, {"type": "null"}]},
    "x_value": {"type": ["number", "null"]},
    "review_id": {"type": ["string", "null"]},
    "review_note": {"type": ["string", "null"]},
    "status": {"type": "string", "enum": ["selected", "unresolved"]},
    "reason": {"type": ["string", "null"]},
}, ["label", "series", "mode", "candidate_id", "pixel", "x_value", "review_id", "review_note", "status", "reason"])

SELECTED_SCHEMA = _json_schema({
    "x_axis": AXIS, "y_axis": AXIS,
    "selections": {"type": "array", "items": SELECTION},
    "reference_label": {"type": ["string", "null"]},
    "baseline_pixel": {"type": ["number", "null"]},
    "clarifications": {"type": "array", "items": _json_schema({
        "id": {"type": "string"}, "question": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "string"}},
        "region": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}, {"type": "null"}]},
    }, ["id", "question", "choices", "region"])},
    "notes": {"type": "array", "items": {"type": "string"}},
})

COMMENT_SCHEMA = _json_schema({
    "comment": {"type": "string"},
    "review_issues": {"type": "array", "items": {"type": "string"}},
})

SELECTION_REVIEW_SCHEMA = _json_schema({
    "items": {"type": "array", "items": _json_schema({
        "selection_id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["confirmed", "corrected", "uncertain"]},
        "pixel": {"anyOf": [{"type": "array", "items": {"type": "number"},
                               "minItems": 2, "maxItems": 2}, {"type": "null"}]},
        "reason": {"type": "string"},
    })},
})

FOCUSED_REVIEW_SCHEMA = _json_schema({"items": {"type": "array", "items": _json_schema({
    **SELECTION_REVIEW_SCHEMA["properties"]["items"]["items"]["properties"],
    "candidate_id": {"type": ["string", "null"]},
})}})


class LunaError(RuntimeError):
    pass


EFFORT_CHOICES = {
    'codex': {'light': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'xhigh'},
    'claude': {'low': 'low', 'medium': 'medium', 'high': 'high', 'extra': 'xhigh'},
}


def assistant_profile(effort: str | None = None) -> dict[str, str]:
    provider=os.environ.get('PLOT_READER_AGENT','codex').strip().lower()
    if provider not in EFFORT_CHOICES:
        raise LunaError('PLOT_READER_AGENT must be codex or claude; no model fallback was used.')
    choice = 'medium' if effort is None else effort
    if not isinstance(choice, str) or choice not in EFFORT_CHOICES[provider]:
        raise LunaError(f"Choose a valid {provider} effort: {', '.join(EFFORT_CHOICES[provider])}.")
    actual = EFFORT_CHOICES[provider][choice]
    if provider=='codex':
        return {'provider':provider,'name':'LUNA','model':'gpt-6-luna','effort':actual,
                'effort_choice':choice,'speed':'priority','label':f'LUNA 6 · {choice} · Fast'}
    return {'provider':provider,'name':'Haiku','model':'claude-haiku-5-5','effort':actual,
            'effort_choice':choice,'speed':'standard','label':f'Haiku 5.5 · {choice}'}


@lru_cache(maxsize=4)
def check_claude_cli(executable: str) -> None:
    result=subprocess.run([executable,'--help'],capture_output=True,text=True,timeout=30)
    required=('--effort','--json-schema','--tools','--restricted','--no-session-persistence')
    if result.returncode or any(flag not in result.stdout for flag in required):
        raise LunaError('Claude Code is too old for this workflow. Update Claude Code from its official installer, sign in, and retry. Haiku 5.5 requires --effort, --json-schema, --tools, --restricted and --no-session-persistence. No alternate model was selected.')


def action_label(workdir: str) -> str:
    name=Path(workdir).name
    labels={'marker-layout':'Identifying plot bounds and legend exclusions',
            'marker-region':'Identifying individual markers and their series',
            'decision-':'Deciding whether a scatter-template scan will help',
            'selected-plan':'Locating the requested peaks and points',
            'selected-repair':'Refining the requested point inventory',
            'coordinate-refinement':'Refining uncertain feature centers',
            'coordinate-review':'Checking selected pixels against the source',
            'complete-plan':'Planning all visible panels and curves',
            'complete-repair':'Correcting the curve plan from local validation',
            'clarification-':'Applying your clarification to the measurement plan',
            'explanation':'Reviewing measured results and writing the explanation'}
    return next((label for prefix,label in labels.items() if name.startswith(prefix)),
                'Reviewing the supplied plot evidence')


def claude_input(prompt: str, images: list[str]) -> bytes:
    content=[{'type':'text','text':prompt}]
    for path in images:
        with Image.open(path) as source:
            image=source.convert('RGB');buffer=io.BytesIO();image.save(buffer,format='PNG')
        content.append({'type':'image','source':{'type':'base64','media_type':'image/png',
            'data':base64.b64encode(buffer.getvalue()).decode('ascii')}})
    return (json.dumps({'type':'user','message':{'role':'user','content':content}})+'\n').encode()


class LunaRunner:
    """Runs bounded, schema-constrained Codex CLI calls with saved login."""

    def __init__(self, emit: EventCallback, timeout: int = 900, *, effort: str | None = None):
        self.emit = emit
        self.timeout = timeout
        self.profile = assistant_profile(effort)
        self.process: asyncio.subprocess.Process | None = None
        self.last_error: str | None = None
        self.structured_output: dict | None = None

    async def cancel(self) -> None:
        proc = self.process
        if proc and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()

    async def call(self, *, prompt: str, schema: dict[str, Any],
                   images: list[str], workdir: str) -> dict[str, Any]:
        profile=self.profile
        executable=shutil.which(profile['provider'])
        if not executable:
            raise LunaError(f"{profile['provider']} CLI is unavailable on PATH")
        if profile['provider']=='claude':
            await run_local(check_claude_cli,executable)
        worker = Path(workdir).resolve()
        worker.mkdir(parents=True, exist_ok=True)
        schema_path = worker / "response.schema.json"
        response_path = worker / "response.json"
        self.last_error = None
        self.structured_output = None
        response_path.unlink(missing_ok=True)
        schema_path.write_text(json.dumps(schema, allow_nan=False))
        args = [executable, "--strict-config", "-a", "never", "exec", "--json",
                "--ignore-user-config", "--skip-git-repo-check", "-m", profile['model'],
                "-c", f"model_reasoning_effort={profile['effort']}", "-c", "service_tier=priority",
                "-s", "read-only", "-C", str(worker)]
        for feature in DISABLED_FEATURES:
            args.extend(["--disable", feature])
        args.extend(["-c", "features.skip_host_skill_discovery=true",
                     "-c", 'web_search="disabled"'])
        for image in images:
            args.extend(["-i", str(Path(image).resolve())])
        args.extend(["--output-schema", str(schema_path), "-o", str(response_path), "-"])
        payload=prompt.encode('utf-8')
        if profile['provider']=='claude':
            args=[executable,'-p','--verbose','--input-format','stream-json',
                '--output-format','stream-json','--model',profile['model'],'--effort',profile['effort'],
                '--json-schema',json.dumps(schema),'--tools','','--restricted',
                '--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
                '--setting-sources','','--no-session-persistence']
            payload=await run_local(claude_input,prompt,images)
        action=action_label(workdir)
        await self.emit({"type": "luna_started",'message':action,'provider':profile['provider']})
        (worker/'inference.json').write_text(json.dumps(profile,indent=2)+'\n')
        try:
            with timing_span("model_wait", worker.name, provider=profile["provider"], model=profile["model"],
                             effort=profile["effort"], effort_choice=profile["effort_choice"]):
                self.process = await asyncio.create_subprocess_exec(
                    *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, cwd=worker,
                    env={k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "CODEX_API_KEY", "CLAUDECODE"}},
                    start_new_session=True)
                stdout_task = asyncio.create_task(self._read_events(self.process.stdout, "stdout"))
                stderr_task = asyncio.create_task(self._read_events(self.process.stderr, "stderr"))
                assert self.process.stdin is not None
                self.process.stdin.write(payload)
                await self.process.stdin.drain()
                self.process.stdin.close()
                try:
                    code = await asyncio.wait_for(self.process.wait(), self.timeout)
                except asyncio.TimeoutError as exc:
                    await self.cancel()
                    raise LunaError("LUNA response timed out") from exc
                await asyncio.gather(stdout_task, stderr_task)
                if code != 0:
                    detail = f": {self.last_error}" if self.last_error else "; see run events"
                    raise LunaError(f"{profile['name']} exited with status {code}{detail}")
                if profile['provider']=='claude' and self.structured_output is not None:
                    response_path.write_text(json.dumps(self.structured_output,allow_nan=False))
                if not response_path.is_file():
                    raise LunaError("LUNA did not return the required structured response")
                try:
                    result = json.loads(response_path.read_text())
                    Draft202012Validator(schema).validate(result)
                except ValidationError as exc:
                    raise LunaError(f"{profile['name']} returned a response that violates the required schema: {exc.message[:200]}") from None
                except (OSError, json.JSONDecodeError) as exc:
                    raise LunaError("LUNA returned malformed JSON") from exc
                await self.emit({"type": "luna_completed",'message':f'Finished: {action[0].lower()+action[1:]}.','provider':profile['provider']})
                return result
        finally:
            self.process = None

    async def _read_events(self, stream: asyncio.StreamReader | None, source: str) -> None:
        if stream is None:
            return
        while line := await stream.readline():
            raw = line.decode("utf-8", "replace").strip()
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                # Do not persist arbitrary CLI logs, model reasoning or credentials.
                continue
            if not isinstance(item,dict):continue
            if item.get('type')=='result':
                if item.get('is_error'):
                    self.last_error='The Claude call failed; check CLI sign-in, model access and account limits.'
                elif isinstance(item.get('structured_output'),dict):
                    self.structured_output=item['structured_output']
            # Do not expose arbitrary prompt/message text in the browser UI.
            event = {key: item[key] for key in ("type", "usage", "thread_id") if key in item}
            if item.get("type") in {"error", "turn.failed"}:
                detail = item.get("error", item)
                if isinstance(detail, dict):
                    detail = detail.get("message", "")
                if isinstance(detail, str) and detail:
                    self.last_error = detail[:500]
                    await self.emit({"type": "luna_diagnostic", "message": detail[:500]})
            if event:
                await self.emit({"type": "luna_event", **event})


def worker_dir(job_dir: str, stage: str) -> str:
    path = Path(job_dir) / "worker" / stage
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
