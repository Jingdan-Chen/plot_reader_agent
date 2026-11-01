"""Replay a saved browser job without modifying it or changing the portal.

Run: .venv/bin/python -m experiments.scatter_templates --help
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image

from plot_reader.scatter_templates import (
    component_atlas, legend_components, save_evidence, scan_templates, validate_profile,
)


async def discover_profile(source, atlas, components, series, output):
    # Kept optional: deterministic replay needs no model or Codex runtime.
    from plot_reader.luna_runtime import LunaRunner, _json_schema

    async def emit(event):
        with (output/"luna-events.jsonl").open("a") as stream:
            stream.write(json.dumps(event)+"\n")
        if event.get("type") in {"luna_started", "luna_completed", "luna_diagnostic"}:
            print(json.dumps(event), flush=True)

    schema = _json_schema({
        "applicable": {"type": "boolean"}, "reason": {"type": "string"},
        "templates": {"type": "array", "items": _json_schema({
            "series": {"type": "string", "enum": series},
            "component_id": {"type": "string", "enum": [c["id"] for c in components]},
            "shape": {"type": "string"}, "reason": {"type": "string"},
        })},
    })
    prompt = f"""Assess whether this scatter plot uses repeated symbols of approximately
constant shape and size within each series. Source first, enlarged catalogue of
connected legend components second. This is TEMPLATE SELECTION, not measurement.
For each series choose exactly one catalogue ID showing its clean complete symbol.
Do not choose text, part of a glyph, or a symbol connected to a long sample line.
Do not infer identities from curve order alone. Compare the source legend and the
actual plotted markers. The host will extract the original grayscale pixels and
scan a narrow size bank using normalized cross-correlation. Never return coordinates
or numerical data values. If markers vary too much, symbols are inseparable, glyphs
are incomplete, or no suitable template exists for a series, set applicable=false
and explain; do not fabricate a kernel. Identical patterns do not ensure that
overlapping markers will be recoverable. In applicable=true return every series
exactly once, with distinct IDs. Series: {json.dumps(series)}.
Catalogue source bounds (native pixels): {json.dumps(components)}"""
    (output/"profile-prompt.txt").write_text(prompt)
    runner = LunaRunner(emit, timeout=240)
    return await runner.call(prompt=prompt, schema=schema, images=[str(source), str(atlas)],
                             workdir=str(output/"worker"))


def validate_review(review, candidates):
    expected = {c["id"] for c in candidates}
    ids = [p["candidate_id"] for p in review["items"]]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Review must account for every candidate exactly once")
    if any(p["verdict"] not in {"supported", "rejected", "uncertain"} for p in review["items"]):
        raise ValueError("Invalid review verdict")


async def review_evidence(source, candidates, paths, output):
    """Optional semantic review consumes heatmaps without authoring coordinates."""
    from plot_reader.luna_runtime import LunaRunner, _json_schema

    if not candidates:
        review = {"items": [], "notes": ["No response peaks passed the detector gates."]}
    else:
        async def emit(event):
            with (output/"review-events.jsonl").open("a") as stream:
                stream.write(json.dumps(event)+"\n")

        schema = _json_schema({
            "items": {"type": "array", "items": _json_schema({
                "candidate_id": {"type": "string", "enum": [c["id"] for c in candidates]},
                "verdict": {"type": "string", "enum": ["supported", "rejected", "uncertain"]},
                "reason": {"type": "string"},
            })}, "notes": {"type": "array", "items": {"type": "string"}},
        })
        prompt = f"""Review experimental scatter-marker candidates against the original source.
The first image is unannotated source. Remaining images pair source/candidate IDs
on the left with per-series normalized-correlation maps on the right. White rings
only highlight peaks. Brightness is template agreement, NOT probability or proof.
For EVERY candidate ID return supported, rejected, or uncertain with a short visual
reason. Supported requires the visible experimental glyph to match that series AND
the candidate ring to be centered on it. Reject fitted-line fragments, other symbol
types, duplicates, and offset centers. Use uncertain for inseparable overlaps or
inadequate resolution. Compare source legend identities; do not infer identity from
curve position alone. Treat conflicting candidates as alternatives; do not support
multiple identities merely because several templates respond. Do not move centers,
add IDs, infer missing points or provide numeric measurements. Mention missed visible
markers in notes. This is a prototype audit, not a complete dataset.
Candidates: {json.dumps(candidates, ensure_ascii=False)}"""
        (output/"review-prompt.txt").write_text(prompt)
        review = await LunaRunner(emit, timeout=300).call(
            prompt=prompt, schema=schema, images=[str(source), *paths], workdir=str(output/"review-worker"))
    (output/"model-review.raw.json").write_text(json.dumps(review, indent=2, ensure_ascii=False))
    validate_review(review, candidates)
    (output/"model-review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False))
    return review


def run(args):
    job, output = Path(args.job_dir).resolve(), Path(args.output).resolve()
    if output == job or job in output.parents:
        raise ValueError("Experiment output must be outside the saved job")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new empty output folder to preserve previous trials")
    source = job/"source-image.png"
    layout = json.loads((job/"marker-layout.json").read_text())
    plan = json.loads((job/"plan.json").read_text())
    series = list(dict.fromkeys(p.get("series", "Series") for p in plan["selections"]))
    components = legend_components(source, layout.get("exclude_regions", []))
    if not components:
        raise ValueError("No isolated legend glyphs: use a different template acquisition method")
    atlas = output/"legend-catalogue.png"
    component_atlas(source, components, atlas)
    (output/"legend-components.json").write_text(json.dumps(components, indent=2))
    profile = (asyncio.run(discover_profile(source, atlas, components, series, output))
               if args.luna else json.loads(Path(args.profile).read_text()))
    (output/"profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    validate_profile(profile, components, series)
    candidates, maps = scan_templates(source, components, profile, layout,
                                      threshold=args.threshold, line_margin=args.line_margin, scales=args.scales)
    paths = save_evidence(source, candidates, maps, output)
    review = asyncio.run(review_evidence(source, candidates, paths, output)) if args.review else None
    # Render original selected kernels so the source of every response is auditable.
    by_id = {c["id"]: c for c in components}
    im = Image.open(source)
    for i, template in enumerate(profile["templates"]):
        im.crop(by_id[template["component_id"]]["bounds"]).save(output/f"template-{i}.png")
    old_file = job/"marker-grounding.json"
    old_candidates = json.loads(old_file.read_text())["candidates"] if old_file.exists() else []
    l, t, r, b = layout["plot_bounds"]
    old_in_plot = [p for p in old_candidates if l <= p["pixel"][0] <= r and t <= p["pixel"][1] <= b
                   and not any(a <= p["pixel"][0] <= c and e <= p["pixel"][1] <= d
                               for a, e, c, d in layout.get("exclude_regions", []))]
    report = {
        "experimental": True, "source_job": str(job),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "profile_source": "gpt-6-luna" if args.luna else str(Path(args.profile).resolve()),
        "threshold": args.threshold, "line_margin": args.line_margin, "scales": args.scales,
        "baseline_candidates_all": len(old_candidates), "baseline_candidates_in_plot": len(old_in_plot),
        "template_candidates": len(candidates), "per_series": dict(Counter(p["series"] for p in candidates)),
        "ambiguous_or_unstable": sum(p["status"] == "needs_review" for p in candidates),
        "response_images": paths,
        "model_review_counts": dict(Counter(p["verdict"] for p in review["items"])) if review else None,
        "limitations": [
            "Candidate counts are not accuracy measurements; this source has no independently labeled truth set.",
            "Candidates and optional model review are preliminary evidence, not final extracted data.",
            "NCC score is not a probability. Template bounding-box center is the registration convention.",
            "No claim of subpixel accuracy: interpolation can be biased by scan blur and overlapping ink.",
            "Dark grayscale symbols only; isolated legend exemplars required in this prototype.",
            "Boundary, close same-series markers, line crossings and complete occlusions remain limitations.",
        ],
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", help="Reuse a saved source-ID template profile without a model call")
    group.add_argument("--luna", action="store_true", help="Ask the existing LUNA runtime to assess and select glyphs")
    parser.add_argument("--threshold", type=float, default=.57)
    parser.add_argument("--line-margin", type=float, default=.06)
    parser.add_argument("--scales", type=float, nargs="+", default=[.85, 1., 1.15],
                        help="Template size ratios relative to the source legend glyph")
    parser.add_argument("--review", action="store_true", help="Ask LUNA to audit source and response maps; never moves centers")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or not 0 <= args.line_margin <= 1:
        parser.error("threshold must be in (0,1], line margin in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
