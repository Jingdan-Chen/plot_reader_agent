"""Browser warning policy around the unchanged frozen measurement engine."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

from . import core


def measure_selected(session_id: str, x_axis: dict[str, Any], selections: list[dict[str, Any]],
                     y_axis: dict[str, Any] | None = None, reference_label: str | None = None,
                     baseline_pixel: float | None = None, use_label_proposals: bool = False) -> dict[str, Any]:
    maps = {name: core.Axis(axis["anchors"], axis.get("scale", "linear"), axis.get("checks", []))
            if axis else None for name, axis in (("x", x_axis), ("y", y_axis))}
    residuals = {name: axis.residual() if axis else None for name, axis in maps.items()}
    warnings = []
    for name, axis in maps.items():
        residual = residuals[name]
        if axis and residual is not None and residual > axis.transformed_error(4):
            pixels = residual / axis.transformed_error(1)
            warnings.append(f"{name}-axis check tick differs from the calibrated axis by about {pixels:.1f} pixels (4-pixel tolerance)")
    # The engine still performs its normal anchor/feature/normalization checks.
    # Check-tick policy and residual uncertainty belong to this browser adapter.
    engine_axes = {name: {k:v for k,v in value.items() if k in {'anchors','scale'}} if value else None
                   for name,value in (('x',x_axis),('y',y_axis))}
    result = core.measure_plot(session_id, engine_axes['x'], selections,
        engine_axes['y'], reference_label, baseline_pixel, use_label_proposals)
    for name, axis in maps.items():
        result[f"{name}_axis"] = asdict(axis) if axis else None
        original = x_axis if name == 'x' else y_axis
        if axis and original and 'unit' in original:
            result[f"{name}_axis"]['unit'] = original['unit']
    result["calibration_residual_transformed"] = residuals
    result["calibration_warnings"] = warnings
    result["browser_calibration_policy"] = "Independent checks retained as warnings; residuals included in uncertainty."
    selections_by_label = {s['label']:s for s in selections}
    for point in result["points"]:
        grounding=selections_by_label.get(point['label'],{}).get('grounding')
        if grounding and point.get('pixel'):
            # The bounded local review can adjust a corner/partial contour.
            # Do not report the frozen exact-mode precision for that process.
            point['grounding']=grounding
            point['method']='source_candidate_review'
            point['pixel_uncertainty']=max(point.get('pixel_uncertainty') or 0,grounding.get('offset_limit',3))
            for name,axis in maps.items():
                if axis: point[f'{name}_uncertainty_transformed']=axis.transformed_error(point['pixel_uncertainty']+1)
        for name in ("x", "y"):
            key = f"{name}_uncertainty_transformed"
            if point.get(key) is not None and residuals[name] is not None:
                point[key] = math.hypot(point[key], residuals[name])
    return result
