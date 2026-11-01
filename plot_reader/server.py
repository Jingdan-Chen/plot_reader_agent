from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from . import core

mcp = FastMCP("Plot Reader")


@mcp.tool(structured_output=False)
def inspect_plot(image_path: str, measurement: str = "requested points", requested_labels: list[str] | None = None) -> list[Any]:
    """Inspect once. Pass printed requested_labels for local OCR and readable review panels. Returns implementation_id, original-pixel ticks/candidates, label_assignments and review_regions. Prominence is NOT assignment confidence. For needs_review labels, inspect the local image and supply review_id and review_note with the measurement selection."""
    result = core.inspect_plot(image_path, measurement, requested_labels)
    return [json.dumps(result)] + [Image(path=p) for p in result["review_image_paths"]]


@mcp.tool(structured_output=False)
def refine_region(session_id: str, bounds: list[int]) -> list[Any]:
    """Magnify an ambiguous region. bounds=[left,top,right,bottom] in original image pixels; response retains the crop-to-original scale and nearby candidate IDs."""
    result = core.refine_region(session_id, bounds)
    return [json.dumps(result), Image(path=result["crop_path"])]


@mcp.tool(structured_output=False)
def measure_plot(session_id: str, x_axis: dict[str, Any], selections: list[dict[str, Any]], y_axis: dict[str, Any] | None = None, reference_label: str | None = None, baseline_pixel: float | None = None, use_label_proposals: bool = False) -> list[Any]:
    """Measure XY selections. use_label_proposals=True accepts supported inspected labels, with explicit selections overriding them; unresolved labels still need review. Axes: {anchor_ids:['X1','X3'],check_ids:['X2'],scale:'linear'|'log10'} or explicit anchors/checks [{pixel,value}]. Selection: {label,candidate_id or pixel:[x,y],mode:'exact'|'peak'|'scatter',review_id?,review_note?}. Ambiguous label assignments require the review region ID and a brief image-based choice; otherwise return unresolved. Explicit abstention: {label,status:'unresolved',reason}. Curve: {label,mode:'curve',x_value,pixel?:[x,approx_y]}. Heights: reference_label and baseline_pixel; y_axis optional. Exported coordinate uncertainty is conditional on the feature assignment; it is not an assignment-confidence interval."""
    result = core.measure_plot(session_id, x_axis, selections, y_axis, reference_label, baseline_pixel,use_label_proposals)
    return [json.dumps(result), Image(path=result["overlay_path"])]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
