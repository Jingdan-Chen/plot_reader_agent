from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from . import core

mcp = FastMCP("Plot Reader")


@mcp.tool(structured_output=False)
def inspect_plot(image_path: str, measurement: str = "requested points") -> list[Any]:
    """Inspect an XY plot image. Returns original-image pixel coordinates for OCR ticks and point candidates, optional candidate label_hint from nearby printed text, plus an annotated overview. OCR is proposed, not ground truth; confirm ticks and labels from the image. Call once per image."""
    result = core.inspect_plot(image_path, measurement)
    return [json.dumps(result), Image(path=result["overview_path"])]


@mcp.tool(structured_output=False)
def refine_region(session_id: str, bounds: list[int]) -> list[Any]:
    """Magnify an ambiguous region. bounds=[left,top,right,bottom] in original image pixels; response retains the crop-to-original scale and nearby candidate IDs."""
    result = core.refine_region(session_id, bounds)
    return [json.dumps(result), Image(path=result["crop_path"])]


@mcp.tool(structured_output=False)
def measure_plot(session_id: str, x_axis: dict[str, Any], selections: list[dict[str, Any]], y_axis: dict[str, Any] | None = None, reference_label: str | None = None, baseline_pixel: float | None = None) -> list[Any]:
    """Measure selected XY plot points. Each axis is either {anchor_ids:['X1','X3'],check_ids:['X2'],scale:'linear'|'log10'} using inspect_plot tick IDs, or {anchors:[{pixel,value},{pixel,value}],scale:'linear'|'log10',checks:[{pixel,value}]}; x pixel is horizontal and y pixel vertical. Each selection is {label,candidate_id or pixel:[x,y],mode:'exact'|'peak'|'scatter',radius?:1..30}. Curve query uses {label,mode:'curve',x_value:number,pixel?:[x,approx_y]}; the y hint identifies a series when curves overlap. For normalized peak height, pass reference_label and baseline_pixel (original-image y); y-axis calibration is optional. Returns checked numbers, uncertainty, CSV/JSON, and an overlay."""
    result = core.measure_plot(session_id, x_axis, selections, y_axis, reference_label, baseline_pixel)
    return [json.dumps(result), Image(path=result["overlay_path"])]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
