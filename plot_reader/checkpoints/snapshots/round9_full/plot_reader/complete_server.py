"""Additional MCP tools available only with PLOT_READER_PROFILE=experimental."""
import json
from typing import Any
from mcp.server.fastmcp import Image
from .server import mcp
from .complete import discover_plot,extract_plot,validate_plan,axis_hypotheses,segmented_axis_proposals


@mcp.tool(structured_output=False)
def discover_complete_plot(image_path:str,output_dir:str)->list[Any]:
    """Propose Cartesian panels and independently checked axis hypotheses. Verify the overview, recover missed panels, and inventory every visible series. Source pixels throughout; captions/molecules excluded."""
    result=discover_plot(image_path,output_dir);result.pop("ocr",None)
    return [json.dumps(result)]+[Image(path=p) for p in result["review_image_paths"]]


@mcp.tool(structured_output=False)
def extract_complete_plot(image_path:str,plan:dict[str,Any],output_dir:str)->list[Any]:
    """Locally export exhaustive peaks, markers and curves from a reviewed plan. Panels have id,bounds,x_axis,y_axis,series. Series have id,kind=curve|markers,seeds,color_rgb?,bounds?,baseline?. Axes require two anchors plus independent checks per segment. Returns export paths/counts and overlay, not dense rows."""
    result=extract_plot(image_path,plan,output_dir)
    return [json.dumps(result),Image(path=result["overlay_path"])]


@mcp.tool(structured_output=False)
def validate_complete_plot_plan(image_path:str,plan:dict[str,Any])->str:
    """Check original bounds, source-supported seeds and independent calibration. Errors supply observed pixel evidence; verify identity before repair."""
    return json.dumps({'errors':validate_plan(image_path,plan)})


@mcp.tool(structured_output=False)
def propose_complete_plot_axes(ticks:list[dict[str,Any]])->str:
    """Compare linear/log mappings and propose separately calibrated tick groups. A segmented proposal still needs visible break confirmation."""
    return json.dumps({'ordinary':axis_hypotheses(ticks),'segmented':segmented_axis_proposals(ticks)})


def main():mcp.run(transport="stdio")
