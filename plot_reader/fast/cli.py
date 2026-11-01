from __future__ import annotations

import argparse
import json

from .core import inspect_plot, measure_plot, refine_region


def main() -> None:
    parser = argparse.ArgumentParser(prog="plot-reader", description="Inspect and measure points in XY plot images")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("image")
    inspect.add_argument("--measurement", default="requested points")
    inspect.add_argument("--full", action="store_true", help="include full OCR boxes and diagnostics")
    refine = commands.add_parser("refine")
    refine.add_argument("session_id")
    refine.add_argument("bounds", help="JSON [x0,y0,x1,y1] in original pixels")
    measure = commands.add_parser("measure")
    measure.add_argument("session_id")
    measure.add_argument("--x-axis", required=True, help="JSON axis object")
    measure.add_argument("--y-axis", help="JSON axis object")
    measure.add_argument("--selections", required=True, help="JSON point selection array")
    measure.add_argument("--reference-label")
    measure.add_argument("--baseline-pixel", type=float, help="original-image y pixel of the signal baseline for normalization")
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_plot(args.image, args.measurement)
        if not args.full:
            result.pop("ocr_text", None)
            result["ticks"] = {axis: [{key: tick[key] for key in ("id", "pixel", "value")} for tick in values] for axis, values in result["ticks"].items()}
    elif args.command == "refine":
        result = refine_region(args.session_id, json.loads(args.bounds))
    else:
        result = measure_plot(args.session_id, json.loads(args.x_axis), json.loads(args.selections), json.loads(args.y_axis) if args.y_axis else None, args.reference_label, args.baseline_pixel)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
