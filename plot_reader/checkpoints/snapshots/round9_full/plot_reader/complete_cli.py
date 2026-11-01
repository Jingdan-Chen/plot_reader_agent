"""Opt-in experimental profile; frozen full/fast interfaces are untouched."""
import argparse
import hashlib
import json
from pathlib import Path
from .complete import discover_plot,extract_plot,validate_plan,axis_hypotheses,segmented_axis_proposals


def main():
    parser=argparse.ArgumentParser(prog="plot-reader --profile experimental")
    commands=parser.add_subparsers(dest="command",required=True)
    discover=commands.add_parser("discover");discover.add_argument("image");discover.add_argument("--output",required=True)
    extract=commands.add_parser("extract");extract.add_argument("image");extract.add_argument("--plan",required=True)
    extract.add_argument("--output",required=True);extract.add_argument("--engine",choices=("candidate","legacy"),default="candidate")
    commands.add_parser("mcp");commands.add_parser("version")
    validate=commands.add_parser('validate');validate.add_argument('image');validate.add_argument('--plan',required=True)
    axes=commands.add_parser('axis-proposals');axes.add_argument('--ticks',required=True)
    args=parser.parse_args()
    if args.command=="mcp":
        from .complete_server import main as serve
        serve();return
    if args.command=="version":
        files=("complete.py","complete_cli.py","complete_server.py")
        value=hashlib.sha256(b"".join(Path(__file__).with_name(p).read_bytes() for p in files)).hexdigest()
        result={"profile":"experimental","implementation_id":value}
    elif args.command=="discover":
        result=discover_plot(args.image,args.output);result.pop("ocr",None)
    elif args.command=='validate':result={'errors':validate_plan(args.image,json.loads(Path(args.plan).read_text()))}
    elif args.command=='axis-proposals':
        ticks=json.loads(Path(args.ticks).read_text());result={'ordinary':axis_hypotheses(ticks),'segmented':segmented_axis_proposals(ticks)}
    else:result=extract_plot(args.image,json.loads(Path(args.plan).read_text()),args.output,args.engine)
    print(json.dumps(result))


if __name__=="__main__":main()
