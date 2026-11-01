"""Read Codex rollout records without treating reasoning tokens as extra output."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

PRICES = {"model": "gpt-6-luna", "tier": "standard", "input_per_million": 0.10, "cached_input_per_million": 0.01, "cache_write_per_million":0.125, "output_per_million": 0.50, "long_context_threshold":272000,"source": "https://developers.openai.com/api/docs/pricing", "checked": "2026-09-23"}


def summarize(path: Path) -> dict:
    starts=[]
    ends=[]
    usage=None
    context=None
    tool_calls=0
    forbidden_reference_calls=[]
    request_usage=[]
    last_total=None
    for line in path.open():
        row=json.loads(line)
        payload=row.get("payload",{})
        if row.get("type")=="response_item" and payload.get("type") in {"custom_tool_call","function_call"}:
            tool_calls+=1
            invocation=str(payload.get("input") or payload.get("arguments") or "")
            if any(token in invocation for token in ("plot-reader-gold.json", "plot_reader/bench.py", "plot_reader.bench", "01a0cc68-74fc-7952-b696-e46650432470", "131, b 108")):
                forbidden_reference_calls.append({"timestamp":row["timestamp"],"tool":payload.get("name")})
        if row.get("type")=="turn_context":
            context=payload
        if row.get("type")=="event_msg":
            event=payload.get("type")
            if event=="task_started": starts.append(row["timestamp"])
            elif event in {"task_complete","turn_aborted"}: ends.append(row["timestamp"])
            elif event=="token_count":
                usage=(payload.get("info") or {}).get("total_token_usage") or usage
                info=payload.get("info") or {}
                signature=json.dumps(usage,sort_keys=True)
                if info.get("last_token_usage") and signature!=last_total:
                    request_usage.append(info["last_token_usage"])
                    last_total=signature
    if usage is None:
        raise ValueError("rollout has no token usage")
    uncached=usage.get("input_tokens",0)-usage.get("cached_input_tokens",0)
    cached=usage.get("cached_input_tokens",0)
    output=usage.get("output_tokens",0)
    def request_cost(item):
        inp=item.get("input_tokens",0);read=item.get("cached_input_tokens",0);write=item.get("cache_write_input_tokens",0)
        factor=2 if inp>PRICES["long_context_threshold"] else 1
        return ((max(0,inp-read-write)*PRICES["input_per_million"]+read*PRICES["cached_input_per_million"]+write*PRICES["cache_write_per_million"])*factor+item.get("output_tokens",0)*PRICES["output_per_million"]*(1.5 if factor==2 else 1))/1_000_000
    request_accounting_complete=bool(request_usage and sum(r.get("input_tokens",0) for r in request_usage)==usage.get("input_tokens",0) and sum(r.get("output_tokens",0) for r in request_usage)==output)
    cost=sum(request_cost(r) for r in request_usage) if request_accounting_complete else (uncached*PRICES["input_per_million"]+cached*PRICES["cached_input_per_million"]+output*PRICES["output_per_million"])/1_000_000
    elapsed=None
    if starts and ends:
        parse=lambda value:datetime.fromisoformat(value.replace("Z","+00:00"))
        elapsed=(parse(ends[-1])-parse(starts[0])).total_seconds()
    return {"file":str(path),"model":context.get("model") if context else None,"reasoning_effort":context.get("effort") if context else None,"service_tier_in_rollout":context.get("service_tier") if context else None,"seconds":elapsed,"tool_calls":tool_calls,"forbidden_reference_calls":forbidden_reference_calls,"input_tokens":usage.get("input_tokens"),"cached_input_tokens":cached,"uncached_input_tokens":uncached,"output_tokens_including_reasoning":output,"reasoning_output_tokens_subset":usage.get("reasoning_output_tokens"),"api_equivalent_usd":round(cost,6),"request_accounting_complete":request_accounting_complete,"max_request_input_tokens":max((r.get("input_tokens",0) for r in request_usage),default=None),"price_snapshot":PRICES}


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("rollouts",nargs="+",type=Path)
    args=parser.parse_args()
    print(json.dumps([summarize(p) for p in args.rollouts],indent=2))


if __name__=="__main__":
    main()
