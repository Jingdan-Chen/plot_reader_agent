"""Annotate failure stages while executing the installed skill's API client.

No requests or materialization logic is duplicated here. This script runs in
the existing interpreter that supplies requests, which may differ from the app.
"""
import importlib.util
import os
from pathlib import Path
import sys


class StageFailure(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage=stage


def annotate(function, stage):
    def invoke(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except StageFailure:
            raise
        except Exception as exc:
            raise StageFailure(stage,str(exc)) from None
    return invoke


def main():
    script=Path(sys.argv[1]).resolve()
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location("installed_paddleocr_skill",script)
    client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    for function,stage in {"submit":"submission","poll":"polling","download":"download",
                           "parse_jsonl":"validation","materialize":"materialization",
                           "verify_local_image_links":"validation"}.items():
        setattr(client,function,annotate(getattr(client,function),stage))
    sys.argv=[str(script),*sys.argv[2:]]
    client.main()


if __name__=="__main__":
    try:
        main()
    except StageFailure as exc:
        message=str(exc)
        for key,value in os.environ.items():
            if value and any(word in key.upper() for word in ("TOKEN","SECRET","API_KEY")):
                message=message.replace(value,"[redacted]")
        print(f"PADDLEOCR_STAGE={exc.stage}\n{message}",file=sys.stderr)
        raise SystemExit(1)
