"""Convert one of the first examples' own result files to bi-results-v1.

    python suite/adapters/convert.py <adapter> <legacy.json> [--json out.json] [--quick]
                                     [--example NAME] [--live]

--live says the legacy file was measured just now on this machine, so the environment is read
from this machine (GPUs, torch, Triton, NCCL, driver). Without it, the environment is what the
legacy file recorded. Exits 1 when a required check failed, as the examples do.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import adapters  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", choices=adapters.ADAPTERS)
    ap.add_argument("legacy", help="the example's own result JSON")
    ap.add_argument("--json", help="write bi-results-v1 here")
    ap.add_argument("--quick", action="store_true", help="the legacy file came from a quick run")
    ap.add_argument("--example", help="name to record (default: the adapter's name)")
    ap.add_argument("--live", action="store_true", help="measured just now on this machine")
    args = ap.parse_args(argv)
    with open(args.legacy) as f:
        legacy = json.load(f)
    res = adapters.convert(args.adapter, legacy, quick=args.quick, example=args.example, live=args.live)
    res.extra["legacy_file"] = os.path.basename(args.legacy)
    res.write(args.json)
    for e in res.extra["errors"]:
        print("error:", e)
    return 1 if res.exit_code() or res.extra["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
