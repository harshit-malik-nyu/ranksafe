#!/usr/bin/env python3
"""
Build the paired benchmark: each original beside its numeric variant.

Deterministic given a seed, so the committed file is reproducible and a
reviewer can regenerate it rather than trusting it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksafe.perturb import parse, perturb, validate, verify_chain  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default=str(ROOT / "evidence" / "gsm8k-test.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "evidence" / "paired.json"))
    ap.add_argument("--magnitude-matched", action="store_true",
                    help="hold digit counts fixed throughout, so a residual "
                         "drop cannot be arithmetic difficulty")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.source)]
    parsed = [parse(str(i), r["question"], r["answer"])
              for i, r in enumerate(rows)]

    stats = {"parsed": len(parsed)}
    usable = [p for p in parsed if p.usable]
    stats["usable"] = len(usable)
    verified = [p for p in usable if verify_chain(p)]
    stats["chain_verified"] = len(verified)

    rng = random.Random(args.seed)
    pairs = []
    for p in verified:
        v = perturb(p, rng, magnitude_matched=args.magnitude_matched)
        if v is None:
            continue
        if not validate(p, v):
            continue
        pairs.append({
            "qid": p.qid,
            "original": {"question": p.question, "answer": p.final},
            "variant": {"question": v.question, "answer": v.final},
            "substitutions": {str(k): val for k, val in v.substitutions.items()},
            "steps": v.steps_recomputed,
        })
        if args.limit and len(pairs) >= args.limit:
            break

    stats["paired"] = len(pairs)
    stats["seed"] = args.seed
    stats["magnitude_matched"] = args.magnitude_matched
    stats["answers_differ"] = sum(
        1 for x in pairs if x["original"]["answer"] != x["variant"]["answer"])

    Path(args.out).write_text(json.dumps(
        {"stats": stats, "pairs": pairs}, indent=2))
    for k, v in stats.items():
        print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
