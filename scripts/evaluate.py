#!/usr/bin/env python3
"""
Run a model on both arms of the paired benchmark.

Calls a real model and raises without a key rather than fabricating output.
The browser harness in docs/ does the same thing through the page's own
sampling capability, so a result can be produced without one — but neither
route invents an answer.

Items are drawn in a fixed order from a seed, so two models see the same
problems and the comparison between them is paired as well as within-model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksafe.score import ArmScore, ModelResult, extract  # noqa: E402

PROMPT = ("Solve the problem. Show brief working, then give the final "
          "numeric answer on its own last line after '####'.\n\n")


def ask(question: str, model: str, key: str) -> str:
    body = json.dumps({
        "model": model, "max_tokens": 600, "temperature": 0,
        "messages": [{"role": "user", "content": PROMPT + question}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data.get("content", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--benchmark", default=str(ROOT / "evidence" / "paired.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("ANTHROPIC_API_KEY is not set. This harness calls a real model; "
              "it does not simulate one.", file=sys.stderr)
        return 1

    pairs = json.loads(Path(args.benchmark).read_text())["pairs"]
    rng = random.Random(args.seed)
    chosen = rng.sample(pairs, min(args.items, len(pairs)))

    res = ModelResult(args.model)
    res.original, res.variant = ArmScore(), ArmScore()

    for i, pair in enumerate(chosen, 1):
        for arm, score in (("original", res.original), ("variant", res.variant)):
            score.n += 1
            try:
                text = ask(pair[arm]["question"], args.model, key)
            except Exception as exc:                    # noqa: BLE001
                print(f"  call failed: {type(exc).__name__}", file=sys.stderr)
                score.unparsed += 1
                continue
            got = extract(text)
            if got is None:
                score.unparsed += 1
            elif abs(got - pair[arm]["answer"]) < 1e-6:
                score.right += 1
        if i % 25 == 0:
            print(f"  {i}/{len(chosen)}  orig {res.original.accuracy:.1%}  "
                  f"var {res.variant.accuracy:.1%}", flush=True)

    out = res.as_dict()
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
