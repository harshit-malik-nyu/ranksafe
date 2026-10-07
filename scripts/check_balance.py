#!/usr/bin/env python3
"""
Verify the magnitude-matched control actually equalises arithmetic difficulty.

Regenerates both arms deterministically from the build seed and profiles the
annotated chains, rather than trusting that matching digit counts was enough.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksafe.difficulty import balance, profile          # noqa: E402
from ranksafe.perturb import (                            # noqa: E402
    parse, perturb, rebuild_answer, validate, verify_chain,
)


def arms(verified, matched: bool, seed: int):
    """Difficulty profiles for both arms of every surviving pair."""
    rng = random.Random(seed)
    o, v = [], []
    for p in verified:
        var = perturb(p, rng, magnitude_matched=matched)
        if var is None or not validate(p, var):
            continue
        text = rebuild_answer(p, var)
        reparsed = parse(var.qid, var.question, text)
        o.append(profile(p.steps))
        v.append(profile(reparsed.steps))
    return o, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat across this many seeds; balance holding on "
                         "one draw could be luck")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "balance.json"))
    args = ap.parse_args()

    rows = [json.loads(l)
            for l in open(ROOT / "evidence" / "gsm8k-test.jsonl")]
    parsed = [parse(str(i), r["question"], r["answer"])
              for i, r in enumerate(rows)]
    verified = [p for p in parsed if p.usable and verify_chain(p)]

    out = {}
    seeds = [args.seed + i for i in range(args.seeds)]

    if args.seeds > 1:
        # Objection 6 in docs/against.md: the committed pairs come from one
        # draw. Balance holding on that draw and nothing else would be luck,
        # so it is checked across seeds before being relied on.
        across = {}
        for label, matched in (("unconstrained", False),
                               ("magnitude_matched", True)):
            points, pairs = [], []
            for sd in seeds:
                o, v = arms(verified, matched, sd)
                b = balance(o, v)
                points.append(b["worst_standardised_difference"])
                pairs.append(len(o))
            across[label] = {
                "seeds": seeds,
                "worst_point_estimates": points,
                "max_across_seeds": max(points),
                "pairs_per_seed": pairs,
                "stable": max(points) < 0.05,
            }
            print(f"\n  {label} across {len(seeds)} seeds")
            print(f"    worst smd per seed: "
                  f"{[round(x, 3) for x in points]}")
            print(f"    max {max(points):.3f} -> "
                  f"{'STABLE' if max(points) < 0.05 else 'VARIES'}")
            print(f"    pairs per seed: {pairs}")
        out["across_seeds"] = across

    for label, matched in (("unconstrained", False),
                           ("magnitude_matched", True)):
        o, v = arms(verified, matched, args.seed)
        out[label] = {"pairs": len(o), **balance(o, v)}
        b = out[label]
        print(f"\n  {label}  ({len(o)} pairs)")
        for f in ("carries", "operand_digits", "divisions", "non_round"):
            if f in b:
                x = b[f]
                print(f"    {f:16s} {x['original_mean']:>7.2f} -> "
                      f"{x['variant_mean']:>7.2f}   smd {x['standardised_difference']:.3f}"
                      f"  (95% upper {x['upper_95']:.3f})"
                      f"  {'ok' if x['balanced'] else 'NOT EXCLUDED'}")
        print(f"    worst 95% upper {b['worst_standardised_difference_upper_95']:.3f}"
              f" -> {'BALANCED' if b['balanced'] else 'IMBALANCE NOT EXCLUDED'}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
