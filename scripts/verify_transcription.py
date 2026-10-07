#!/usr/bin/env python3
"""
Verify the transcription against columns that were not transcribed.

63 rows of two numbers each were typed out of a PDF. One typo corrupts every
figure downstream, and the analysis would not look wrong — it would look like
a result.

The paper publishes four columns per model: Diff, GSM8k, GSM1k, Z-score. Only
the two accuracies were transcribed. Recomputing Diff and Z from those and
checking they reproduce the published values is an independent check on the
transcription, using data the transcription did not include.

Diff is the strong check: it is an exact identity, so any disagreement is a
typo. Z is the weak one and is treated as such — the published accuracies are
printed to three decimals, so a Z recomputed from them carries the rounding,
and the paper does not say whether its two-proportion test pools the variance.
Trying both: pooled is closer on 32 rows and unpooled on 31, which means the
formula is not recoverable from the table and does not need to be.

So Z is checked against the interval the printing allows, not against a fixed
tolerance. A fixed tolerance is wrong here because rounding is not uniform in
its effect: gpt2-xl scores 0.009 and 0.007, where plus or minus 0.0005 is a
quarter of the difference and the implied Z spans 0.281 to 0.842. A tolerance
loose enough for that row would be meaningless for the rest.

Each accuracy therefore stands for an interval, the four corners give the Z
range, and the published value must fall inside it. That is scale-free: tight
where the data is precise, wide only where the printing genuinely leaves it
wide.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Diff and Z-score as printed in Appendix F, standard prompt. Transcribed
# separately from the accuracies and deliberately not used anywhere else.
PUBLISHED = {
    "Yi-6B-Chat": (0.080, 4.135),
    "math-shepherd-mistral-7b-rl": (0.072, 4.488),
    "command": (0.065, 3.336),
    "Xwin-Math-13B-V1.0": (0.064, 3.334),
    "phi-2": (0.063, 3.167),
    "Meta-Llama-3-8B-Instruct": (0.062, 3.532),
    "Xwin-Math-7B-V1.0": (0.060, 3.040),
    "Meta-Llama-3-8B": (0.054, 2.734),
    "phi-1.5": (0.051, 2.814),
    "Phind-CodeLlama-34B-v2": (0.049, 2.531),
    "CodeLlama-34b-Instruct-hf": (0.045, 2.338),
    "Phi-3-medium-128k-instruct": (0.044, 3.103),
    "CodeLlama-13b-Python-hf": (0.044, 2.759),
    "gemma-7b": (0.043, 2.198),
    "Phi-3-mini-4k-instruct": (0.040, 2.385),
    "Yi-34B-Chat": (0.035, 1.883),
    "mistral-medium-latest": (0.035, 2.104),
    "Mixtral-8x7B-v0.1": (0.035, 1.771),
    "Xwin-Math-70B-V1.0": (0.034, 2.107),
    "Mixtral-8x7B-Instruct-v0.1": (0.030, 1.588),
    "Mistral-7B-v0.1": (0.027, 1.421),
    "Mixtral-8x22B-Instruct-v0.1": (0.026, 1.913),
    "CodeLlama-70b-Instruct-hf": (0.026, 1.323),
    "Llama-2-7b-hf": (0.025, 1.892),
    "Mistral-7B-Instruct-v0.1": (0.025, 1.309),
    "CodeLlama-70b-hf": (0.024, 1.221),
    "gemma-7b-it": (0.023, 1.247),
    "mistral-small-latest": (0.022, 1.343),
    "CodeLlama-13b-hf": (0.021, 1.247),
    "Phi-3-medium-4k-instruct": (0.020, 1.519),
    "Mixtral-8x22B-v0.1": (0.020, 1.138),
    "CodeLlama-34b-hf": (0.017, 0.919),
    "gemma-2b": (0.015, 0.966),
    "Meta-Llama-3-70B-Instruct": (0.014, 1.251),
    "CodeLlama-7b-Python-hf": (0.013, 1.040),
    "dbrx-base": (0.012, 0.707),
    "pythia-12b": (0.011, 1.701),
    "Phi-3-mini-128k-instruct": (0.011, 0.645),
    "Meta-Llama-3-70B": (0.011, 0.707),
    "CodeLlama-34b-Python-hf": (0.010, 0.549),
    "gpt-3.5-turbo": (0.009, 0.546),
    "Mistral-7B-Instruct-v0.2": (0.009, 0.469),
    "claude-3-haiku-20240307": (0.009, 0.532),
    "Llama-2-70b-hf": (0.008, 0.445),
    "CodeLlama-7b-Instruct-hf": (0.007, 0.472),
    "gemini-1.5-pro-preview-0514": (0.006, 0.472),
    "gemini-1.5-pro-preview-0409": (0.005, 0.403),
    "CodeLlama-13b-Instruct-hf": (0.005, 0.257),
    "dbrx-instruct": (0.004, 0.211),
    "gpt-4-turbo": (0.003, 0.270),
    "gpt2-xl": (0.002, 0.778),
    "gpt-4o": (0.002, 0.219),
    "gemini-pro": (-0.001, -0.081),
    "mistral-large-latest": (-0.001, -0.049),
    "gemma-2b-it": (-0.001, -0.106),
    "claude-2.1": (-0.004, -0.336),
    "CodeLlama-7b-hf": (-0.007, -0.525),
    "Llama-2-13b-hf": (-0.011, -0.629),
    "gpt-4": (-0.012, -1.161),
    "claude-3-sonnet-20240229": (-0.016, -0.894),
    "claude-3-opus-20240229": (-0.022, -1.421),
    "deepseek-math-7b-rl": (-0.031, -1.963),
    "gemini-1.5-flash-preview-0514": (-0.038, -2.507),
}


# Half-width of the interval each printed 3-decimal accuracy stands for.
PRINT_PRECISION = 0.0005

# Slack on the interval, absorbing the unrecoverable pooling choice. Pooled is
# closer on 32 rows and unpooled on 31, so the formula cannot be determined
# from the table; the two differ by well under this.
POOLING_SLACK = 0.03


def pooled_z(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion Z, pooled variance — the standard two-sample test."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    data = json.loads((ROOT / "evidence" / "gsm1k_published.json").read_text())
    n1, n2 = data["n_gsm8k"], data["n_gsm1k"]

    diff_bad, z_bad, missing, worst_z = [], [], [], 0.0
    for m in data["models"]:
        name = m["model"]
        if name not in PUBLISHED:
            missing.append(name)
            continue
        pub_diff, pub_z = PUBLISHED[name]

        got_diff = round(m["gsm8k"] - m["gsm1k"], 3)
        if abs(got_diff - pub_diff) > 0.0011:
            diff_bad.append((name, got_diff, pub_diff))

        got_z = pooled_z(m["gsm8k"], n1, m["gsm1k"], n2)
        worst_z = max(worst_z, abs(got_z - pub_z))

        corners = [pooled_z(m["gsm8k"] + a, n1, m["gsm1k"] + b, n2)
                   for a in (-PRINT_PRECISION, PRINT_PRECISION)
                   for b in (-PRINT_PRECISION, PRINT_PRECISION)]
        lo, hi = min(corners) - POOLING_SLACK, max(corners) + POOLING_SLACK
        if not (lo <= pub_z <= hi):
            z_bad.append((name, round(got_z, 3), pub_z))

    print(f"  rows checked              {len(data['models']) - len(missing)}")
    print(f"  Diff mismatches (exact)   {len(diff_bad)}")
    print(f"  Z outside rounding range  {len(z_bad)}")
    print(f"  worst Z deviation         {worst_z:.4f}")
    if missing:
        print(f"  not in the check table    {missing}")
    for n, g, p in diff_bad[:10]:
        print(f"    DIFF  {n:36s} computed {g:+.3f}  published {p:+.3f}")
    for n, g, p in z_bad[:10]:
        print(f"    Z     {n:36s} computed {g:+.3f}  published {p:+.3f}"
              "  (outside what rounding allows)")

    ok = not diff_bad and not z_bad and not missing
    print(f"\n  transcription {'VERIFIED' if ok else 'HAS ERRORS'}")
    (ROOT / "evidence" / "transcription_check.json").write_text(json.dumps({
        "rows_checked": len(data["models"]) - len(missing),
        "diff_mismatches": diff_bad, "z_mismatches": z_bad,
        "missing": missing, "worst_z_deviation": worst_z,
        "verified": ok,
        "method": ("Diff and Z-score were transcribed separately from the "
                   "accuracies and recomputed from them. Agreement on both "
                   "checks the accuracies using columns they do not contain."),
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
