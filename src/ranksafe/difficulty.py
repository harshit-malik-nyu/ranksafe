"""
Is the arithmetic actually matched, or only the digit count?

Why assert when you can measure
-------------------------------
`magnitude_matched` holds digit counts fixed on every input, intermediate and
final answer. The claim attached to it is that the arithmetic is therefore no
harder, so a residual accuracy drop cannot be explained by harder sums.

Digit count is not the only thing that makes arithmetic hard. 23 x 4 and
27 x 4 have identical digit counts and different carry structure; 50 + 50 and
47 + 38 are both two-digit additions and one is obviously easier. A control
justified by a proxy needs the proxy checked, not trusted.

So this computes a difficulty profile over the whole annotated chain and
tests whether the two arms are balanced on it. If they are not, the control
has not done its job and the matched set's drop is still confounded — which
is worth knowing before a result is read off it, not after.

What goes into the profile
--------------------------
    carries        digit positions where an addition or subtraction carries
                   or borrows. The classic driver of arithmetic error.
    operand_size   total digits across both operands.
    is_division    division is harder than the other three, and non-exact
                   division harder again.
    non_round      operands ending in zero are easier; this counts the ones
                   that do not.

None of these is a complete model of arithmetic difficulty. Together they are
enough to detect a control that has failed, which is all they are for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import fmean, pstdev

OP = re.compile(r"([-+*/])")


@dataclass
class Profile:
    steps: int = 0
    carries: int = 0
    operand_digits: int = 0
    divisions: int = 0
    non_round: int = 0

    def as_dict(self) -> dict:
        return {"steps": self.steps, "carries": self.carries,
                "operand_digits": self.operand_digits,
                "divisions": self.divisions, "non_round": self.non_round}


def _carries(a: float, b: float, op: str) -> int:
    """
    Carry or borrow positions in a column-wise addition or subtraction.

    Only meaningful for integers, which is what these problems mostly use.
    Returns 0 for anything else rather than inventing a number.
    """
    if op not in "+-" or a != int(a) or b != int(b):
        return 0
    x, y = abs(int(a)), abs(int(b))
    n, carry = 0, 0
    if op == "+":
        while x or y:
            s = x % 10 + y % 10 + carry
            carry = 1 if s >= 10 else 0
            n += carry
            x //= 10
            y //= 10
        return n
    hi, lo = (x, y) if x >= y else (y, x)
    borrow = 0
    while hi or lo:
        d = hi % 10 - lo % 10 - borrow
        borrow = 1 if d < 0 else 0
        n += borrow
        hi //= 10
        lo //= 10
    return n


def profile(steps) -> Profile:
    """Difficulty profile of one problem's annotated chain."""
    p = Profile()
    for s in steps:
        p.steps += 1
        ops = OP.findall(s.expr)
        nums = s.operands
        if len(nums) >= 2 and ops:
            p.carries += _carries(nums[0], nums[1], ops[0])
        p.operand_digits += sum(len(str(abs(int(n)))) for n in nums
                                if n == int(n))
        p.divisions += sum(1 for o in ops if o == "/")
        p.non_round += sum(1 for n in nums
                           if n == int(n) and int(n) % 10 != 0)
    return p


def balance(originals: list[Profile], variants: list[Profile]) -> dict:
    """
    Are the two arms balanced on arithmetic difficulty?

    Reported as a standardised difference per component — the mean gap in
    units of the pooled standard deviation. Anything under 0.1 is the
    conventional threshold for "balanced" in matched designs, and is used
    here because this is exactly a matched design.
    """
    out, worst = {}, 0.0
    for field in ("carries", "operand_digits", "divisions", "non_round",
                  "steps"):
        a = [getattr(x, field) for x in originals]
        b = [getattr(x, field) for x in variants]
        if not a or not b:
            continue
        ma, mb = fmean(a), fmean(b)
        sd = ((pstdev(a) ** 2 + pstdev(b) ** 2) / 2) ** 0.5
        smd = abs(ma - mb) / sd if sd > 0 else 0.0

        # A standardised difference is an estimate and needs its precision
        # attached. "Balanced at 0.022" from thirty pairs would be a
        # different statement from the same number at eight hundred, and a
        # reader cannot tell them apart without this.
        n1, n2 = len(a), len(b)
        se = ((n1 + n2) / (n1 * n2) + smd ** 2 / (2 * (n1 + n2))) ** 0.5
        upper = smd + 1.96 * se

        worst = max(worst, upper)
        out[field] = {"original_mean": ma, "variant_mean": mb,
                      "standardised_difference": smd,
                      "upper_95": upper, "n": n1,
                      # Judged on the upper bound, not the point estimate:
                      # a design is balanced if imbalance can be excluded,
                      # not if the sample happened to land near zero.
                      "balanced": upper < 0.1}

    point = max((v["standardised_difference"] for v in out.values()
                 if isinstance(v, dict)), default=0.0)
    n = max((v.get("n", 0) for v in out.values() if isinstance(v, dict)),
            default=0)

    out["worst_standardised_difference"] = point
    out["worst_standardised_difference_upper_95"] = worst
    out["n"] = n
    out["balanced"] = worst < 0.1
    out["point_estimates_balanced"] = point < 0.05

    # Three states, not two.
    #
    # "Balanced" and "imbalance excluded" are different claims, and at this
    # sample size only the first is available. A 95% interval on a
    # standardised difference has a half-width near 0.099 at n=788, so a
    # point estimate of 0.02 cannot be certified below the 0.1 convention
    # however well matched the arms actually are. Collapsing that into a
    # tick or a cross would misreport one way or the other.
    if worst < 0.1:
        verdict = (f"Balanced, and imbalance above 0.1 is excluded: the "
                   f"largest standardised difference is {point:.3f} with a "
                   f"95% upper bound of {worst:.3f}. A residual accuracy drop "
                   "cannot be attributed to harder sums.")
    elif point < 0.05:
        need = int(2 / ((0.1 - point) / 1.96) ** 2) if point < 0.1 else 0
        verdict = (f"Point estimates show balance — the largest standardised "
                   f"difference is {point:.3f} — but at n={n} the 95% upper "
                   f"bound is {worst:.3f}, so imbalance at the 0.1 convention "
                   f"is not excluded. Certifying it would need about "
                   f"{need:,} pairs, which this dataset does not yield. The "
                   "honest claim is that the arms look balanced and the "
                   "sample cannot prove it.")
    else:
        verdict = (f"NOT balanced: the largest standardised difference is "
                   f"{point:.3f} (95% upper {worst:.3f}). The control has not "
                   "done its job and any drop measured on this set is still "
                   "confounded with arithmetic difficulty.")
    out["verdict"] = verdict
    return out
