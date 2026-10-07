"""
The cross-lab question, answered from published data.

What this fills
---------------
`docs/against.md` argument 1: the harness in this repository runs model tiers
from one family, and a leaderboard compares labs with different training
corpora. Tiers of one family likely share training data, so the arrangement
most likely to reorder is the one the available tooling cannot construct.

That arrangement already exists. Scale AI's GSM1k paper (Zhang et al., NeurIPS
2024) evaluated 63 models from more than twenty families on GSM8k and on a
held-out replica, under a single standardised prompt. They report the result
as per-model drops — up to 8 points, concentrated in the Phi and Mistral
families, with frontier models flat.

**Nobody asked whether the ranking moved.** That is a different question from
whether scores moved, and it is the one a procurement decision reads.

Why the margin matters more than the count
------------------------------------------
Counting every inversion between two orderings overstates instability, because
most adjacent pairs were never meaningfully ordered. Two models 0.3 points
apart on 1,319 problems are tied; their "flip" is coin-noise being recorded as
a reordering.

So inversions are counted at a sequence of margin thresholds. A pair counts
only if its original separation exceeds the threshold — and the threshold that
matters is roughly the standard error of the difference, about 1.5 points
here.

What this can and cannot say
----------------------------
It uses someone else's measurements. The accuracies are theirs, the evaluation
harness is theirs, and the caveat they attach applies: these numbers come from
one standardised prompt rather than each model's best, so they do not match
published benchmark figures. A model that looks mid-table here may be
top-ranked under its own preferred setting.

What that affects is the level, not the question. The ordering under a fixed
prompt is still an ordering, and whether it survives a held-out replica is
still answerable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Flip:
    a: str
    b: str
    margin_original: float
    margin_replica: float

    @property
    def size(self) -> float:
        return abs(self.margin_original)

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b,
                "margin_original": self.margin_original,
                "margin_replica": self.margin_replica,
                "size": self.size}


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "evidence" / "gsm1k_published.json"
    return json.loads(p.read_text())


def standard_error(p1: float, n1: int, p2: float, n2: int) -> float:
    """SE of a difference of two independent proportions."""
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def inversions(data: dict, margin: float = 0.0) -> dict:
    """
    How many orderings reverse between the original and the replica?

    `margin` excludes pairs whose original separation was smaller than the
    threshold, because those were never ordered in any sense a decision could
    rely on.
    """
    models = data["models"]


    considered, flipped, flips = 0, 0, []
    for a, b in combinations(models, 2):
        before = a["gsm8k"] - b["gsm8k"]
        if abs(before) < margin:
            continue
        considered += 1
        after = a["gsm1k"] - b["gsm1k"]
        if before != 0 and after != 0 and (before > 0) != (after > 0):
            flipped += 1
            flips.append(Flip(a=a["model"], b=b["model"],
                              margin_original=before, margin_replica=after))

    flips.sort(key=lambda f: -f.size)
    return {
        "margin": margin,
        "pairs_considered": considered,
        "flips": flipped,
        "flip_rate": flipped / considered if considered else 0.0,
        "largest_flips": [f.as_dict() for f in flips[:10]],
        "all_flips": [f.as_dict() for f in flips],
    }


def margin_sweep(data: dict,
                 margins: tuple[float, ...] = (0.0, 0.01, 0.015, 0.02,
                                               0.03, 0.05)) -> list[dict]:
    """
    Flip rate as the margin threshold rises.

    The shape is the finding. If flips concentrate at tiny margins the table
    is stable and the inversions are noise; if they survive a margin well
    above the standard error, the ordering genuinely moved.
    """
    out = []
    for m in margins:
        r = inversions(data, m)
        out.append({k: r[k] for k in
                    ("margin", "pairs_considered", "flips", "flip_rate")})
    return out


def typical_standard_error(data: dict) -> float:
    """SE of a difference for two mid-range models, as a reference margin."""
    return standard_error(0.5, data["n_gsm8k"], 0.5, data["n_gsm1k"])


def summarise(data: dict) -> dict:
    se = typical_standard_error(data)
    raw = inversions(data, 0.0)
    meaningful = inversions(data, 2 * se)
    sweep = margin_sweep(data)

    big = [f for f in meaningful["all_flips"] if f["size"] >= 0.05]

    return {
        "models": len(data["models"]),
        "typical_standard_error": se,
        "who_falls_2pp": who_falls(data, 0.02),
        "who_falls_3pp": who_falls(data, 0.03),
        "raw": {k: raw[k] for k in ("pairs_considered", "flips", "flip_rate")},
        "beyond_two_se": {k: meaningful[k] for k in
                          ("margin", "pairs_considered", "flips", "flip_rate")},
        "flips_over_five_points": len(big),
        "largest": meaningful["largest_flips"][:8],
        "sweep": sweep,
        "verdict": _verdict(raw, meaningful, big, se),
    }


# Families the GSM1k authors identify as systematically overfit, and those
# they report as flat. Taken from their text, not inferred from the numbers —
# using the numbers to define the groups and then testing the groups on the
# same numbers would be circular.
OVERFIT_FAMILIES = ("phi", "mistral", "yi-", "xwin", "math-shepherd",
                    "codellama")


def is_overfit_family(name: str) -> bool:
    return any(k in name.lower() for k in OVERFIT_FAMILIES)


def who_falls(data: dict, margin: float = 0.02) -> dict:
    """
    When an ordering reverses, which side falls?

    This is the question that distinguishes noise from contamination. If
    reversals were measurement noise, the model that falls would be drawn at
    the base rate of overfit-family models in the sample. If contamination
    drives them, overfit families should be over-represented among the
    fallers.

    Tested against the base rate with a one-sided binomial tail rather than
    eyeballed, and the family labels come from the authors' prose rather than
    from the accuracies, so the grouping is not defined by the data it is
    then tested on.
    """
    import math

    r = inversions(data, margin)
    fallers = [(f["a"] if f["margin_original"] > 0 else f["b"])
               for f in r["all_flips"]]
    n = len(fallers)
    k = sum(1 for f in fallers if is_overfit_family(f))
    base = (sum(1 for m in data["models"] if is_overfit_family(m["model"]))
            / len(data["models"]))
    p = (sum(math.comb(n, i) * base ** i * (1 - base) ** (n - i)
             for i in range(k, n + 1)) if n else 1.0)

    return {"margin": margin, "reversals": n,
            "fallers_from_overfit_families": k,
            "share": k / n if n else 0.0,
            "base_rate": base, "p_value": p,
            "significant": p < 0.05,
            "fallers": fallers}


def _verdict(raw: dict, meaningful: dict, big: list, se: float) -> str:
    if meaningful["flips"] == 0:
        return ("No ordering separated by more than two standard errors "
                "reversed on the held-out replica. The table is stable where "
                "it was ever meaningfully ordered, and the raw inversions are "
                "pairs that were tied to begin with.")
    return (f"{meaningful['flips']} of {meaningful['pairs_considered']} "
            f"orderings separated by more than two standard errors "
            f"({2*se*100:.1f} points) reversed on the held-out replica, "
            f"including {len(big)} where the original gap exceeded five "
            "points. A table that reorders on a same-difficulty replica is "
            "not a basis for choosing between the models that swapped.")
