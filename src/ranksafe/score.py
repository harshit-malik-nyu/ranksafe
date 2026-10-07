"""
Extract an answer, score it, and ask whether the ranking held.

Where this kind of evaluation goes wrong
----------------------------------------
Not in the model and not in the benchmark — in the parser. A model that says
"the answer is 26" and a parser that reads the last number in "26 eggs at $2
each, so 52" are the same failure as a wrong answer, and it does not show up
as a parser bug. It shows up as a model looking worse than it is.

Worse, the failure is **asymmetric** here. Perturbed questions have unfamiliar
magnitudes, so a model may produce longer working on them. A parser biased
toward the last number in longer text would manufacture exactly the drop this
project is looking for.

So extraction is deliberate and conservative:

    1. an explicit marker wins. "#### 26", "answer: 26", "answer is 26".
    2. failing that, the last number in the final line only.
    3. failing that, no answer — scored wrong, never guessed.

And the parser is tested on the same kinds of text for both arms, because a
parser that behaves differently on long and short outputs is the one thing
that would fake this project's result.

Rank stability
--------------
The headline is not any model's accuracy. It is whether the **ordering**
survives, because the ordering is what a procurement decision reads. A table
whose rows keep their places under perturbation is usable even if every score
falls; one that reorders is not, whatever the scores are.

Reported as pairwise inversions: how many of the n(n-1)/2 orderings flip
between the original and the variant arm. Reporting a rank correlation instead
would compress exactly the information a buyer needs — *which* pair flipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

MARKER = re.compile(
    r"(?:####|\*\*answer\*\*|answer\s*(?:is|:)|final answer\s*(?:is|:))"
    r"\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)
NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract(text: str) -> float | None:
    """
    Pull a numeric answer out of free text.

    Returns None rather than guessing. A model that did not state an answer
    scores wrong, which is correct — but it must not be scored wrong because
    the parser grabbed a number from the middle of its working.
    """
    if not text:
        return None

    hits = MARKER.findall(text)
    if hits:
        return _to_float(hits[-1])

    # No marker: only the last line, and only if it ends in a number. This is
    # deliberately strict. Scanning the whole output would find the last
    # intermediate result of a long derivation, which is systematically more
    # likely on the perturbed arm.
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    nums = NUMBER.findall(lines[-1])
    return _to_float(nums[-1]) if nums else None


def _to_float(tok: str) -> float | None:
    try:
        return float(tok.replace(",", "").rstrip("."))
    except ValueError:
        return None


def correct(response: str, expected: float, tol: float = 1e-6) -> bool:
    got = extract(response)
    return got is not None and abs(got - expected) <= tol


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

@dataclass
class ArmScore:
    n: int = 0
    right: int = 0
    unparsed: int = 0

    @property
    def accuracy(self) -> float:
        return self.right / self.n if self.n else 0.0

    @property
    def unparsed_rate(self) -> float:
        return self.unparsed / self.n if self.n else 0.0


@dataclass
class ModelResult:
    model: str
    original: ArmScore = field(default_factory=ArmScore)
    variant: ArmScore = field(default_factory=ArmScore)

    @property
    def drop(self) -> float:
        return self.original.accuracy - self.variant.accuracy

    @property
    def parser_asymmetry(self) -> float:
        """
        Difference in unparsed rate between arms.

        If this is large the drop is suspect: the parser may simply be
        failing more often on one side, which looks identical to a capability
        difference and is not one.
        """
        return self.variant.unparsed_rate - self.original.unparsed_rate

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "n": self.original.n,
            "accuracy_original": self.original.accuracy,
            "accuracy_variant": self.variant.accuracy,
            "drop": self.drop,
            "unparsed_original": self.original.unparsed_rate,
            "unparsed_variant": self.variant.unparsed_rate,
            "parser_asymmetry": self.parser_asymmetry,
        }


def rank_stability(results: list[ModelResult],
                   min_margin: float = 0.0,
                   check_power: bool = True) -> dict:
    """
    Does the ordering survive perturbation?

    `min_margin` lets a caller ignore pairs whose original scores were within
    a given distance, because a pair separated by half a point was never
    ordered in any meaningful sense and counting its flip overstates the
    instability.
    """
    if len(results) < 2:
        return {"models": len(results),
                "note": "a ranking needs at least two models"}

    flips, considered, detail = 0, 0, []
    for a, b in combinations(results, 2):
        margin = abs(a.original.accuracy - b.original.accuracy)
        if margin < min_margin:
            continue
        considered += 1
        before = a.original.accuracy - b.original.accuracy
        after = a.variant.accuracy - b.variant.accuracy
        flipped = (before > 0) != (after > 0) and before != 0 and after != 0
        if flipped:
            flips += 1
        detail.append({"pair": [a.model, b.model],
                       "margin_original": before,
                       "margin_variant": after,
                       "flipped": flipped})

    # What could this run have seen? Without it, "no flips" reads as
    # "the ranking held" when it may mean "we could not have detected one".
    power = None
    if check_power and results:
        from .power import detectable
        power = detectable(results[0].original.n)

    return {
        "models": len(results),
        "pairs_considered": considered,
        "flips": flips,
        "flip_rate": flips / considered if considered else 0.0,
        "min_margin": min_margin,
        "pairs": detail,
        "power": power,
        "verdict": _verdict(flips, considered, results, power),
    }


def _verdict(flips: int, considered: int, results: list[ModelResult],
             power: dict | None = None) -> str:
    worst = max((r.parser_asymmetry for r in results), default=0.0)
    if worst > 0.05:
        return (f"Unusable: the parser failed {worst*100:.1f} points more often "
                "on the perturbed arm than the original. That difference is "
                "indistinguishable from a capability drop and has to be fixed "
                "before any of these numbers mean anything.")
    if considered == 0:
        return "No pairs were separated enough to have an ordering."

    drops = [r.drop for r in results]
    span = f"{min(drops)*100:+.1f} to {max(drops)*100:+.1f} points"

    if flips:
        return (f"{flips} of {considered} orderings flipped. A table that "
                "reorders when the numbers in the questions change is not a "
                "basis for choosing a model, whatever caused it.")

    # No flips. Whether that is a finding depends entirely on whether one
    # could have been seen, and a run this size usually could not.
    if power:
        floor = power["smallest_detectable_flip"]
        return (f"No ordering flipped, and this run could not have detected "
                f"one smaller than {floor*100:.1f} points — so that is a "
                "statement about the sample, not about the leaderboard. "
                f"Scores moved by {span}. Detecting a flip the size the "
                "literature reports needs roughly four times the items of "
                "detecting the drop itself.")
    return (f"The ordering held on every pair. Scores moved by {span}, but no "
            "model overtook another.")
