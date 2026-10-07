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


def reversal_significance(data: dict, a: str, b: str) -> dict:
    """
    Is a specific reversal distinguishable from zero?

    A reversal is a difference of differences — model A's lead on the
    original minus its lead on the replica — so it needs the standard error
    of that quantity, not of either accuracy.

    Worth computing because the five-point reversal is quoted as the headline
    case, and quoting a swing without testing it is the same error as
    reporting a drop without testing it.
    """
    m = {x["model"]: x for x in data["models"]}
    if a not in m or b not in m:
        return {"found": False}
    n1, n2 = data["n_gsm8k"], data["n_gsm1k"]
    A, B = m[a], m[b]

    before = A["gsm8k"] - B["gsm8k"]
    after = A["gsm1k"] - B["gsm1k"]
    did = before - after

    var = (A["gsm8k"] * (1 - A["gsm8k"]) / n1
           + B["gsm8k"] * (1 - B["gsm8k"]) / n1
           + A["gsm1k"] * (1 - A["gsm1k"]) / n2
           + B["gsm1k"] * (1 - B["gsm1k"]) / n2)
    se = var ** 0.5
    z = did / se if se else 0.0
    p = 2 * (1 - _normal_cdf(abs(z)))

    return {"found": True, "a": a, "b": b,
            "margin_original": before, "margin_replica": after,
            "swing": did, "standard_error": se, "z": z, "p_value": p,
            "significant": p < 0.05}


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


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


def test_all_reversals(data: dict, margin: float = 0.03) -> dict:
    """
    Every reversal tested, with a correction for testing every reversal.

    Nine tests at alpha 0.05 expect roughly half a false positive by chance,
    so a count of "how many were significant" is not interpretable without
    saying how many were tried. Holm-Bonferroni controls the family-wise
    error rate and is used rather than Bonferroni because it is uniformly
    more powerful at the same guarantee.

    Reported both ways. The uncorrected count is what the individual tests
    say; the corrected count is what survives having asked nine questions.
    """
    r = inversions(data, margin)
    tests = [reversal_significance(data, f["a"], f["b"])
             for f in r["all_flips"]]
    tests = [t for t in tests if t.get("found")]
    n = len(tests)

    # Holm-Bonferroni: sort ascending, compare the i-th to alpha/(n-i).
    order = sorted(range(n), key=lambda i: tests[i]["p_value"])
    survives = [False] * n
    for rank, idx in enumerate(order):
        if tests[idx]["p_value"] <= 0.05 / (n - rank):
            survives[idx] = True
        else:
            break
    for i, t in enumerate(tests):
        t["survives_holm"] = survives[i]

    return {
        "margin": margin, "tests": n,
        "significant_uncorrected": sum(1 for t in tests if t["significant"]),
        "significant_holm": sum(survives),
        "expected_false_positives_uncorrected": 0.05 * n,
        "results": tests,
        "verdict": _reversal_verdict(n, sum(1 for t in tests if t["significant"]),
                                     sum(survives)),
    }


def _reversal_verdict(n: int, raw: int, holm: int) -> str:
    if holm == 0:
        return (f"None of the {n} reversals survives correction for having "
                "tested all of them. The orderings moved; none of the "
                "individual moves can be distinguished from chance once the "
                "search is accounted for.")
    return (f"{holm} of {n} reversals survive Holm-Bonferroni correction "
            f"({raw} were significant before correcting for having tested "
            f"{n}). These are real swings, not noise — though whether they "
            "concentrate in contaminated families is a separate question "
            "this sample cannot settle.")


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
        "reversal_tests_3pp": test_all_reversals(data, 0.03),
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

    COUNTED OVER DISTINCT MODELS, NOT PAIRS
    ---------------------------------------
    A binomial tail over reversals assumes each is an independent draw. They
    are not. One model that falls behind six others produces six reversals
    and one observation: math-shepherd-mistral-7b-rl accounts for four of the
    nine reversals beyond a three-point margin.

    Counting pairs gave p = 0.014 at that margin. Counting distinct fallers
    gives **p = 0.089** on 5 of 6 — the same direction, no longer significant.
    Both are returned, because the pairwise figure is the one that looks like
    a result and the distinct figure is the one that is.
    """
    import math

    r = inversions(data, margin)
    fallers = [(f["a"] if f["margin_original"] > 0 else f["b"])
               for f in r["all_flips"]]
    base = (sum(1 for m in data["models"] if is_overfit_family(m["model"]))
            / len(data["models"]))

    def tail(k: int, n: int) -> float:
        if not n:
            return 1.0
        return sum(math.comb(n, i) * base ** i * (1 - base) ** (n - i)
                   for i in range(k, n + 1))

    n_pairs = len(fallers)
    k_pairs = sum(1 for f in fallers if is_overfit_family(f))

    distinct = sorted(set(fallers))
    n_models = len(distinct)
    k_models = sum(1 for f in distinct if is_overfit_family(f))
    p_models = tail(k_models, n_models)

    return {"margin": margin,
            "reversals": n_pairs,
            "distinct_fallers": n_models,
            "fallers_from_overfit_families": k_models,
            "share": k_models / n_models if n_models else 0.0,
            "base_rate": base,
            "p_value": p_models,
            "significant": p_models < 0.05,
            # Kept visible so the inflation is auditable rather than hidden.
            "pairwise_k_n": [k_pairs, n_pairs],
            "pairwise_p_value": tail(k_pairs, n_pairs),
            "most_frequent_faller": max(
                set(fallers), key=fallers.count) if fallers else None,
            "fallers": distinct}


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
            "points. The table is broadly stable. Whether the reversals "
            "concentrate in the families the authors call overfit is "
            "suggestive and not established: counted over distinct models "
            "rather than pairs, the sample is too small to separate from "
            "chance.")
