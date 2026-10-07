"""
How many items does this need before a result means anything?

Why this comes before the harness
---------------------------------
It is easy to run forty problems through three models, see a six-point drop,
and report it. It is also meaningless: forty paired binary observations do not
separate a six-point drop from zero.

Computing the requirement first is cheaper than discovering it afterwards, and
the number turns out to constrain the design — a browser-run evaluation has a
patience budget, and that budget may not reach the sample the question needs.

The comparison is paired, which helps
-------------------------------------
Each model answers the same problem in both arms, so the relevant quantity is
not two independent proportions but the **discordant pairs**: items the model
gets right in one arm and wrong in the other. McNemar's test uses only those,
which is why this needs far fewer items than an unpaired comparison of the
same effect would.

    n ≈ (z_{α/2} + z_β)² · p_disc / δ²

with δ the true drop and p_disc the share of items that change state. p_disc
is not known in advance and is the dominant uncertainty here: a model that is
either reliably right or reliably wrong on an item type has few discordant
pairs and needs fewer items; one that is marginal everywhere needs more.

Rank flips need more than drops
-------------------------------
Detecting that model A overtook model B is a comparison of two differences,
and an interaction needs roughly four times the sample of a main effect of the
same size. A design powered to see a ten-point drop is **not** powered to see
a ten-point change in a gap, and conflating them is the easiest way to report
a flip that is noise.

Both numbers are reported, so a run that can detect drops but not flips is
visibly that rather than silently that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Normal quantiles, hardcoded rather than pulling in scipy for two numbers.
Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _z(p: float) -> float:
    return Z.get(round(p, 3), 1.96)


@dataclass
class Requirement:
    effect: float
    power: float
    alpha: float
    discordance: float
    items_for_drop: int
    items_for_flip: int

    @property
    def calls_for_drop(self) -> int:
        """Two arms per item."""
        return self.items_for_drop * 2

    def calls(self, models: int) -> int:
        return self.items_for_flip * 2 * models

    def as_dict(self) -> dict:
        return {
            "effect": self.effect, "power": self.power, "alpha": self.alpha,
            "discordance": self.discordance,
            "items_for_drop": self.items_for_drop,
            "items_for_flip": self.items_for_flip,
            "calls_for_drop_one_model": self.calls_for_drop,
        }


def required(effect: float, *, power: float = 0.80, alpha: float = 0.05,
             discordance: float = 0.20) -> Requirement:
    """
    Items needed to detect a drop of `effect`, and to detect a rank flip.

    The flip figure is four times the drop figure: it is an interaction, a
    difference of differences, and that penalty is structural rather than a
    safety margin.
    """
    if effect <= 0:
        raise ValueError("effect must be positive")
    za, zb = _z(1 - alpha / 2), _z(power)
    n_drop = math.ceil((za + zb) ** 2 * discordance / effect ** 2)
    return Requirement(
        effect=effect, power=power, alpha=alpha, discordance=discordance,
        items_for_drop=n_drop, items_for_flip=n_drop * 4,
    )


def detectable(items: int, *, power: float = 0.80, alpha: float = 0.05,
               discordance: float = 0.20) -> dict:
    """
    The other direction: given a budget of items, what can be seen?

    This is the number a browser-run evaluation actually faces, because the
    constraint is how long someone will sit watching a progress bar.
    """
    if items <= 0:
        return {"items": items, "note": "no items"}
    za, zb = _z(1 - alpha / 2), _z(power)
    drop = math.sqrt((za + zb) ** 2 * discordance / items)
    flip = math.sqrt((za + zb) ** 2 * discordance / (items / 4))
    return {
        "items": items, "power": power,
        "smallest_detectable_drop": min(1.0, drop),
        "smallest_detectable_flip": min(1.0, flip),
        "note": ("A flip is an interaction and needs four times the sample of "
                 "a drop of the same size."),
    }


def plan(models: int, seconds_per_call: float = 2.5,
         effects: tuple[float, ...] = (0.20, 0.10, 0.05)) -> list[dict]:
    """
    What each target effect costs in calls and wall time.

    Wall time is the real constraint on a browser-run evaluation and belongs
    in the plan rather than in a footnote discovered at minute forty.
    """
    out = []
    for e in effects:
        r = required(e)
        calls = r.calls(models)
        out.append({
            "effect_points": e * 100,
            "items_for_drop": r.items_for_drop,
            "items_for_flip": r.items_for_flip,
            "calls_for_flip": calls,
            "minutes": round(calls * seconds_per_call / 60, 1),
        })
    return out
