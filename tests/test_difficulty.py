"""
Tests for the difficulty profiler and the balance check.

This module exists because the control it checks was broken. Matching digit
counts was asserted to hold arithmetic fixed; measuring it showed the variant
arm carried 19% more carries and 15% more non-round operands than the
original. A benchmark claiming to be controlled while being confounded is
worse than one that claims nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranksafe.difficulty import Profile, balance, profile
from ranksafe.perturb import Step, _roundness

ROOT = Path(__file__).resolve().parents[1]


class TestCarries:

    def test_a_simple_addition_has_none(self):
        assert profile([Step("11+11", 22, "")]).carries == 0

    def test_a_carrying_addition_is_counted(self):
        assert profile([Step("17+5", 22, "")]).carries == 1

    def test_multiple_carries_are_counted(self):
        assert profile([Step("99+1", 100, "")]).carries == 2

    def test_borrows_are_counted(self):
        assert profile([Step("20-1", 19, "")]).carries == 1

    def test_non_integers_are_not_guessed_at(self):
        """Column-wise carrying is not meaningful for decimals, so it
        returns zero rather than inventing a number."""
        assert profile([Step("1.5+2.7", 4.2, "")]).carries == 0


class TestRoundness:

    def test_trailing_zeros_are_counted_and_capped(self):
        assert _roundness(50) == 1
        assert _roundness(500) == 2
        assert _roundness(5000) == 2
        assert _roundness(47) == 0

    def test_zero_and_decimals_are_zero(self):
        assert _roundness(0) == 0
        assert _roundness(2.5) == 0


class TestBalance:

    def _p(self, carries, nonround=4, n=200):
        return [Profile(steps=2, carries=carries, operand_digits=12,
                        divisions=1, non_round=nonround) for _ in range(n)]

    def test_identical_arms_are_balanced(self):
        b = balance(self._p(1), self._p(1))
        assert b["balanced"]
        assert "cannot be attributed to harder sums" in b["verdict"]

    def test_a_shifted_arm_is_caught(self):
        """
        A real imbalance must fail rather than be absorbed. Zero variance in
        the fixture makes this the strict case: any mean difference is an
        infinite standardised difference.
        """
        import random
        rng = random.Random(0)
        a = [Profile(steps=2, carries=rng.choice([0, 1, 2]), operand_digits=12,
                     divisions=1, non_round=4) for _ in range(300)]
        b = [Profile(steps=2, carries=rng.choice([1, 2, 3]), operand_digits=12,
                     divisions=1, non_round=5) for _ in range(300)]
        out = balance(a, b)
        assert not out["balanced"]
        assert "still confounded" in out["verdict"]

    def test_the_threshold_is_the_matched_design_convention(self):
        import inspect

        from ranksafe import difficulty
        doc = " ".join(inspect.getdoc(difficulty.balance).split())
        assert "0.1" in doc and "matched designs" in doc


class TestTheCommittedBalance:

    @pytest.fixture(scope="class")
    def measured(self):
        p = ROOT / "evidence" / "balance.json"
        if not p.exists():
            pytest.skip("balance not measured")
        return json.loads(p.read_text())

    def test_the_matched_set_is_balanced(self, measured):
        """
        THE CONTROL, VERIFIED. If this fails, the benchmark's central claim —
        that a residual drop is not arithmetic difficulty — is unsupported.
        """
        m = measured["magnitude_matched"]
        assert m["balanced"]
        assert m["worst_standardised_difference"] < 0.1

    def test_the_unconstrained_set_is_not(self, measured):
        """
        The contrast is the evidence that the control does something. If both
        arms balanced without it, matching would be pointless.
        """
        assert not measured["unconstrained"]["balanced"]

    def test_matching_costs_coverage(self, measured):
        assert measured["magnitude_matched"]["pairs"] < \
               measured["unconstrained"]["pairs"]
        assert measured["magnitude_matched"]["pairs"] > 500
