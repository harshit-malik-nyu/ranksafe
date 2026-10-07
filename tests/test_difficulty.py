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
        b = balance(self._p(1, n=3000), self._p(1, n=3000))
        assert b["balanced"]
        assert "imbalance above 0.1 is excluded" in b["verdict"]

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

    def test_the_matched_set_balances_on_point_estimates(self, measured):
        """
        THE CONTROL. Every component's standardised difference sits near
        zero — 0.026 at worst against 0.228 unconstrained.
        """
        m = measured["magnitude_matched"]
        assert m["point_estimates_balanced"]
        assert m["worst_standardised_difference"] < 0.05

    def test_the_sample_cannot_certify_it_and_says_so(self, measured):
        """
        The distinction a tick or a cross would destroy. At n=787 a 95%
        interval on a standardised difference is about 0.099 wide, so a point
        estimate of 0.026 cannot be shown below the 0.1 convention however
        well matched the arms are. The verdict reports what would be needed
        rather than claiming certification it does not have.
        """
        m = measured["magnitude_matched"]
        assert not m["balanced"]
        assert "the sample cannot prove it" in m["verdict"]
        assert "pairs, which this dataset does not yield" in m["verdict"]

    def test_the_unconstrained_set_is_genuinely_imbalanced(self, measured):
        """
        The contrast is the evidence that the control does something. Not
        merely uncertified — its point estimates are an order of magnitude
        larger, 0.228 against 0.026.
        """
        u, m = measured["unconstrained"], measured["magnitude_matched"]
        assert not u["point_estimates_balanced"]
        assert u["worst_standardised_difference"] > 5 * \
               m["worst_standardised_difference"]

    def test_matching_costs_coverage(self, measured):
        assert measured["magnitude_matched"]["pairs"] < \
               measured["unconstrained"]["pairs"]
        assert measured["magnitude_matched"]["pairs"] > 500


class TestBalanceIsNotLuck:
    """
    The committed pairs come from one seed. Balance holding on that draw and
    nothing else would be luck, so it is checked across seeds before being
    relied on.
    """

    @pytest.fixture(scope="class")
    def across(self):
        p = ROOT / "evidence" / "balance.json"
        if not p.exists():
            pytest.skip("balance not measured")
        d = json.loads(p.read_text())
        if "across_seeds" not in d:
            pytest.skip("single-seed run")
        return d["across_seeds"]

    def test_the_matched_set_balances_on_every_seed(self, across):
        m = across["magnitude_matched"]
        assert m["stable"]
        assert max(m["worst_point_estimates"]) < 0.05
        assert len(m["seeds"]) >= 5

    def test_the_unconstrained_set_fails_on_every_seed(self, across):
        """
        Consistency on both sides is the point. An imbalance that appeared on
        one draw would be noise; one that appears on all of them is the
        confound the control exists to remove.
        """
        u = across["unconstrained"]
        assert min(u["worst_point_estimates"]) > 0.15

    def test_the_gap_between_them_holds_on_every_seed(self, across):
        m = across["magnitude_matched"]["worst_point_estimates"]
        u = across["unconstrained"]["worst_point_estimates"]
        for a, b in zip(m, u):
            assert b > 4 * a, f"matched {a}, unconstrained {b}"

    def test_pair_counts_are_stable_across_seeds(self, across):
        """
        A draw that yielded far fewer pairs would be selecting different
        problems, and the balance figure would not be comparable.
        """
        p = across["magnitude_matched"]["pairs_per_seed"]
        assert max(p) - min(p) < 0.1 * min(p)
