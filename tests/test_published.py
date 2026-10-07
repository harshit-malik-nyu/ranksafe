"""
Tests for the cross-lab analysis.

This module answers the objection the rest of the repository cannot: the
harness here runs tiers of one family, and a leaderboard compares labs. The
arrangement exists in published data, and the question asked of it — did the
ORDER move — is not the question that data was published to answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ranksafe.published import (
    inversions, is_overfit_family, load, margin_sweep, summarise,
    typical_standard_error, who_falls,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return load()


class TestTheData:

    def test_it_matches_the_papers_headline_figure(self, data):
        """
        The authors state drops of up to 8 points. If the transcription is
        right, the largest absolute difference is exactly that.
        """
        worst = max(abs(m["gsm8k"] - m["gsm1k"]) for m in data["models"])
        assert abs(worst - 0.080) < 0.001

    def test_the_sample_spans_labs(self, data):
        names = [m["model"].lower() for m in data["models"]]
        for lab in ("gpt-", "claude", "gemini", "llama", "mistral", "phi"):
            assert any(lab in n for n in names), lab
        assert len(data["models"]) > 50

    def test_it_is_attributed(self, data):
        assert "arXiv:2405.00332" in data["source"]
        assert "standardised" in data["note"] or "standardised" in data["source"] \
            or "standardized" in data["note"]


class TestRankStability:

    def test_most_raw_inversions_are_ties(self, data):
        """
        Counting every inversion overstates instability: two models a third of
        a point apart on 1,319 problems were never ordered. The flip rate must
        fall sharply as the margin threshold rises, or the margin is doing no
        work.
        """
        sweep = {r["margin"]: r["flip_rate"] for r in margin_sweep(data)}
        assert sweep[0.0] > 4 * sweep[0.05]

    def test_the_table_is_broadly_stable(self, data):
        """
        Only one ordering separated by five points or more reverses. The
        honest headline is stability, not chaos.
        """
        r = inversions(data, 0.05)
        assert r["flips"] <= 2
        assert r["pairs_considered"] > 1500

    def test_but_reversals_do_occur_beyond_noise(self, data):
        se = typical_standard_error(data)
        r = inversions(data, 2 * se)
        assert r["flips"] > 0


class TestWhoFalls:

    def test_overfit_families_are_over_represented_among_fallers(self, data):
        """
        THE FINDING. If reversals were noise, the falling model would be drawn
        at the base rate of overfit-family models. It is not: at a 3-point
        margin, 8 of 9 fallers come from families the authors independently
        identify as systematically overfit, against a 48% base rate.
        """
        w = who_falls(data, 0.03)
        assert w["share"] > w["base_rate"] * 1.5
        assert w["significant"], f"p = {w['p_value']}"

    def test_the_family_labels_come_from_the_authors_not_the_numbers(self):
        """
        Defining the groups from the accuracies and then testing the groups on
        the same accuracies would be circular.
        """
        import inspect

        from ranksafe import published
        doc = " ".join(inspect.getdoc(published.who_falls).split())
        assert "not defined by the data it is then tested on" in doc
        assert is_overfit_family("Phi-3-medium-128k-instruct")
        assert not is_overfit_family("gpt-4o")

    def test_the_test_is_a_binomial_tail_not_an_eyeball(self, data):
        w = who_falls(data, 0.02)
        assert 0 < w["p_value"] < 1
        assert w["base_rate"] > 0.3


class TestHonesty:

    def test_the_prompt_caveat_is_carried(self):
        import inspect

        from ranksafe import published
        doc = " ".join(published.__doc__.split())
        assert "one standardised prompt rather than each model's best" in doc
        assert "do not match published benchmark figures" in doc

    def test_it_is_marked_as_someone_elses_measurements(self):
        import inspect

        from ranksafe import published
        doc = " ".join(published.__doc__.split())
        assert "It uses someone else's measurements" in doc
