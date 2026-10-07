"""
Tests for the cross-lab analysis.

This module answers the objection the rest of the repository cannot: the
harness here runs tiers of one family, and a leaderboard compares labs. The
arrangement exists in published data, and the question asked of it — did the
ORDER move — is not the question that data was published to answer.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from ranksafe.published import (
    inversions, is_overfit_family, load, margin_sweep,
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

    def test_the_direction_is_there(self, data):
        """
        Overfit families are over-represented among the models that fall —
        83% against a 48% base rate at a three-point margin.
        """
        w = who_falls(data, 0.03)
        assert w["share"] > w["base_rate"] * 1.5

    def test_but_it_is_not_significant_once_counted_properly(self, data):
        """
        THE CORRECTION, and it retracts a headline.

        A binomial tail over reversals assumes each is an independent draw.
        One model falling behind six others produces six reversals and one
        observation: math-shepherd-mistral-7b-rl accounts for four of the
        nine reversals beyond a three-point margin.

        Counting pairs gave p = 0.014. Counting distinct fallers gives
        p = 0.089 on 5 of 6 — same direction, no longer significant.
        """
        w = who_falls(data, 0.03)
        assert not w["significant"]
        assert w["p_value"] > 0.05
        assert w["distinct_fallers"] < w["reversals"]

    def test_the_inflated_figure_stays_visible(self, data):
        """
        Both numbers are returned. The pairwise one is what the analysis
        looked like before the correction, and hiding it would make the
        correction unauditable.
        """
        w = who_falls(data, 0.03)
        assert w["pairwise_p_value"] < 0.05 < w["p_value"]
        assert w["most_frequent_faller"]

    def test_one_model_drives_the_pairwise_count(self, data):
        w = who_falls(data, 0.02)
        assert "math-shepherd" in w["most_frequent_faller"]

    def test_the_family_labels_come_from_the_authors_not_the_numbers(self):
        """
        Defining the groups from the accuracies and then testing the groups on
        the same accuracies would be circular.
        """
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
        from ranksafe import published
        doc = " ".join(published.__doc__.split())
        assert "one standardised prompt rather than each model's best" in doc
        assert "do not match published benchmark figures" in doc

    def test_it_is_marked_as_someone_elses_measurements(self):
        from ranksafe import published
        doc = " ".join(published.__doc__.split())
        assert "It uses someone else's measurements" in doc


class TestEachReversalIsTested:
    """
    The reversals were quoted before they were tested. A swing is a
    difference of differences and needs the standard error of that quantity,
    not of either accuracy.
    """

    def test_the_headline_reversal_is_a_real_swing(self, data):
        from ranksafe.published import reversal_significance
        r = reversal_significance(data, "Phi-3-medium-128k-instruct",
                                  "gemini-1.5-flash-preview-0514")
        assert r["found"]
        assert r["swing"] > 0.07
        assert r["z"] > 3
        assert r["significant"]

    def test_an_unknown_model_returns_not_found_rather_than_zero(self, data):
        from ranksafe.published import reversal_significance
        assert not reversal_significance(data, "nope", "gpt-4o")["found"]

    def test_the_swing_is_a_difference_of_differences(self, data):
        from ranksafe.published import reversal_significance
        r = reversal_significance(data, "Phi-3-medium-128k-instruct",
                                  "gemini-1.5-flash-preview-0514")
        assert abs(r["swing"] - (r["margin_original"] - r["margin_replica"])) < 1e-9


class TestMultipleComparisons:
    """
    Nine tests at alpha 0.05 expect roughly half a false positive. A count of
    how many were significant is not interpretable without saying how many
    were tried.
    """

    def test_correction_is_applied_and_costs_something(self, data):
        from ranksafe.published import test_all_reversals
        t = test_all_reversals(data, 0.03)
        assert t["significant_holm"] < t["significant_uncorrected"]
        assert t["significant_holm"] >= 1

    def test_the_expected_false_positive_count_is_stated(self, data):
        from ranksafe.published import test_all_reversals
        t = test_all_reversals(data, 0.03)
        assert t["expected_false_positives_uncorrected"] == pytest.approx(
            0.05 * t["tests"])

    def test_holm_is_a_step_down_not_plain_bonferroni(self, data):
        """
        Holm compares the i-th smallest p to alpha/(n-i), so it is uniformly
        more powerful than Bonferroni at the same family-wise guarantee. The
        count must be at least what Bonferroni would give.
        """
        from ranksafe.published import test_all_reversals
        t = test_all_reversals(data, 0.03)
        bonf = sum(1 for r in t["results"] if r["p_value"] <= 0.05 / t["tests"])
        assert t["significant_holm"] >= bonf

    def test_the_verdict_separates_the_two_questions(self, data):
        """
        The swings are real; whether they concentrate in contaminated
        families is a separate claim this sample cannot settle. Collapsing
        them is how the retracted version went wrong.
        """
        from ranksafe.published import test_all_reversals
        v = test_all_reversals(data, 0.03)["verdict"]
        assert "real swings" in v
        assert "separate question this sample cannot settle" in v


class TestTranscription:
    """
    63 rows of two numbers were typed out of a PDF. One typo corrupts every
    figure downstream and the analysis would not look wrong — it would look
    like a result.
    """

    @pytest.fixture(scope="class")
    def check(self):
        p = ROOT / "evidence" / "transcription_check.json"
        if not p.exists():
            pytest.skip("transcription not verified")
        return json.loads(p.read_text())

    def test_every_row_was_checked(self, check):
        assert check["rows_checked"] == 63

    def test_diff_reproduces_exactly(self, check):
        """
        The strong check. Diff is an identity — GSM8k minus GSM1k — so any
        disagreement is a typo, and there are none.
        """
        assert check["diff_mismatches"] == []

    def test_z_falls_inside_what_the_rounding_allows(self, check):
        """
        The weak check, treated as weak — and scale-free.

        A fixed tolerance was the first attempt and was wrong: gpt2-xl scores
        0.009 and 0.007, where the printing's half-thousandth is a quarter of
        the difference and the implied Z spans 0.281 to 0.842. A tolerance
        loose enough for that row is meaningless for the rest.

        Each accuracy stands for an interval instead, and the published Z
        must fall inside the range those intervals imply.
        """
        assert check["z_mismatches"] == []

    def test_the_check_uses_columns_the_transcription_does_not_contain(self, check):
        assert "columns they do not contain" in check["method"]

    def test_it_is_verified(self, check):
        assert check["verified"]
