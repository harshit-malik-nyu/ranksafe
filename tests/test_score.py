"""
Tests for extraction and rank stability.

Most of these guard the parser, because the parser is where this kind of
evaluation fails silently. A parser that behaves differently on long and short
outputs would manufacture the exact result the project is looking for, and
perturbed questions plausibly produce longer working.
"""

from __future__ import annotations

from ranksafe.score import (
    ArmScore, ModelResult, correct, extract, rank_stability,
)


class TestExtraction:

    def test_the_gsm8k_marker_wins(self):
        assert extract("some working\n#### 26") == 26

    def test_natural_phrasings_are_found(self):
        for t in ("The answer is 42", "Answer: 42", "final answer: 42",
                  "**Answer** 42"):
            assert extract(t) == 42, t

    def test_the_last_marker_wins_not_the_first(self):
        """
        Models revise. "The answer is 20... wait, the answer is 26" must
        score 26.
        """
        assert extract("the answer is 20. Actually, the answer is 26.") == 26

    def test_commas_and_currency_are_handled(self):
        assert extract("#### $1,234") == 1234
        assert extract("Answer: 1,234.50") == 1234.5

    def test_working_is_not_mistaken_for_an_answer(self):
        """
        THE FAILURE THAT WOULD FAKE THIS PROJECT. Scanning the whole output
        finds the last intermediate of a derivation, and perturbed questions
        plausibly produce longer derivations — so a greedy parser invents a
        drop that is not there.
        """
        t = ("First 16 - 3 - 4 = 9 eggs.\n"
             "Then 9 * 2 = 18 dollars.\n"
             "So she makes 18.")
        assert extract(t) == 18
        greedy = ("Step one gives 9.\nStep two gives 18.\n"
                  "Let me double check: 9 times 2 is 18 and 16 minus 7 is 9.\n"
                  "#### 18")
        assert extract(greedy) == 18, "the marker must beat the trailing working"

    def test_no_answer_returns_none_rather_than_a_guess(self):
        assert extract("I am not sure how to approach this.") is None
        assert extract("") is None
        assert extract("   \n  ") is None

    def test_a_missing_answer_scores_wrong_not_crash(self):
        assert correct("no idea", 26) is False

    def test_the_parser_is_symmetric_on_length(self):
        """
        The same answer, stated briefly and at length, must parse the same.
        If it does not, arm length differences become score differences.
        """
        short = "#### 26"
        long = ("Let me work through this carefully. We start with 20 eggs. "
                "She eats 3, leaving 17. She bakes with 4, leaving 13. "
                "At $2 each that is 26. Checking: 13 times 2 is 26.\n#### 26")
        assert extract(short) == extract(long) == 26


class TestRankStability:

    def _m(self, name, o, v, uo=0.0, uv=0.0, n=1000):
        """
        n=1000 so sub-percent margins survive the integer conversion. At
        n=100 an accuracy of 0.801 and one of 0.800 both became 80 right, and
        a test about narrow margins had no margin to test.
        """
        r = ModelResult(name)
        r.original = ArmScore(n=n, right=round(o * n), unparsed=round(uo * n))
        r.variant = ArmScore(n=n, right=round(v * n), unparsed=round(uv * n))
        return r

    def test_a_held_ordering_is_reported_as_decision_grade(self):
        rs = [self._m("a", 0.90, 0.80), self._m("b", 0.70, 0.60)]
        out = rank_stability(rs)
        assert out["flips"] == 0
        assert "decision-grade" in out["verdict"]

    def test_a_flip_is_caught(self):
        rs = [self._m("a", 0.80, 0.60), self._m("b", 0.75, 0.70)]
        out = rank_stability(rs)
        assert out["flips"] == 1
        assert "not a basis for choosing a model" in out["verdict"]

    def test_scores_can_fall_without_the_ranking_moving(self):
        """
        The distinction the project rests on. Every model dropping twenty
        points is survivable; one overtaking another is not.
        """
        rs = [self._m("a", 0.90, 0.70), self._m("b", 0.70, 0.50)]
        out = rank_stability(rs)
        assert out["flips"] == 0
        assert all(r.drop > 0.15 for r in rs)

    def test_a_margin_filter_ignores_pairs_that_were_never_ordered(self):
        rs = [self._m("a", 0.801, 0.70), self._m("b", 0.800, 0.75)]
        assert rank_stability(rs)["flips"] == 1
        assert rank_stability(rs, min_margin=0.02)["pairs_considered"] == 0

    def test_parser_asymmetry_invalidates_the_whole_result(self):
        """
        If the parser fails more on one arm, a capability drop and a parsing
        drop look identical. The verdict must refuse rather than report.
        """
        rs = [self._m("a", 0.90, 0.60, uo=0.01, uv=0.20),
              self._m("b", 0.70, 0.60)]
        out = rank_stability(rs)
        assert "Unusable" in out["verdict"]
        assert "indistinguishable from a capability drop" in out["verdict"]

    def test_one_model_is_not_a_ranking(self):
        assert "at least two models" in rank_stability([self._m("a", .9, .8)])["note"]
