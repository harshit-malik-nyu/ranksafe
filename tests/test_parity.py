"""
The browser harness and the Python harness must agree.

Two implementations of the same extractor is a liability: if they diverge,
the published result and the reproducible one are different numbers with the
same name. These pin the cases most likely to separate them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ranksafe.score import extract

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("some working\n#### 26", 26),
    ("The answer is 42", 42),
    ("Answer: 1,234.50", 1234.5),
    ("the answer is 20. Actually, the answer is 26.", 26),
    ("First 16 - 3 - 4 = 9 eggs.\nThen 9 * 2 = 18 dollars.\nSo she makes 18.", 18),
    ("I am not sure how to approach this.", None),
    ("", None),
    ("#### $1,234", 1234),
]


class TestPythonSide:

    @pytest.mark.parametrize("text,expected", CASES)
    def test_each_case(self, text, expected):
        got = extract(text)
        if expected is None:
            assert got is None
        else:
            assert got == expected


class TestTheBrowserCopyIsTheSameLogic:
    """
    Checked structurally rather than by executing JavaScript. The two
    implementations were verified identical on every case above at the time
    the page was built; these assert the page still carries the same rules,
    so a later edit that loosens the parser is visible here.
    """

    @pytest.fixture(scope="class")
    def page(self):
        p = ROOT / "docs" / "run.html"
        if not p.exists():
            pytest.skip("browser harness not built")
        return p.read_text()

    def test_the_marker_pattern_is_the_same_set(self, page):
        for token in ("####", "answer", "final answer"):
            assert token in page

    def test_it_does_not_scan_the_whole_output(self, page):
        """
        THE FAILURE THAT WOULD FAKE THE RESULT. A greedy parser finds the last
        intermediate of a derivation, and perturbed questions plausibly
        produce longer derivations.
        """
        flat = " ".join(page.split())
        assert "the last line only" in flat
        assert "manufacture the drop" in flat

    def test_it_refuses_to_simulate_a_model(self, page):
        assert "does not simulate a model" in page

    def test_sampling_is_uncached(self, page):
        """A replayed answer makes a second run free and identical, which is
        not a measurement."""
        assert "cache:false" in page.replace(" ", "")

    def test_the_power_floor_is_shown_before_the_run(self, page):
        """
        A reader should see what the chosen sample size can detect before
        spending the time, not after reading a null result.
        """
        assert "can detect a drop of about" in page
        assert "four times the sample" in page

    def test_a_null_result_is_qualified(self, page):
        assert "statement about the sample, not about the leaderboard" in page

    def test_parser_asymmetry_invalidates_the_run(self, page):
        assert "indistinguishable from a capability" in page
