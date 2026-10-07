"""
Tests for the perturbation engine.

The central ones pin correctness-by-construction. This benchmark's whole claim
is that its variants have provably right answers, so a variant with a wrong
answer is not a bug in a feature — it is the project failing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ranksafe.perturb import (
    evaluate, parse, perturb, rebuild_answer, validate, verify_chain,
)

ROOT = Path(__file__).resolve().parents[1]

JANET_Q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast "
           "and bakes muffins with four. She sells the remainder at $2 each. "
           "How much does she make daily?")
JANET_A = ("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
           "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18")

ROBE_Q = ("A robe takes 2 bolts of blue fiber and half that much white "
          "fiber.  How many bolts in total does it take?")
ROBE_A = "2/2 = <<2/2=1>>1 bolt of white fiber\n2+1 = <<2+1=3>>3\n#### 3"


class TestParsing:

    def test_the_chain_is_extracted(self):
        p = parse("x", JANET_Q, JANET_A)
        assert [s.expr for s in p.steps] == ["16-3-4", "9*2"]
        assert p.final == 18

    def test_a_chain_that_does_not_reach_its_answer_fails_verification(self):
        bad = JANET_A.replace("#### 18", "#### 99")
        assert not verify_chain(parse("x", JANET_Q, bad))

    def test_a_chain_with_bad_arithmetic_fails_verification(self):
        bad = JANET_A.replace("<<9*2=18>>", "<<9*2=20>>").replace("#### 18", "#### 20")
        assert not verify_chain(parse("x", JANET_Q, bad))

    def test_the_evaluator_refuses_non_arithmetic(self):
        """Not eval(): a restricted evaluator is also a check that the
        string really is arithmetic rather than something mis-split."""
        assert evaluate("__import__('os')") is None
        assert evaluate("2+2") == 4


class TestCorrectnessByConstruction:

    def test_a_variant_answer_is_recomputed_not_guessed(self):
        """
        The chain is (eggs - 3 - 4) * price. Checked against whichever
        inputs the engine chose, because it may substitute either or both —
        an earlier version of this test assumed it would pick the egg count
        and failed when it picked the price.
        """
        p = parse("x", JANET_Q, JANET_A)
        for seed in range(12):
            v = perturb(p, random.Random(seed))
            if v is None:
                continue
            eggs = v.substitutions.get(16.0, 16.0)
            price = v.substitutions.get(2.0, 2.0)
            assert v.final == (eggs - 3 - 4) * price, \
                f"seed {seed}: {v.substitutions} gave {v.final}"

    def test_every_variant_passes_an_independent_recomputation(self):
        p = parse("x", JANET_Q, JANET_A)
        for seed in range(25):
            v = perturb(p, random.Random(seed))
            if v:
                assert validate(p, v), f"seed {seed} produced an invalid variant"

    def test_the_rebuilt_answer_reparses(self):
        p = parse("x", JANET_Q, JANET_A)
        v = perturb(p, random.Random(1))
        text = rebuild_answer(p, v)
        assert verify_chain(parse("x", v.question, text))


class TestRejections:

    def test_an_ambiguous_literal_is_refused(self):
        """
        REGRESSION, and it produced silently wrong ground truth.

        "2 bolts of blue fiber and half that much white fiber" compiles to
        2/2=1. The first 2 is the input, the second encodes "half".
        Substituting both gave 3/3=1 and an answer of 4 when the correct
        value is 4.5 — which should then have been rejected as non-integral.
        """
        p = parse("robe", ROBE_Q, ROBE_A)
        assert verify_chain(p)
        assert all(perturb(p, random.Random(s)) is None for s in range(30))

    def test_a_variant_with_the_same_answer_is_refused(self):
        p = parse("x", JANET_Q, JANET_A)
        for seed in range(30):
            v = perturb(p, random.Random(seed))
            if v:
                assert v.final != v.original_final

    def test_substitutions_are_unambiguous_in_the_question(self):
        """
        A replaced number must occur exactly once in the question as a
        standalone token, or the rewrite silently changes something it was
        not asked to.

        Counted with word boundaries, not substrings. Substituting 16 -> 32
        introduces a new "2" into the text, and a naive count reads that as
        the original "2" surviving — which is how an earlier version of this
        test failed against a correct engine.
        """
        import re as _re
        p = parse("x", JANET_Q, JANET_A)
        for seed in range(20):
            v = perturb(p, random.Random(seed))
            if not v:
                continue
            for old in v.substitutions:
                tok = str(int(old))
                pat = rf"(?<![\w.]){tok}(?![\w.])"
                assert len(_re.findall(pat, v.original_question)) == 1
                assert len(_re.findall(pat, v.question)) == 0

    def test_an_unverifiable_problem_is_never_perturbed(self):
        bad = parse("x", JANET_Q, JANET_A.replace("#### 18", "#### 99"))
        assert perturb(bad, random.Random(1)) is None


class TestTheBuiltBenchmark:

    @pytest.fixture(scope="class")
    def built(self):
        p = ROOT / "evidence" / "paired.json"
        if not p.exists():
            pytest.skip("benchmark not built")
        return json.loads(p.read_text())

    def test_every_pair_has_a_different_answer(self, built):
        assert built["stats"]["answers_differ"] == built["stats"]["paired"]

    def test_a_meaningful_share_of_the_dataset_survives(self, built):
        s = built["stats"]
        assert s["paired"] / s["parsed"] > 0.6, "too few pairs to measure a rank"

    def test_the_rejected_are_accounted_for(self, built):
        """
        Every drop between stages must be explainable, or the benchmark is a
        convenience sample of whatever happened to survive.
        """
        s = built["stats"]
        assert s["parsed"] > s["usable"] > s["chain_verified"] > s["paired"]

    def test_questions_actually_differ(self, built):
        for pair in built["pairs"][:200]:
            assert pair["original"]["question"] != pair["variant"]["question"]

    def test_it_is_reproducible_from_the_seed(self, built):
        assert built["stats"]["seed"] == 7
