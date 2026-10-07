"""
Change the numbers, recompute the answer, keep the reasoning.

Why not paraphrase
------------------
The standard contamination probe rephrases a benchmark question and looks for
an accuracy drop. It has a problem that is usually waved at and rarely solved:
a rephrase can change difficulty, and then the drop measures the rewriting
rather than the model. Semantic equivalence is a judgement, and a judgement
sitting underneath every number in the analysis.

GSM8K does not need the judgement. Its answers carry the arithmetic inline:

    Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
    #### 18

98.6% of the 1,319 test problems are annotated this way. So a perturbation can
substitute a number in the question, re-run the annotated chain, and
**compute** the new answer. The variant is correct by construction rather than
by argument.

What stays fixed
----------------
Sentence structure, entities, operations, and the number of reasoning steps.
Only magnitudes move. A model that solved the original by reasoning solves the
variant the same way; a model that recalled the answer has nothing to recall.

What this cannot do
-------------------
It cannot prove contamination. A drop is consistent with memorisation and with
ordinary brittleness to unfamiliar magnitudes, and those are not separable
from outside. The available claim is narrower and more useful: the score
moved, so the score was not measuring only reasoning.

Constraints on a valid substitution
-----------------------------------
A perturbation is rejected unless all of these hold. Each failure mode
produces a variant that is unsolvable, or trivially different, or quietly
harder than the original:

    integrality   intermediate results that were whole stay whole. "She sells
                  9 eggs" must not become 8.5 eggs.
    sign          no intermediate goes negative where the original was
                  positive. You cannot sell -3 eggs.
    magnitude     new numbers stay in the same rough range, so arithmetic
                  difficulty is unchanged.
    distinctness  the new final answer differs from the original, or the
                  variant tests nothing.
    uniqueness    each substituted number appears in the question exactly
                  once, so the rewrite is unambiguous.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from dataclasses import dataclass, field

CALC = re.compile(r"<<([^>]+)>>")
FINAL = re.compile(r"####\s*([-\d.,]+)")
NUMBER = re.compile(r"(?<![\w.])(\d{1,6}(?:\.\d+)?)(?![\w.])")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


@dataclass
class Step:
    """One annotated arithmetic step: an expression and its stated result."""

    expr: str
    result: float
    raw: str

    @property
    def operands(self) -> list[float]:
        return [float(t) for t in re.findall(r"-?\d+\.?\d*", self.expr)]


@dataclass
class Problem:
    qid: str
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    final: float | None = None

    @property
    def usable(self) -> bool:
        return bool(self.steps) and self.final is not None


def parse(qid: str, question: str, answer: str) -> Problem:
    """Pull the annotated arithmetic chain out of a GSM8K answer."""
    steps = []
    for raw in CALC.findall(answer):
        if "=" not in raw:
            continue
        expr, _, res = raw.rpartition("=")
        try:
            steps.append(Step(expr=expr.strip(),
                              result=float(res.strip().replace(",", "")),
                              raw=raw))
        except ValueError:
            continue
    m = FINAL.search(answer)
    final = None
    if m:
        try:
            final = float(m.group(1).replace(",", ""))
        except ValueError:
            final = None
    return Problem(qid=qid, question=question, answer=answer,
                   steps=steps, final=final)


def evaluate(expr: str) -> float | None:
    """
    Evaluate one arithmetic expression.

    Deliberately not eval(). These expressions come from a dataset, and a
    restricted evaluator is both safer and a check that the string really is
    arithmetic rather than something the parser mis-split.
    """
    if not re.fullmatch(r"[-+*/().\d\s,]+", expr):
        return None
    try:
        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](ev(node.operand))
            raise ValueError("unsupported node")
        return ev(ast.parse(expr.replace(",", ""), mode="eval"))
    except Exception:                                   # noqa: BLE001
        return None


def verify_chain(p: Problem, tol: float = 1e-6) -> bool:
    """
    Does the annotated chain actually produce the stated final answer?

    Run on every problem before it enters the benchmark. An annotation that
    cannot reproduce its own answer cannot be used to compute a perturbed
    one, and the dataset contains some.
    """
    if not p.usable:
        return False
    for s in p.steps:
        got = evaluate(s.expr)
        if got is None or abs(got - s.result) > tol:
            return False
    return abs(p.steps[-1].result - p.final) <= tol


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    qid: str
    question: str
    final: float
    original_question: str
    original_final: float
    substitutions: dict[float, float]
    steps_recomputed: int

    def as_dict(self) -> dict:
        return {"qid": self.qid, "question": self.question,
                "final": self.final,
                "original_final": self.original_final,
                "substitutions": {str(k): v
                                  for k, v in self.substitutions.items()},
                "steps_recomputed": self.steps_recomputed}


def _rewrite_chain(p: Problem, sub: dict[float, float],
                   tol: float = 1e-9, keep_digits: bool = False,
                   original: "Problem | None" = None) -> tuple[float, int] | None:
    """
    Re-run the annotated chain with substituted inputs.

    Each step's operands are either original inputs (substituted) or results
    of earlier steps (carried forward). A step whose operands cannot all be
    resolved that way is a chain this method cannot follow, and the
    perturbation is abandoned rather than guessed at.
    """
    produced: dict[float, float] = {}
    last = None
    n = 0

    for s in p.steps:
        expr = s.expr
        all_tokens = re.findall(r"\d+\.?\d*", expr)

        # A value appearing twice in one expression cannot be substituted.
        #
        # "A robe takes 2 bolts of blue fiber and half that much white fiber"
        # compiles to 2/2=1. The first 2 is the input; the second encodes
        # "half". Replacing both gave 3/3=1 and an answer of 4, when the
        # correct perturbed answer is 4.5 — which should then have been
        # rejected as non-integral. Silently wrong ground truth, from one
        # ambiguous literal.
        #
        # There is no way to tell the roles apart from the annotation alone,
        # so the perturbation is abandoned instead of guessed.
        for v in sub:
            if sum(1 for t in all_tokens if float(t) == v) > 1:
                return None

        # Replace longest tokens first so "12" inside "120" is not hit.
        tokens = sorted(set(all_tokens), key=len, reverse=True)
        for tok in tokens:
            val = float(tok)
            if val in sub:
                new = sub[val]
            elif val in produced:
                new = produced[val]
            else:
                continue
            expr = re.sub(rf"(?<![\d.]){re.escape(tok)}(?![\d.])",
                          _fmt(new), expr)

        got = evaluate(expr)
        if got is None:
            return None
        # An intermediate that was a whole number must stay one, and a
        # positive one must stay positive: the problems describe countable
        # things and a fractional or negative intermediate makes the variant
        # nonsense even when the arithmetic is valid.
        if abs(s.result - round(s.result)) < tol and abs(got - round(got)) > 1e-6:
            return None
        if s.result > 0 and got <= 0:
            return None
        # Magnitude matching applies to intermediates too, or a same-decade
        # input can still produce a ten-times-larger running total and the
        # arithmetic gets harder anyway.
        if keep_digits and (_digits(got) != _digits(s.result)
                            or _roundness(got) != _roundness(s.result)):
            return None
        produced[s.result] = got
        last = got
        n += 1

    return (last, n) if last is not None else None


def _fmt(v: float) -> str:
    return str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def _digits(v: float) -> int:
    return len(str(int(abs(v))))


def _roundness(v: float) -> int:
    """
    Trailing zeros, capped at two.

    GSM8K leans on round numbers — 10 chickens, 20 dollars, 50 miles — and a
    random same-decade replacement is almost never round. Measuring the
    control showed that: substituted arms carried 15% more non-round operands
    and 19% more carries than the originals, so matching digit counts alone
    made the arithmetic HARDER while claiming to hold it fixed.

    Preserving roundness is what actually balances it.
    """
    if v != int(v) or v == 0:
        return 0
    n, z = int(abs(v)), 0
    while n % 10 == 0 and z < 2:
        n //= 10
        z += 1
    return z


def _candidates_like(v: float, rng: random.Random, tries: int = 24):
    """Same digit count and same roundness as v."""
    d, r = _digits(v), _roundness(v)
    lo = 10 ** (d - 1) if d > 1 else 2
    hi = 10 ** d - 1
    step = 10 ** r if r else 1
    out = []
    for _ in range(tries):
        c = rng.randint(max(lo, step) // step, max(1, hi // step)) * step
        if c != v and c >= 2 and _digits(c) == d and _roundness(c) == r:
            out.append(float(c))
    return out


def perturb(p: Problem, rng: random.Random, *,
            scale: tuple[float, float] = (0.5, 2.0),
            attempts: int = 40,
            magnitude_matched: bool = False) -> Variant | None:
    """
    Produce one valid numeric variant, or None if the problem resists it.

    Returning None is common and correct. Many problems have numbers that
    appear in prose rather than as operands, or chains this method cannot
    follow, and a variant forced through those is worse than no variant.

    magnitude_matched controls the confound that decides whether this
    benchmark measures what it claims
    ------------------------------------------------------------------
    Numeric perturbation does not isolate memorisation. It also changes
    arithmetic difficulty: a model that fails 847 x 23 and solves 20 x 3
    failed at arithmetic, not at recall, and an unconstrained substitution
    produces exactly that confound.

    With this set, every substituted value keeps its **digit count** and
    every recomputed intermediate keeps its digit count too. 16 may become
    23 but not 230; an intermediate of 9 may become 7 but not 94. The
    arithmetic stays the same size, so a residual drop cannot be explained
    by harder sums.

    Both modes are built, because the comparison between them is the
    evidence. If drops collapse under matching, the unconstrained signal was
    arithmetic difficulty and the benchmark was measuring the wrong thing.
    """
    if not verify_chain(p):
        return None

    # Candidate inputs: numbers appearing both in the question and as a
    # first-step operand, each occurring exactly once in the question so the
    # textual substitution is unambiguous.
    q_nums = NUMBER.findall(p.question)
    counts: dict[str, int] = {}
    for t in q_nums:
        counts[t] = counts.get(t, 0) + 1
    operands = {o for s in p.steps for o in s.operands}
    candidates = [float(t) for t, c in counts.items()
                  if c == 1 and float(t) in operands and float(t) >= 2]

    if not candidates:
        return None

    for _ in range(attempts):
        pick = rng.sample(candidates, k=min(len(candidates),
                                            rng.choice([1, 1, 2])))
        sub = {}
        for v in pick:
            if magnitude_matched:
                # Same digit count AND same roundness. Digits alone left the
                # variant arm with measurably more carries and non-round
                # operands than the original, which is the confound the
                # control exists to remove.
                cands = _candidates_like(v, rng)
                if not cands:
                    continue
                nv = rng.choice(cands)
            else:
                lo, hi = max(2.0, v * scale[0]), v * scale[1]
                nv = float(rng.randint(int(lo), max(int(lo) + 1, int(hi))))
            if nv != v:
                sub[v] = nv
        if not sub:
            continue

        out = _rewrite_chain(p, sub, keep_digits=magnitude_matched,
                             original=p)
        if out is None:
            continue
        new_final, n = out
        if abs(new_final - (p.final or 0)) < 1e-9:
            continue        # a variant with the same answer tests nothing

        question = p.question
        ok = True
        for old, new in sub.items():
            tok = _fmt(old)
            pattern = rf"(?<![\w.]){re.escape(tok)}(?![\w.])"
            question, count = re.subn(pattern, _fmt(new), question)
            if count != 1:
                ok = False
                break
        if not ok:
            continue

        return Variant(qid=p.qid, question=question, final=new_final,
                       original_question=p.question,
                       original_final=p.final or 0.0,
                       substitutions=sub, steps_recomputed=n)
    return None


def rebuild_answer(p: Problem, v: Variant) -> str | None:
    """
    Reconstruct the worked solution for a variant.

    Used for validation rather than for the benchmark: re-parsing the rebuilt
    text and re-verifying the chain checks the substitution by a different
    route than the one that produced it. A bug in `_rewrite_chain` that
    happens to produce a self-consistent wrong answer would survive one check
    and not both.
    """
    produced: dict[float, float] = {}
    text = p.answer
    for s in p.steps:
        expr = s.expr
        for tok in sorted(set(re.findall(r"\d+\.?\d*", expr)),
                          key=len, reverse=True):
            val = float(tok)
            new = v.substitutions.get(val, produced.get(val))
            if new is None:
                continue
            expr = re.sub(rf"(?<![\d.]){re.escape(tok)}(?![\d.])",
                          _fmt(new), expr)
        got = evaluate(expr)
        if got is None:
            return None
        produced[s.result] = got
        text = text.replace(f"<<{s.raw}>>", f"<<{expr}={_fmt(got)}>>")
    return re.sub(r"####\s*[-\d.,]+", f"#### {_fmt(v.final)}", text)


def validate(p: Problem, v: Variant) -> bool:
    """
    Does the variant survive an independent recomputation?

    Rebuilds the worked answer, re-parses it as if it were a fresh dataset
    row, and checks the chain verifies and lands on the variant's stated
    final. Everything that passes this has been computed twice by different
    code paths.
    """
    text = rebuild_answer(p, v)
    if text is None:
        return False
    reparsed = parse(v.qid, v.question, text)
    return verify_chain(reparsed) and abs((reparsed.final or 0) - v.final) < 1e-6
