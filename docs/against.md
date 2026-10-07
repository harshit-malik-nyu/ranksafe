# The case against this benchmark

The strongest argument that it should not change how anyone reads a
leaderboard.

---

## 1. It cannot test a leaderboard

The harness runs model *tiers* — quick, default, complex — from one family
behind one sampling interface. That is not a leaderboard. A leaderboard
compares models from different labs, trained on different corpora, with
different contamination exposure, and **that difference is the whole
mechanism** the project claims to probe.

Tiers of one family likely share training data. If contamination drives rank
instability, the arrangement most likely to show it is the one this cannot
construct.

So the available claim is narrower than the framing: *does the ordering of
these three tiers survive numeric perturbation?* Whether a real leaderboard
survives it is the question, and this is a method for asking it, not an
answer.

**This is the most serious objection and nothing in the repository fixes it.**
Fixing it needs API access to several labs' models, which is a budget
question rather than a design one.

## 2. There is no result

The benchmark is built, balanced, powered and runnable. No model has been run
on it and no number is committed. Everything above describes an instrument.

An instrument that has never been used is not evidence about anything, and the
README says so rather than implying the design work is a finding.

## 3. A drop still is not memorisation

The control removes arithmetic difficulty as an explanation. It does not
isolate memorisation, because other explanations survive it:

- **Distributional familiarity.** A model may handle "16 eggs" more fluently
  than "23 eggs" because the former is commoner in text generally, not because
  it saw this problem.
- **Tokenisation.** Numbers tokenise unevenly, and a substitution changes the
  token sequence in ways digit-matching does not control.
- **Brittleness.** A model can be genuinely worse at unfamiliar magnitudes
  without having memorised anything.

The honest claim remains the weak one: the score moved, so the score was not
measuring only reasoning. Why it moved is not recoverable from outside.

## 4. The balance control is verified on a proxy

`difficulty.py` measures carries, operand digits, divisions and non-round
operands. That is four numbers standing in for "how hard is this arithmetic",
and arithmetic difficulty is not four numbers.

Multiplication tables a model has memorised (7×8) versus those it has not
(17×23) are both "two operands, one carry structure" to this profiler. If a
substitution systematically moves between those, the control misses it.

The profiler was good enough to **catch a real failure** — matching digits
alone left the variant arm with 19% more carries — so it is not useless. It is
not a complete model of difficulty and does not claim to be.

## 5. The perturbable subset is not the dataset

787 of 1,319 problems survive to the balanced set. The 532 that do not are
problems with numbers in prose, chains the rewriter cannot follow, values
appearing twice in one expression, or substitutions that break integrality.

**Those are not a random sample.** Problems with simple single-use integer
inputs are over-represented; problems with fractions, ratios and restated
quantities are dropped. A model's score on this subset is not its score on
GSM8K, and a drop measured here may not transfer to the parts that were
excluded.

## 6. One seed, one draw

The committed pairs come from seed 7. A different seed produces different
substitutions, and nothing here measures how much the result moves across
seeds. Given that each problem admits many valid variants, seed variance is a
real source of uncertainty that is currently unquantified.

## 7. GSM8K may be the wrong benchmark to care about

It is a solved benchmark by frontier standards. Rank instability on a
benchmark nobody uses for procurement is a methodological demonstration rather
than a finding anyone should act on. The method transfers to benchmarks that
matter; the specific numbers do not.

---

## What survives

- **The perturbation is correct by construction.** Answers are computed
  through the dataset's own annotated arithmetic and verified by an
  independent second recomputation. That does not depend on any of the above.
- **The control was measured, not asserted** — and failed the first time,
  which is the evidence that measuring it was necessary.
- **The power analysis is arithmetic.** A rank flip is an interaction and
  needs roughly four times the sample of a main effect, whatever benchmark it
  is run on.
- **The parser is guarded in the direction that matters.** A greedy extractor
  would manufacture exactly the drop this looks for, and both implementations
  were checked against each other on the cases that would separate them.

## The objection I cannot answer

Argument 1. This measures tiers of one model family and calls the result rank
stability. The arrangement where contamination would most plausibly reorder a
table — different labs, different corpora — is the one the available tooling
cannot construct.
