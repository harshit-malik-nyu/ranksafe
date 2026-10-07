# Does a ranking survive changing the numbers?

Companies pick models from leaderboards. Procurement, architecture, "which
model do we standardise on" — all trace back to a benchmark table.

The research literature asks *"is model X contaminated?"*. That is unanswerable
from outside, and an ICLR 2026 result suggests it is becoming unanswerable at
all: across ten detection methods on reasoning models, **AUROC sat near 50%**.
The "contamination equals memorisation" assumption fails when models answer
through chain-of-thought.

**The decision-grade question is different. Does the *ranking* hold?** If
changing the numbers in the questions reorders the table, the figure driving
the purchase is noise, and whose training set caused it stops mattering.

**What this repository can and cannot test.** It builds the instrument and
runs it across model *tiers* from one family. That is not a leaderboard —
a leaderboard compares labs with different training corpora and different
contamination exposure, and that difference is the whole mechanism in
question. Tiers of one family likely share training data, so the arrangement
most likely to show instability is the one this cannot construct. The method
is the contribution; a leaderboard result would need API access across
several labs.

---

## Why numeric perturbation, not paraphrase

The standard probe rephrases a question and looks for an accuracy drop. It has
a problem usually waved at and rarely solved: **a rephrase can change
difficulty**, and then the drop measures the rewriting. Semantic equivalence is
a judgement, and it sits underneath every number in the analysis.

GSM8K does not need the judgement. Its answers carry the arithmetic inline:

```
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18
```

So a variant can substitute a number, re-run the annotated chain, and
**compute** the new answer. Correct by construction rather than by argument.

| Stage | Problems |
|---|---:|
| GSM8K test set | 1,319 |
| Carry `<<calc>>` annotations | 1,301 (98.6%) |
| Chain reproduces its own stated answer | 1,208 (91.6%) |
| Valid numeric variant found | 1,025 |

**93 problems have annotations that do not reproduce their own answer.** They
are excluded before anything else happens.

## Every answer is computed twice

A variant is only accepted if a second, independent recomputation agrees: the
worked solution is rebuilt with the new numbers, re-parsed as if it were a
fresh dataset row, and re-verified end to end. One variant in 1,028 failed
that check and was dropped.

## What gets rejected, and why

A substitution is refused unless all of these hold. Each failure produces a
variant that is unsolvable, trivially different, or quietly harder:

| | |
|---|---|
| **Integrality** | intermediates that were whole stay whole — "she sells 8.5 eggs" is not a question |
| **Sign** | no intermediate goes negative where the original was positive |
| **Magnitude** | new values stay in range, so arithmetic difficulty is unchanged |
| **Distinctness** | the new answer differs from the original, or the variant tests nothing |
| **Unambiguity** | each substituted number appears exactly once in the question |

### The rejection that mattered most

> *"A robe takes 2 bolts of blue fiber and half that much white fiber."*

This compiles to `2/2=1`, `2+1=3`. The first `2` is an input; the second
encodes *"half"*. An early version substituted both, producing `3/3=1` and an
answer of **4** — when the correct value is 4.5, which should then have been
rejected as non-integral. **Silently wrong ground truth from one ambiguous
literal.**

There is no way to tell the two roles apart from the annotation alone, so any
value appearing twice in one expression now aborts the perturbation. That cost
20 problems and bought correctness.

## The confound, and the control that failed before it worked

Numeric perturbation does not isolate memorisation. **It also changes
arithmetic difficulty.** A model that fails 847 × 23 and solves 20 × 3 failed
at arithmetic, not at recall, and an unconstrained substitution produces
exactly that confound.

So the benchmark has a second mode that holds digit counts fixed on every
input, intermediate and final answer. The claim attached to it: a residual
drop cannot be harder sums.

**That claim was false, and measuring it is how I found out.**

| | Original | Variant | Standardised difference |
|---|---:|---:|---:|
| Carries | 0.69 | 0.82 | **0.129** |
| Non-round operands | 4.49 | 5.17 | **0.251** |

Matching digits made the arithmetic *harder* while claiming to hold it fixed —
and slightly worse than no matching at all. GSM8K leans on round numbers (10
chickens, 20 dollars, 50 miles) and a random same-decade replacement is almost
never round. More non-round operands means more carries.

Preserving **roundness** as well as digit count fixes it:

| | Original | Variant | Standardised difference |
|---|---:|---:|---:|
| Carries | 0.65 | 0.67 | **0.022** |
| Non-round operands | 4.33 | 4.34 | **0.003** |
| Operand digits | 12.83 | 12.84 | **0.001** |
| Divisions | 0.55 | 0.55 | **0.000** |

Worst point estimate **0.026**, against **0.228** for the unconstrained set.

### "Balanced" and "imbalance excluded" are different claims

A standardised difference is an estimate and needs its precision attached. At
n=787 a 95% interval on one is about **0.099 wide**, so a point estimate of
0.026 has an upper bound of 0.125 and **cannot be certified below the 0.1
convention however well matched the arms actually are**. Certifying it would
need roughly **1,421 pairs**, which this dataset does not yield.

So the honest claim is the middle one of three, and the code reports it as
such: *the arms look balanced and the sample cannot prove it.* The contrast
still carries the argument — the unconstrained set is not merely uncertified,
its point estimates are an order of magnitude larger.

It costs coverage: **787 pairs** rather than 1,029. The browser harness runs
the balanced set by default, because it is the one whose drop means anything.

`scripts/check_balance.py` regenerates both arms from the build seed and
re-measures, so the control is verified rather than asserted. Run with
`--seeds 5` it checks that balance is not an artefact of one draw: every
matched seed lands under 0.040 and every unconstrained seed above 0.218.

### Order effects, removed rather than argued away

The harness interleaves model tiers instead of running one to completion, and
randomises which arm of each pair is asked first. Calls carry no conversation
history so neither should matter — but running all of tier A then all of tier
B confounds the tier with anything that drifts during the run, and asking the
original first every time makes arm order perfectly collinear with arm. Both
are free to remove, and a reviewer should not have to take *shouldn't matter*
on trust.

## What this evaluation can and cannot see

Computed before building the harness, because discovering it afterwards is
how underpowered results get reported as findings.

The comparison is **paired** — each model answers the same problem in both
arms — so the relevant quantity is discordant pairs and McNemar applies. That
makes it far cheaper than an unpaired comparison. It is still not cheap.

| Target effect | Items for a drop | Items for a flip | Calls across 3 tiers | Minutes |
|---:|---:|---:|---:|---:|
| 20pp | 40 | 160 | 960 | 40 |
| 10pp | 157 | 628 | 3,768 | **157** |
| 5pp | 628 | 2,512 | 15,072 | **628** |

**A rank flip is an interaction** — a difference of differences — and needs
roughly **four times** the sample of a main effect the same size. A run
powered to see a ten-point drop is not powered to see a ten-point change in a
gap, and conflating those is the easiest way to report noise as a reordering.

What a browser-patience budget buys:

| Items | Detects a drop of | Detects a flip of |
|---:|---:|---:|
| 40 | 19.8pp | 39.6pp |
| 100 | 12.5pp | 25.1pp |
| 200 | 8.9pp | 17.7pp |
| 400 | 6.3pp | 12.5pp |

The literature reports drops up to **13pp** for the worst-overfitting models
and around **1pp** for frontier ones. So a run of this scale can see the first
and **cannot see the second at any feasible size** — 1pp would need roughly
15,700 items.

**This is wired into the verdict.** A run that finds no flips reports that it
could not have detected one below its floor, rather than reporting that the
ranking held. Those are different claims and only the first is available here.

## Running it

**[Run it here](https://claude.ai/artifact/N1AX7QhhzJspFTqZ9rKhU8)** — no key,
no setup. Pick a sample size and the page tells you what that size can detect
*before* you spend the time, then runs two or three model tiers across both
arms.

`scripts/evaluate.py` does the same thing from the command line and raises
without an API key rather than fabricating output.

The extractor exists twice, in Python and in the page. That is a liability —
two implementations of the same rule can diverge and produce different numbers
under the same name — so they were checked against each other on every case
that might separate them, including the greedy-parser trap. **Zero
mismatches**, and `tests/test_parity.py` asserts the page still carries the
same rules.

## Status

The benchmark is built, the harness runs, and **no result is committed yet**.
That is the next thing, and the honest version of it will say what the run
could not have seen as prominently as what it did.

## Reproducing

```bash
pip install -e ".[dev]"
pytest -q
python scripts/build_benchmark.py --seed 7
```

Deterministic given the seed, so the committed pairs can be regenerated rather
than trusted.

## The cross-lab question, answered from published data

The limitation above — tiers of one family, not a leaderboard — is the
objection this repository cannot fix with its own tooling. **That arrangement
already exists in published data.**

Scale AI's GSM1k paper evaluated **63 models from more than twenty families**
on GSM8k and on a held-out replica built to match its difficulty, under one
standardised prompt. They report per-model *drops*. Nobody asked whether the
*ordering* moved — which is a different question and the one a procurement
decision reads.

### The table is broadly stable

| Margin | Pairs | Reversals | Rate |
|---:|---:|---:|---:|
| 0.0pp | 1,953 | 45 | 2.3% |
| 2.0pp | 1,860 | 16 | 0.9% |
| 3.0pp | 1,813 | 9 | 0.5% |
| **5.0pp** | 1,723 | **1** | **0.1%** |

Counting every inversion overstates instability — two models a third of a
point apart on 1,319 problems were never ordered. The rate collapses as the
margin rises, which is the signature of noise rather than reordering. **Only
one ordering separated by five points or more reverses.**

### But the reversals are not random

If reversals were measurement noise, the model that *falls* would be drawn at
the base rate of overfit-family models in the sample — **47.6%**. It is not:

| Margin | Fallers from overfit families | Base rate | p |
|---:|---:|---:|---:|
| 2.0pp | 12 / 16 — **75%** | 47.6% | 0.025 |
| 3.0pp | 8 / 9 — **89%** | 47.6% | 0.014 |

The single reversal surviving a five-point margin is
`Phi-3-medium-128k-instruct` losing a **7.2-point lead** to
`gemini-1.5-flash`, ending 1.0 behind — an 8.2-point swing between a family
the authors call systematically overfit and one they call flat.

Family labels come from the authors' prose, not from the accuracies, so the
grouping is not defined by the data it is then tested on.

**The finding is therefore specific rather than alarming.** A leaderboard
does not dissolve under a held-out replica. It reorders rarely — and when it
does, it reorders in the direction contamination predicts, between exactly
the pairs where one side is suspected of having seen the benchmark.

*Accuracies from Zhang et al., NeurIPS 2024 Datasets and Benchmarks Track,
arXiv:2405.00332, Appendix F. They come from one standardised prompt rather
than each model's best, so they do not match published benchmark figures; what
that affects is the level, not whether an ordering survives.*

## The case against

[`docs/against.md`](docs/against.md) lists seven objections, leading with the
one that cannot be answered from here: this measures tiers of one model family
and calls it rank stability.

Also recorded: a drop still is not memorisation, the balance control is
verified on a four-number proxy for arithmetic difficulty, the 787 perturbable
problems are not a random sample of the 1,319, and the committed pairs come
from a single seed whose variance is unmeasured.

## License

MIT. GSM8K is MIT-licensed, © OpenAI 2021.
