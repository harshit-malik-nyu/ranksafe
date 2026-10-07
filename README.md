# Does the leaderboard survive changing the numbers?

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

## Status

The paired benchmark is built and committed. What is **not** done: running
models on it. That comes next, and the result — whether ranks hold or flip —
is the point of the exercise.

## Reproducing

```bash
pip install -e ".[dev]"
pytest -q
python scripts/build_benchmark.py --seed 7
```

Deterministic given the seed, so the committed pairs can be regenerated rather
than trusted.

## License

MIT. GSM8K is MIT-licensed, © OpenAI 2021.
