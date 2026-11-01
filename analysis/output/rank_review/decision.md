# A tie is not a preserved ordering

**Recommendation:** distinguish strict observed orderings from ties and refuse
comparisons with invalid measurement contracts before interpreting flip rates.

| Synthetic case | Result |
|---|---|
| Original scores tied | No strict ordering to preserve; excluded |
| Variant scores tied | Strict ordering no longer exists; excluded and disclosed |
| Three models, one original tie | Two comparable pairs; two flips |
| Parser fails more often on original arm | Unusable, just as for the opposite direction |
| Ordering preserved in observed counts | Descriptive result, not proof of population stability |

Previously, ties contributed to a zero-flip denominator and the parser gate
checked only variant-minus-original failures. That could make incomplete
ordering evidence look stable and miss contamination in the reverse direction.
The implementation now reports tie counts, uses a null flip rate when no strict
pairs are comparable, and applies the parser gate to the absolute difference.

## Measurement contract

Model names must be unique. Outcome counts must be nonnegative integers with
right plus unparsed no greater than the arm denominator. All arms and models
must have equal positive observed counts. Equal counts are necessary here but
do not prove that models were evaluated on the same item identities; callers
must verify that alignment separately. Missing evaluations must not silently
be converted to wrong answers merely to satisfy the count check.

The existing 5-percentage-point parser threshold is a heuristic quality gate,
not a calibrated statistical test. The power output is a planning approximation,
not an observed significance test or a hard limit below which effects cannot be
seen. Ties, incomplete evaluation and parser contamination require investigation,
not automatic procurement approval. No model superiority is established here.

## Reproduce and inspect

```sh
python -m pytest -q tests/test_rank_boundaries.py
python analysis/rank_review.py
```

The companion JSON contains fixture inputs, outcomes and source hashes. Tests
cover both tie directions, mixed-pair denominators, arm-swapping symmetry and
invalid counts. Historical model runs are preserved; these synthetic cases do
not masquerade as new live-model evidence or realized business outcomes.
