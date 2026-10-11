# Decision assurance: ranksafe

Use rank sensitivity to identify a shortlist, then collect paired task evidence for the actual selection. A descriptive reversal is a prompt to investigate, not proof of superiority.

**Evidence:** None of nine selected reversals survives Holm correction over all 1,953 searched pairs. Missing item-paired covariance prevents exact paired inference from the aggregate table.

**Uncertainty:** Assumed procurement-loss rates are independent scenario inputs, not measured by leaderboard inversions.

## Robustness of the assumed business case

| Alternative | Feasible grid cases / 27 | Worst regret when feasible | Maximum volume at fixed review capacity |
|---|---:|---:|---:|
| Full paired evaluation before each selection | 9 | $29,880.00 | 26 |
| Aggregate screen then paired shortlist | 9 | $4,365.00 | 50 |
| Select the headline leaderboard winner | 0 | No feasible case | 400 |

**Robust minimax choice:** None. No option is feasible across the entire stress grid.

Finite assumed grid, not a probability distribution or protection outside the grid. Conditional regret never overrides feasibility. Null robust option means no unconditional recommendation. Cost regret is relative to the cheapest feasible option in that same scenario. Failure counts can overlap.

## Missing-outcome sensitivity

| Outcomes masked per arm | Primary effect bounds | Incident effect bounds | Every completion clears margins? |
|---|---:|---:|---|
| 0% | [3.7%, 3.7%] | [1.0%, 1.0%] | False |
| 5% | [-1.5%, 8.5%] | [-4.1%, 5.9%] | False |
| 10% | [-7.0%, 13.0%] | [-9.1%, 10.9%] | False |
| 20% | [-17.2%, 22.8%] | [-19.3%, 20.7%] | False |

These are identification bounds on deliberately masked synthetic outcomes, not confidence intervals or measured missingness. Failures and non-starters remain in the assigned denominator. Passing this sensitivity check would still require statistical and domain-quality gates.

## Domain measurement contract

**Decision:** Choose which candidate comparisons justify a model procurement or release decision.

**Primary outcome:** Prespecified paired performance differences on decision-relevant tasks, adjusted across the searched comparison family.

**Denominator:** All searched comparisons for multiplicity; all eligible paired task items for performance estimation.

**Independent unit:** Paired task item, with shared-source or benchmark clustering where applicable.

**Quality guardrail:** Full search-family accounting, practical effect size, critical task slices and procurement cost.

**Minimum required data:** Candidate IDs, complete search history, item-level paired outcomes, task weights, comparison family and procurement assumptions.

**Missingness and failure handling:** Aggregate scores cannot reconstruct missing paired covariance. Report approximate inference and do not invent item-level precision.

**Release rule:** No selected reversal is a confirmed discovery unless it survives the appropriate family correction and practical decision threshold.

**Reversal trigger:** Results fail full-family correction, paired evidence contradicts the aggregate ranking, or relevant task weights change.

## Interview defense

**Why report that none of nine reversals survives?**

It prevents selection after searching 1,953 pairs from being presented as confirmatory evidence. A credible negative result can prevent a bad purchase.

**What would make the conclusion stronger?**

Retained paired item outcomes, a prespecified comparison family and decision-specific task weights, not more selective headline examples.

## Implementation handoff

1. Freeze the eligible population, denominator, outcome horizon and proposed decision owner before comparison.
2. Reconcile assignment and outcome records at the independence unit; classify missingness without deleting cases.
3. Estimate the domain outcome and its uncertainty separately from activation. Apply risk and capacity vetoes before comparing costs.
4. Record approve, hold, or reverse with the evidence version and violated gate. An unmeasured required domain gate means HOLD.

The supplied cluster-budget grid is a planning sensitivity check with assumed ICC. It is not an estimate from real clusters and does not guarantee joint power. No stakeholder approval or real experiment is asserted.
