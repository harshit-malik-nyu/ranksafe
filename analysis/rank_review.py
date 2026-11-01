"""Build deterministic synthetic acceptance examples for ranking decisions."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ranksafe.score import ArmScore, ModelResult, rank_stability


def model(name,o,v,uo=0,uv=0):
    return ModelResult(name,ArmScore(100,o,uo),ArmScore(100,v,uv))

cases={
    'original_tie':[model('a',80,70),model('b',80,60)],
    'variant_tie':[model('a',80,70),model('b',70,70)],
    'mixed_ties_and_flips':[model('a',80,50),model('b',80,60),model('c',70,70)],
    'reverse_parser_asymmetry':[model('a',60,70,20,0),model('b',50,60)],
    'observed_order_preserved':[model('a',80,70),model('b',70,60)],
}
results={name:{'inputs':[{'model':r.model,'original':vars(r.original),'variant':vars(r.variant)} for r in rows],
               'result':rank_stability(rows)} for name,rows in cases.items()}
paths=['src/ranksafe/score.py','src/ranksafe/power.py','tests/test_rank_boundaries.py','analysis/rank_review.py']
payload={'scope':'Synthetic acceptance cases, not model performance or procurement evidence',
         'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'cases':results}
out=ROOT/'analysis/output/rank_review';out.mkdir(parents=True,exist_ok=True)
(out/'results.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
(out/'decision.md').write_text('''# A tie is not a preserved ordering

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
''')
