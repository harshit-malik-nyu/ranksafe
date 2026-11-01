import pytest
from ranksafe.score import ArmScore, ModelResult, rank_stability


def model(name,o,v,uo=0,uv=0,n=100):
    return ModelResult(name,ArmScore(n,o,uo),ArmScore(n,v,uv))


def test_ties_are_not_evidence_of_preserved_order():
    for rows in [[model('a',80,70),model('b',80,60)],
                 [model('a',80,70),model('b',70,70)]]:
        r=rank_stability(rows)
        assert r['pairs_considered']==0
        assert r['flip_rate'] is None
        assert 'Tied pairs were excluded' in r['verdict']


def test_mixed_ties_keep_the_correct_pair_denominator():
    r=rank_stability([model('a',80,50),model('b',80,60),model('c',70,70)])
    assert r['pairs_tied_original']==1
    assert r['pairs_considered']==2
    assert r['flips']==2
    assert r['flip_rate']==1


def test_parser_gate_is_symmetric():
    rows=[model('a',60,70,20,0),model('b',50,60)]
    swapped=[ModelResult(r.model,r.variant,r.original) for r in rows]
    assert 'Unusable' in rank_stability(rows)['verdict']
    assert 'Unusable' in rank_stability(swapped)['verdict']


@pytest.mark.parametrize('rows',[
    [model('a',0,0,n=0),model('b',0,0,n=0)],
    [model('a',80,70),model('b',70,60,n=99)],
    [model('a',80,70),model('a',70,60)],
    [model('a',80,70,21),model('b',70,60)],
    [model('a',-1,70),model('b',70,60)],
])
def test_invalid_evaluation_contract_is_rejected(rows):
    with pytest.raises(ValueError):rank_stability(rows)


@pytest.mark.parametrize('margin',[-1,2,float('nan')])
def test_invalid_margin_is_rejected(margin):
    with pytest.raises(ValueError):rank_stability([model('a',80,70),model('b',70,60)],min_margin=margin)
