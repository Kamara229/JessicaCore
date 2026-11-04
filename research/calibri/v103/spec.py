"""Frozen genuinely labeled original public TRAIN slice, disjoint from v97.

This is a fresh diagnostic, NOT validation for selecting source-only
re-encoding and NOT the locked v89 independent TEST.
"""
from __future__ import annotations
BOOLQ_RANGE=(8780,8804)
ASSERTIONS=(
 ("arithmetic","What is sixteen minus three? Reply with only the number.","13"),
 ("arithmetic","What is 8 times 7? Reply with only the number.","56"),
 ("geography","What is the capital city of Switzerland? Reply with only the city.","Bern"),
 ("geography","What is the capital city of Portugal? Reply with only the city.","Lisbon"),
)
for domain,prompt,truth in ASSERTIONS:
    assert domain and prompt and truth
assert BOOLQ_RANGE[0]>8736 and BOOLQ_RANGE[1]<=9427
assert len({p for _,p,_ in ASSERTIONS})==len(ASSERTIONS)
