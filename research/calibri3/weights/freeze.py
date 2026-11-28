"""Explicit source freezing; no changes to codebooks, indices or scales."""
from contextlib import contextmanager
from v176.operators import TrainWeight

def all_sources(restored):
    model=restored[0][0]
    result={id(s):s for s in model.modules() if isinstance(s,TrainWeight)}
    for bank in restored[1:]:
        if isinstance(bank,dict):
            result.update({id(s):s for s in bank.values() if isinstance(s,TrainWeight)})
    result.update({id(s):s for s in restored[0][1].values()})
    return list(result.values())

@contextmanager
def freeze_sources(sources,selected,enabled):
    ids={id(p) for p in selected}
    params={id(p):p for s in sources for p in s.parameters()}
    flags={k:p.requires_grad for k,p in params.items()}
    for s in sources:s.clear()
    try:
        if enabled:
            for k,p in params.items():
                if k not in ids:p.requires_grad_(False)
        yield {"source_count":len(sources),"source_parameter_tensors":len(params),
               "requires_grad_tensors":sum(p.requires_grad for p in params.values())}
    finally:
        for s in sources:s.clear()
        for k,p in params.items():p.requires_grad_(flags[k])
