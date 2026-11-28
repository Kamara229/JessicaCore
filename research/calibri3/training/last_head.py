"""Exact last-position readout for the current prefix-only objective.
Never changes hidden states or causal attention; never used with model labels.
"""
from contextlib import contextmanager

@contextmanager
def last_position_head(model):
    counts={"calls":0,"input_positions":0,"computed_positions":0}
    def hook(m,args):
        h=args[0]
        assert h.ndim==3 and h.shape[-1]==320
        counts["calls"]+=1
        counts["input_positions"]+=h.shape[0]*h.shape[1]
        counts["computed_positions"]+=h.shape[0]
        return (h[:,-1:,:],*args[1:])
    handle=model.lm_head.register_forward_pre_hook(hook,prepend=True)
    try:yield counts
    finally:handle.remove()
