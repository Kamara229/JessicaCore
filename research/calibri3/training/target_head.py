"""Training-only readout at required target position, not every prefix token."""
from contextlib import contextmanager

@contextmanager
def last_position_readout(model):
    # Existing native() and native_batch() capture [:,-1:] and consume only
    # final logits. With no labels, slicing the head input before its linear
    # projection is mathematically equivalent. Never apply to full-sequence
    # loss, generation consumers requiring all logits, or head probes.
    counts={"calls":0,"input_positions":0,"projected_positions":0}
    def reduce(_module,args):
        h=args[0]
        assert h.ndim==3 and h.shape[1]>=1
        counts["calls"]+=1
        counts["input_positions"]+=h.shape[0]*h.shape[1]
        counts["projected_positions"]+=h.shape[0]
        return (h[:,-1:,:],*args[1:])
    handle=model.lm_head.register_forward_pre_hook(reduce)
    try:yield counts
    finally:handle.remove()
