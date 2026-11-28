"""Composition of compact proposal and TRAIN-only transactional verification."""
from .proposals import propose
from .transaction import trial

def consider(source,x,dy,train_loss,clear_all,*,budget=16):
    # Restrict to the existing physical codebook width; never activate
    # unused padded centroid rows from the historical K32 container.
    hi=int(source.codes.max())
    width=next(k for k in (8,16,32) if hi<k)
    c,s=source.fields()
    proposals=propose(x,dy,c[:width],s,source.codes,source.shape,budget=budget)
    if not proposals:return {"accepted":False,"changes":0,"proposed":0}
    return {**trial(source,proposals,train_loss,clear_all),"proposed":len(proposals)}
