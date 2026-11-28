"""Scoped, bounded cache of reconstructed matrices for frozen sources only.
Training optimization, not a change to the physical model format.
"""
from collections import OrderedDict
from contextlib import contextmanager
import torch


def signature(source):
    fields=(source.centroids,source.log_scales,source.codes,source.base_scales)
    return tuple((id(t),t.data_ptr(),t._version,str(t.device),t.dtype,t.requires_grad) for t in fields)


@contextmanager
def frozen_cache(source_cls,sources,byte_cap=640*1024**2):
    assert byte_cap>=0
    original=source_cls.forward;clear=source_cls.clear
    tracked={id(s):s for s in sources}
    cache=OrderedDict()
    stats=dict(hits=0,misses=0,bypasses=0,invalidations=0,evictions=0,
               avoided_clear_calls=0,reconstructed_values=0,resident_bytes=0,peak_bytes=0)
    def drop(key):
        if key in cache:
            _,_,n=cache.pop(key);stats['resident_bytes']-=n
    def forward(self):
        key=id(self)
        if key not in tracked or any(p.requires_grad for p in self.parameters()):
            if key in cache:
                drop(key);stats['invalidations']+=1
            stats['bypasses']+=1
            return original(self)
        stamp=signature(self)
        if key in cache:
            old,value,_=cache[key]
            if old==stamp:
                cache.move_to_end(key);stats['hits']+=1
                return value
            drop(key);stats['invalidations']+=1
        stats['misses']+=1
        clear(self)
        with torch.no_grad():value=original(self).detach()
        clear(self)
        assert not value.requires_grad and value.grad_fn is None
        stats['reconstructed_values']+=value.numel()
        n=value.numel()*value.element_size()
        if n<=byte_cap:
            while cache and stats['resident_bytes']+n>byte_cap:
                drop(next(iter(cache)));stats['evictions']+=1
            cache[key]=(stamp,value,n);stats['resident_bytes']+=n
            stats['peak_bytes']=max(stats['peak_bytes'],stats['resident_bytes'])
        return value
    def reset(self):
        # The old cache still clears on every call. Frozen matrices live in
        # a distinct bounded cache and are version-checked at each access.
        if id(self) in cache:stats['avoided_clear_calls']+=1
        return clear(self)
    for s in tracked.values():clear(s)
    source_cls.forward=forward;source_cls.clear=reset
    try:yield stats
    finally:
        source_cls.forward=original;source_cls.clear=clear
        cache.clear();stats['resident_bytes']=0
        for s in tracked.values():clear(s)
