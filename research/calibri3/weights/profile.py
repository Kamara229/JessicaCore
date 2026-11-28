"""Count real legacy weight reconstructions without GPU synchronization per call."""
from contextlib import contextmanager
from time import perf_counter
from v176.operators import TrainWeight

@contextmanager
def profile_legacy():
    original=TrainWeight.forward;clear=TrainWeight.clear
    stats={"calls":0,"cache_misses":0,"logical_values_reconstructed":0,"clear_calls":0,"clear_nonempty":0,"host_dispatch_seconds":0.}
    def forward(self):
        stats["calls"]+=1
        if self.cached is None:
            stats["cache_misses"]+=1
            stats["logical_values_reconstructed"]+=self.shape[0]*self.shape[1]
        t=perf_counter()
        try:return original(self)
        finally:stats["host_dispatch_seconds"]+=perf_counter()-t
    def reset(self):
        stats["clear_calls"]+=1
        stats["clear_nonempty"]+=int(self.cached is not None)
        return clear(self)
    TrainWeight.forward=forward;TrainWeight.clear=reset
    try:yield stats
    finally:TrainWeight.forward=original;TrainWeight.clear=clear
