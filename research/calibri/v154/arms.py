"""No newly trained or serialized weights: subsets of the SAME paid archive.

The model always retains 11,292,174 bytes of original CB153 in input source
as a physically complete codebook. Each arm merely toggles operators in RAM
for diagnosis and does not declare any arm a separately measured physical
candidate or a new 0.7bit archive.
"""
from dataclasses import replace
from .protocol import MATRIX_KEYS,ARMS


def make_arms(physical):
    keys=tuple(MATRIX_KEYS)
    result={ARMS[0]:(),ARMS[1]:keys}
    for key in keys:
        stem=key.split(".")[-2]
        result[f"only_{stem}"]=(key,)
    for key in keys:
        stem=key.split(".")[-2]
        result[f"except_{stem}"]=tuple(x for x in keys if x!=key)
    if tuple(result)!=ARMS:raise ValueError("registered ablation operators missing")
    return result


def subset(physical,enabled):
    keys=set(enabled)
    if not keys.issubset(set(MATRIX_KEYS)):
        raise ValueError("unregistered teacher-owned source operator")
    return replace(physical,patches={k:physical.patches[k] for k in enabled})
