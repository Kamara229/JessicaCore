"""Family/depth-balanced reusable Green Tunnel matrix schedule.

Every original *unique* compressed source operator receives at least
one opportunity to change its physical discrete source indices PER
recovery cycle. All paid codebook centroids can still receive global
whole-model gradient in every step. No v186-specific 96-operator subset.

This scheduler does NOT assume layers must be called 'model.layers';
a model/codec adapter provides family and depth for each source item.
MoE experts are independent canonical operators, never deduplicated
without verified tied weight aliases.
"""
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

@dataclass(frozen=True)
class Operator:
    canonical_weight_name: str
    family: str
    depth: int
    original_nparams: int
    def __post_init__(self):
        if not self.canonical_weight_name or not self.family:
            raise ValueError("Source operator/adapter family required")
        if self.depth<0 or self.original_nparams<=0:
            raise ValueError("Source operator geometry invalid")


def plan_cycles(operators:Iterable[Operator],cycles:int,
                start_cycle:int=0)->tuple[tuple[str,...], ...]:
    """Braid families at each depth, complete unique-source coverage."""
    op=list(operators)
    if not 1<=cycles<=32 or start_cycle<0:
        raise ValueError("Finite positive recovery cycles required")
    if not op:
        raise ValueError("Empty original paid source operator set")
    ids=[o.canonical_weight_name for o in op]
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate source operator: tied aliases must be canonicalized")
    # Source families in deterministic order. Braid family/depth
    # rather than first training all attention and only then MLP.
    depth_groups=defaultdict(list)
    for x in op:
        depth_groups[x.depth].append(x)
    depths=sorted(depth_groups)
    base=[]
    for depth in depths:
        base.extend(sorted(depth_groups[depth],
                           key=lambda o:(o.family,o.canonical_weight_name)))
    assert set(x.canonical_weight_name for x in base)==set(ids)
    out=[]
    for cycle in range(start_cycle,start_cycle+cycles):
        # Rotate depth order across cycles to prevent persistent
        # first-layer/last-layer optimization scheduling advantage.
        shift=cycle%len(depths)
        ordered_depths=depths[shift:]+depths[:shift]
        all_names=tuple(x.canonical_weight_name
                        for depth in ordered_depths
                        for x in sorted(depth_groups[depth],
                            key=lambda op:(op.family,op.canonical_weight_name)))
        if len(all_names)!=len(op) or set(all_names)!=set(ids):
            raise AssertionError("A model layer lost source update coverage")
        out.append(all_names)
    return tuple(out)
