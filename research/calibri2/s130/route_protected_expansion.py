"""CALIBRI S130: route-isolated 39.82M additive core; research only.

The 36.38M original core and all S101/S126-Q weights stay unchanged.
New residual parameters are added by S129. During this controlled trial
new weight effects are limited to prefill of the original S93 training route,
the only route containing the selected S122 paired lessons. The same new
weights have no effect on S87/S91 routes. Cached continuation bypasses all
the new residual hooks; no cross-request mutable recurrent state is retained.
"""
from __future__ import annotations

from calibri2.s129.additive_core_expansion import AdditiveCoreExpansion


class RouteIsolatedAdditiveCore(AdditiveCoreExpansion):
    def bind_original_route_bridge(self,bridge):
        # Avoid registering the existing frozen bridge as new core parameters.
        object.__setattr__(self,"_bridge_for_route_only",bridge)
        return self

    def _layer_hook(self,index):
        normal=super()._layer_hook(index)
        def protected(_layer,_inputs,output):
            if (self._active
                and getattr(self,"_bridge_for_route_only",None) is not None
                and getattr(self._bridge_for_route_only,"_active_profile",None)!="S93"):
                if len(self._visited)!=index:
                    raise AssertionError("S130 frozen route decoder order changed")
                self._visited.append(index)
                return output  # exactly identical original output on other routes
            return normal(_layer,_inputs,output)
        return protected
