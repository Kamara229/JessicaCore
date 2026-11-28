"""Move all real model banks and legacy hook modules, reusing audited G44 ABI."""
import torch
from types import MethodType
from research.calibri2.fusion_growth.g35_full_mps_preflight import native_device_mask

def move(restored,l6,slot,balance,device):
    model,sources,bridge,full,expansion,child,_=restored[0]
    mid=child.child.child
    mid.delta_components=MethodType(native_device_mask,mid)
    donor=bridge._existing()
    modules=[model,bridge,full,expansion,child,l6,slot,balance]
    if donor is not None:modules.append(donor)
    for bank in restored[1:]:
        if isinstance(bank,dict):modules.extend(v for v in bank.values() if isinstance(v,torch.nn.Module))
        elif isinstance(bank,torch.nn.Module):modules.append(bank)
    # Legacy donor can be alive only through hook defaults, not bridge weakref.
    # Include the exact bound module objects from installed hooks.
    for owner in list(model.modules()):
        for hook in owner._forward_hooks.values():
            for obj in list(getattr(hook,"__defaults__",None) or ()) + list((getattr(hook,"__kwdefaults__",None) or {}).values()):
                if isinstance(obj,torch.nn.Module):modules.append(obj)
            for cell in getattr(hook,"__closure__",None) or ():
                obj=cell.cell_contents
                if isinstance(obj,torch.nn.Module):modules.append(obj)
            bound=getattr(hook,"__self__",None)
            if isinstance(bound,torch.nn.Module):modules.append(bound)
    modules=list({id(m):m for m in modules}.values())
    for m in modules:m.to(device)
    for s in sources.values():s.clear();s.to(device)
    assert all(p.device.type==device for m in modules for p in m.parameters())
    assert all(b.device.type==device for m in modules for b in m.buffers())

    return modules
