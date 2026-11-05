"""Original input second moments from historical public TRAIN text only."""
import torch
from paid_lowrank_v76 import get_tokens, WINDOW
from .protocol import FIT_CORPORA, calibration_key


def prepare_text():
    windows, provenance = [], {}
    for name in FIT_CORPORA:
        ids, origin = get_tokens(name)
        if origin["split"] != "train" or len(ids) % WINDOW:
            raise ValueError("Calibration must be complete public TRAIN windows")
        windows.extend(ids[i:i+WINDOW].clone() for i in range(0, len(ids), WINDOW))
        provenance[name] = origin
    return windows, provenance


@torch.inference_mode()
def collect(model, archive, windows):
    keys = {calibration_key(n) for n, d in archive.meta["tensor_descriptors"].items()
            if d["dtype"] == "CB59_K32V8"}
    if len(keys) != 121:
        raise ValueError("Unexpected full model calibration coverage")
    moments, counts, hooks = {}, {}, []
    try:
        for name in sorted(keys):
            module = model.get_submodule(name)
            def capture(mod, args, name=name):
                x = args[0].detach().reshape(-1, mod.in_features).double()
                h = x.T @ x
                if name in moments:
                    moments[name].add_(h)
                    counts[name] += len(x)
                else:
                    moments[name] = h
                    counts[name] = len(x)
            hooks.append(module.register_forward_pre_hook(capture))
        for ids in windows:
            model(input_ids=ids[None], use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if set(moments) != keys:
        raise ValueError("Unobserved original operator")
    return {k: (v / counts[k]).numpy() for k, v in moments.items()}, counts
