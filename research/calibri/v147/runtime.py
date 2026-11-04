"""No extra full-output GEMV: fuse half-gain CB140 and int8 vocab bias."""
import numpy as np
import torch
from direct_runtime_v59 import PackedEmbedding,PackedLinear
from v143.runtime import FusedKernel,PackedFusedHead,build


def fused_half_head(model,paid,folder,bias=None):
    input_layer=model.model.embed_tokens
    old_head=model.lm_head
    if not isinstance(input_layer,PackedEmbedding) or not isinstance(old_head,PackedLinear):
        raise ValueError("not original native 5-bit model")
    if input_layer.handle is not old_head.handle:
        raise ValueError("original pretrained weights not physically tied")
    if old_head.bias is not None:
        raise ValueError("unexpected non-zero original head bias")
    native=FusedKernel(build(folder/"calibri147_fused_output.so"),
                       old_head.handle,paid)
    # Gain=0.5 applied to just 512 runtime FP32 center values, not the source.
    native.delta*=np.float32(.5)
    if bias is not None:
        b=np.asarray(bias,dtype=np.float32)
        if b.shape!=(49152,) or not np.isfinite(b).all():
            raise ValueError("corrupted paid vocab bias")
    else:
        b=None
    new_head=PackedFusedHead(old_head.handle,native,b)
    model.lm_head=new_head
    if model.model.embed_tokens.handle is not new_head.handle:
        raise ValueError("half-head broke source codebook tying")
    if any(p.ndim>=2 for p in model.parameters()):
        raise ValueError("dense original pretrained weights reappeared")
    return new_head
