"""S126-Q physical codec: rank-4 shared interlayer Green Tunnel at 2 bits/weight.

RESEARCH ONLY. Runtime may reconstruct tiny 320x4+4x320 adapter matrices
from a physically packed 2-bit sidecar. Original large CB61 compressed
matrix source is untouched; FP32 matrix coefficients are NEVER serialized
in this candidate. Two globally shared FP16 scales only. Twelve layer
gates and twelve recurrent mix logits remain FP16. Original S111 340
prompt-dependent coefficients are frozen at EXACT zero, encoded by runtime
configuration, and do not exist in the physical sidecar as trainable weights.

Signed 4-code alphabet {-3,-1,+1,+3} times one FP16 matrix scalar.
After any training, reload this actual physical payload AND perform full
autoregressive 456 identity guard before accepting a candidate.
"""
from __future__ import annotations
import torch
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel

PREFIX="full_layer_tunnel.q2."
ENCODED_KEYS={
    PREFIX+"encoder_codes",PREFIX+"decoder_codes",
    PREFIX+"encoder_scale",PREFIX+"decoder_scale",
    PREFIX+"gates_fp16",PREFIX+"mix_logits_fp16",
}
DEPTH=12
WIDTH=320
RANK=4


def _encode_matrix_q2(weights:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    x=weights.detach().cpu().to(torch.float32).contiguous()
    if x.ndim!=2 or x.numel()%4!=0 or not bool(torch.isfinite(x).all()):
        raise ValueError("Q2 matrix needs a finite 2D multiple-of-four shape")
    maximum=float(x.abs().max().item())
    # Deterministic single scalar for this entire projection, persisted in FP16.
    if maximum>0:
        scale=torch.tensor([maximum/3.0],dtype=torch.float16)
    else:
        scale=torch.tensor([1.0],dtype=torch.float16)
    if float(scale.float()[0])<=0 or not bool(torch.isfinite(scale).all()):
        raise ValueError("Q2 scale not representable as FP16")
    normalized=x.flatten()/scale.float()[0]
    codes=torch.round((normalized+3.0)/2.0).clamp(0,3).to(torch.uint8)
    groups=codes.view(-1,4)
    packed=(
        groups[:,0].to(torch.int32) |
        (groups[:,1].to(torch.int32)<<2) |
        (groups[:,2].to(torch.int32)<<4) |
        (groups[:,3].to(torch.int32)<<6)
    ).to(torch.uint8).contiguous()
    return packed,scale


def _decode_matrix_q2(packed:torch.Tensor,scale:torch.Tensor,
                      shape:tuple[int,int])->torch.Tensor:
    if (packed.dtype!=torch.uint8 or packed.ndim!=1
            or packed.numel()!=shape[0]*shape[1]//4
            or scale.dtype!=torch.float16 or scale.shape!=(1,)
            or not bool(torch.isfinite(scale).all())
            or float(scale[0])<=0):
        raise ValueError("Invalid S126-Q packed 2-bit projection")
    v=packed.cpu().to(torch.int32)
    codes=torch.stack(tuple(((v>>(2*k))&3) for k in range(4)),dim=1)
    return ((codes.to(torch.float32)*2.0-3.0)*
            scale.cpu().float()[0]).reshape(shape).contiguous()


def export_compact_q2(tunnel:FullLayerStateTunnel)->dict[str,torch.Tensor]:
    if (tunnel.width,tunnel.depth,tunnel.rank)!=(WIDTH,DEPTH,RANK):
        raise ValueError("Only exact calibrated S126-Q rank4/width320/12 ABI")
    encoder_codes,encoder_scale=_encode_matrix_q2(tunnel.encoder)
    decoder_codes,decoder_scale=_encode_matrix_q2(tunnel.decoder)
    return {
        PREFIX+"encoder_codes":encoder_codes,
        PREFIX+"decoder_codes":decoder_codes,
        PREFIX+"encoder_scale":encoder_scale,
        PREFIX+"decoder_scale":decoder_scale,
        PREFIX+"gates_fp16":tunnel.gates.detach().cpu().half().contiguous(),
        PREFIX+"mix_logits_fp16":tunnel.mix_logits.detach().cpu().half().contiguous(),
    }


def load_compact_q2(tunnel:FullLayerStateTunnel,sidecar:dict[str,torch.Tensor]):
    if set(sidecar)!=ENCODED_KEYS:
        raise ValueError("S126-Q sidecar missing or unexpected physical tensor")
    if (tunnel.width,tunnel.depth,tunnel.rank)!=(WIDTH,DEPTH,RANK):
        raise ValueError("S126-Q geometry mismatch")
    if (sidecar[PREFIX+"gates_fp16"].shape!=(DEPTH,)
            or sidecar[PREFIX+"mix_logits_fp16"].shape!=(DEPTH,)
            or any(sidecar[PREFIX+k].dtype!=torch.float16
                   for k in ("gates_fp16","mix_logits_fp16"))
            or not all(bool(torch.isfinite(sidecar[PREFIX+k]).all())
                       for k in ("gates_fp16","mix_logits_fp16"))):
        raise ValueError("S126-Q gate scalar ABI corrupted")
    enc=_decode_matrix_q2(sidecar[PREFIX+"encoder_codes"],
        sidecar[PREFIX+"encoder_scale"],(RANK,WIDTH))
    dec=_decode_matrix_q2(sidecar[PREFIX+"decoder_codes"],
        sidecar[PREFIX+"decoder_scale"],(WIDTH,RANK))
    with torch.no_grad():
        for dest,x in ((tunnel.encoder,enc),(tunnel.decoder,dec),
                       (tunnel.gates,sidecar[PREFIX+"gates_fp16"]),
                       (tunnel.mix_logits,sidecar[PREFIX+"mix_logits_fp16"])):
            dest.copy_(x.to(device=dest.device,dtype=dest.dtype))


def physical_raw_bytes(sidecar:dict[str,torch.Tensor])->int:
    if set(sidecar)!=ENCODED_KEYS:
        raise ValueError("Incorrect physical sidecar")
    return sum(v.numel()*v.element_size() for v in sidecar.values())
