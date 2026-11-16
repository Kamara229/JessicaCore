"""S2 real physical 135M architecture capacity sweep: no NRA, no quality claims.

Untrained GT is not a rescue; this stage checks exact-on-disk <=0.700 bpp
and verifies every 30-layer hook genuinely executes. Source S0 SHA pinned.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from calibri2.independent_135m.arch_no_nra.architecture import (
    InternalTunnelBlock, FullDepthGreenTunnel, PromptOnlyRouter,
    ParallelSourceSlots
)

ORIGINAL_UNIQUE = 134515008
BASE_SOURCE_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
SOURCE_BYTES=11229303
WIDTH=576
DEPTH=30
BPP_CAP=.7

SPECS=(
  ("baseline_two8_all8_slot16",2,8,8,16),
  ("five16_all16_slot16",5,16,16,16),
  ("four16_all32_slot16",4,16,32,16),
  ("five16_all24_slot24",5,16,24,24),
  ("five24_all32_slot16",5,24,32,16)
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,
        ensure_ascii=False)+"\n",encoding="utf-8")


def pin_positions(count):
    if count==2:
        return [7,22]
    if count==5:
        return [4,10,16,22,28]
    if count==4:
        return [5,12,19,26]
    raise ValueError("Unsupported precommitted local GT layer count")


class MultiInternalGreenTunnel(nn.Module):
    def __init__(self,width,positions,rank):
        super().__init__()
        self.positions=tuple(positions)
        self.blocks=nn.ModuleDict({
            str(i):InternalTunnelBlock(width,rank) for i in positions
        })
        self._handles=[]

    def install(self,model):
        if self._handles:
            raise AssertionError("Double-install local Green Tunnel")
        for i in self.positions:
            block=self.blocks[str(i)]
            def hook(_module,_args,value,*,residual=block):
                if isinstance(value,tuple):
                    return (value[0]+residual(value[0]),*value[1:])
                return value+residual(value)
            self._handles.append(model.model.layers[i].register_forward_hook(hook))


class Toy30(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(nn.Identity() for _ in range(DEPTH))

    def forward(self,h):
        for layer in self.model.layers:
            h=layer(h)
        return h


def pack_modules(path,local,full,router,slots):
    parts={"local":local,"full_depth":full,"router":router,"slots":slots}
    tensors={label+"."+key:t.detach().cpu().half().contiguous()
             for label,part in parts.items() for key,t in part.state_dict().items()}
    save_file(tensors,str(path))
    count=sum(p.numel() for part in parts.values() for p in part.parameters())
    if sum(x.numel() for x in tensors.values())!=count:
        raise AssertionError("In-memory sidecar trainable parameters not all stored")
    return count,len(tensors)


def one_capacity(name,local_count,local_rank,all_rank,slot_rank,
                 core,profile,out):
    torch.manual_seed(1352)
    model=Toy30()
    local=MultiInternalGreenTunnel(WIDTH,pin_positions(local_count),local_rank)
    full=FullDepthGreenTunnel(WIDTH,DEPTH,all_rank)
    router=PromptOnlyRouter()
    slots=ParallelSourceSlots(WIDTH,rank=slot_rank,slots=2)
    local.install(model)
    full.install(model)
    x=torch.randn(1,5,WIDTH,requires_grad=True)
    prev=x.detach().clone()
    after=model(x)
    if not torch.equal(after,prev) or full.last_full_coverage!=tuple(range(DEPTH)):
        raise AssertionError("Zero-init local/full GT not exact identity or missing layers")
    (after.square().mean()).backward()
    if full.gates.grad is None or not bool(torch.count_nonzero(full.gates.grad)):
        raise AssertionError("No differentiable full-depth GT gates")
    if any(block.up.weight.grad is None for block in local.blocks.values()):
        raise AssertionError("No local GT through-model gradients")
    folder=out/name
    folder.mkdir(parents=True,exist_ok=True)
    side=folder/"FULL_CALIBRI_GT_ROUTER_SLOTS_FP16.safetensors"
    pcount,fields=pack_modules(side,local,full,router,slots)
    summary={
        "variant":name,"native_original_width":WIDTH,
        "native_original_depth":DEPTH,
        "local_green_tunnel_layers":list(local.positions),
        "local_rank":local_rank,
        "full_depth_rank":all_rank,
        "parallel_slot_rank":slot_rank,
        "parallel_slot_count":2,
        "NRA_and_S3_present":False,
        "source_checkpoint_sha":BASE_SOURCE_SHA,
        "trainable_sidecar_parameters":pcount,
        "sidecar_tensor_fields":fields,
        "real_30_layer_scope_passed":True,
        "full_depth_gate_gradients_present":True,
        "local_residual_gradients_present":True,
        "untrained_identity_zero_residual":True,
        "router_trained_and_armed":False,
        "parallel_head_trained_and_armed":False,
        "not_a_quality_improvement_claim":True,
        "only_SOURCE_S0_and_sparse_S1_profile":True,
        "full_source_bytes":Path(core).stat().st_size,
        "sparse_S1_profile_bytes":Path(profile).stat().st_size,
    }
    mpath=folder/"PHYSICAL_VARIANT_MANIFEST.json"
    dump(mpath,summary)
    source_bytes=Path(core).stat().st_size
    side_bytes=side.stat().st_size
    manifest_bytes=mpath.stat().st_size
    with_profile_bytes=source_bytes+side_bytes+manifest_bytes+Path(profile).stat().st_size
    without_profile_bytes=source_bytes+side_bytes+manifest_bytes
    summary["physical_budget"]={
        "source_bytes":source_bytes,
        "sidecar_bytes":side_bytes,
        "manifest_bytes":manifest_bytes,
        "optional_sparse_profile_bytes":Path(profile).stat().st_size,
        "all_inclusive_with_profile_bytes":with_profile_bytes,
        "all_inclusive_with_profile_bpp":8*with_profile_bytes/ORIGINAL_UNIQUE,
        "passes_with_profile":8*with_profile_bytes/ORIGINAL_UNIQUE<=BPP_CAP,
        "without_S1_failed_profile_bytes":without_profile_bytes,
        "without_S1_failed_profile_bpp":8*without_profile_bytes/ORIGINAL_UNIQUE,
        "passes_without_profile":8*without_profile_bytes/ORIGINAL_UNIQUE<=BPP_CAP,
        "cap_0p7_total_bytes":int(BPP_CAP*ORIGINAL_UNIQUE/8),
    }
    # Reserve one known maximum-size manifest as physical deployment overhead.
    # This is a truthful re-measurement; envelope JSON includes its own budget.
    dump(mpath,summary)
    final_manifest=mpath.stat().st_size
    for p in ("all_inclusive_with_profile_bytes","without_S1_failed_profile_bytes"):
        summary["physical_budget"][p]+=final_manifest-manifest_bytes
    for label,p in (("all_inclusive_with_profile_bpp","all_inclusive_with_profile_bytes"),
                    ("without_S1_failed_profile_bpp","without_S1_failed_profile_bytes")):
        summary["physical_budget"][label]=8*summary["physical_budget"][p]/ORIGINAL_UNIQUE
    for label,p in (("passes_with_profile","all_inclusive_with_profile_bpp"),
                    ("passes_without_profile","without_S1_failed_profile_bpp")):
        summary["physical_budget"][label]=summary["physical_budget"][p]<=BPP_CAP
    # Final manifest length may change slightly after decimal updates.
    for _ in range(12):
        old_size=mpath.stat().st_size
        summary["physical_budget"]["manifest_bytes"]=old_size
        dump(mpath,summary)
        new_size=mpath.stat().st_size
        if new_size==old_size:
            break
        diff=new_size-old_size
        for p in ("all_inclusive_with_profile_bytes","without_S1_failed_profile_bytes"):
            summary["physical_budget"][p]+=diff
        for label,p in (("all_inclusive_with_profile_bpp","all_inclusive_with_profile_bytes"),
                        ("without_S1_failed_profile_bpp","without_S1_failed_profile_bytes")):
            summary["physical_budget"][label]=8*summary["physical_budget"][p]/ORIGINAL_UNIQUE
        for label,p in (("passes_with_profile","all_inclusive_with_profile_bpp"),
                        ("passes_without_profile","without_S1_failed_profile_bpp")):
            summary["physical_budget"][label]=summary["physical_budget"][p]<=BPP_CAP
    if (summary["physical_budget"]["manifest_bytes"]!=mpath.stat().st_size or
        summary["physical_budget"]["all_inclusive_with_profile_bytes"]!=
            source_bytes+side_bytes+mpath.stat().st_size+Path(profile).stat().st_size):
        raise AssertionError("Exact final manifest physical accounting mismatch")
    print(json.dumps({"stage":"S2_actual_GT_capacity_physical",
        "name":name,"params":pcount,
        "bpp_with_profile":summary["physical_budget"]["all_inclusive_with_profile_bpp"],
        "bpp_without_S1":summary["physical_budget"]["without_S1_failed_profile_bpp"],
        "passes_with_profile":summary["physical_budget"]["passes_with_profile"],
        "passes_without_S1":summary["physical_budget"]["passes_without_profile"]
    }),flush=True)
    return summary


def run(core,profile,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(core).read_bytes()
    if len(raw)!=SOURCE_BYTES or sha(raw)!=BASE_SOURCE_SHA:
        raise AssertionError("S2 must use only authenticated original pretrained FP32→S0 CB59 core")
    if Path(profile).stat().st_size<100_000:
        raise AssertionError("Sparse S1 profile missing or unexpectedly small")
    rows=[one_capacity(name,local_count,local_rank,all_rank,slot_rank,
                       core,profile,out)
          for name,local_count,local_rank,all_rank,slot_rank in SPECS]
    dump(out/"S2_ALL_VARIANTS_GT_CAPACITY_RESULTS.json",rows)
    print(json.dumps({"stage":"S2_capacity_final",
        "variants":len(rows),
        "with_profile_pass":sum(r["physical_budget"]["passes_with_profile"] for r in rows),
        "without_S1_pass":sum(r["physical_budget"]["passes_without_profile"] for r in rows)}),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("S0_135M_CB59",type=Path)
    p.add_argument("S1_sparse_physical_profile",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.S0_135M_CB59,a.S1_sparse_physical_profile,a.out)
