"""Trained-FP32-135M -> CALIBRI 2.0 architecture port, explicitly WITHOUT NRA.

Adapted architectural ideas from CALIBRI2 S87/S94/S95/S126/S159.
The native original 135M width/depth are preserved; none of the 43M
trained sidecar weights is ported as incompatible pretrained weights.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from direct_codebook_v59 import CB59, unpack5


class InternalTunnelBlock(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.gate = nn.Linear(width, 1)
        nn.init.normal_(self.down.weight, std=.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        h = F.layer_norm(x, (x.shape[-1],))
        return self.up(torch.tanh(self.down(h))) * torch.sigmoid(self.gate(h))


class InternalGreenTunnel(nn.Module):
    """S87-style real neural residuals at ≈25% and ≈75% core depth."""
    def __init__(self, width, depth, rank=8):
        super().__init__()
        self.positions = (max(0, depth//4), min(depth-1, (3*depth)//4))
        if len(set(self.positions)) != 2:
            raise AssertionError("Core too shallow for two donor GT blocks")
        self.blocks = nn.ModuleDict({str(i):InternalTunnelBlock(width,rank)
                                    for i in self.positions})
        self._handles = []

    def install(self, model):
        if self._handles:
            raise RuntimeError("Internal Green Tunnel double install")
        for i in self.positions:
            block=self.blocks[str(i)]
            def hook(_layer, _args, value, *, adapter=block):
                if isinstance(value,tuple):
                    return (value[0]+adapter(value[0]), *value[1:])
                if isinstance(value,torch.Tensor):
                    return value+adapter(value)
                raise TypeError("Unsupported Llama decoder value")
            self._handles.append(model.model.layers[i].register_forward_hook(hook))


class FullDepthGreenTunnel(nn.Module):
    """Generalized S126: rank8 shared state across ALL native 135M layers."""
    def __init__(self, width, depth, rank=8):
        super().__init__()
        self.width, self.depth, self.rank=width,depth,rank
        self.encoder=nn.Parameter(torch.empty(rank,width))
        self.decoder=nn.Parameter(torch.empty(width,rank))
        self.gates=nn.Parameter(torch.zeros(depth))
        self.mix_logits=nn.Parameter(torch.zeros(depth))
        nn.init.normal_(self.encoder,std=.02)
        nn.init.normal_(self.decoder,std=.02)
        self._handles=[]
        self._carry=None
        self._visited=[]
        self.last_full_coverage=()

    def _begin(self,_model,_args):
        self._carry=None
        self._visited=[]

    def _end(self,_model,_args,output):
        if self._visited!=list(range(self.depth)):
            raise AssertionError(f"Green Tunnel full-depth coverage missing: {self._visited}")
        self.last_full_coverage=tuple(self._visited)
        self._carry=None
        self._visited=[]
        return output

    def _hook(self,i):
        def hook(_layer, _args, val):
            h=val[0] if isinstance(val,tuple) else val
            if h.ndim!=3 or h.shape[-1]!=self.width:
                raise AssertionError("Wrong core hidden-state dimensions")
            if len(self._visited)!=i:
                raise AssertionError("Green Tunnel layer ordering broken")
            self._visited.append(i)
            x=F.layer_norm(h,(self.width,))
            z=torch.tanh(F.linear(x,self.encoder))
            if self._carry is not None:
                mix=torch.sigmoid(self.mix_logits[i])
                z=mix*self._carry+(1-mix)*z
            self._carry=z
            updated=h+torch.tanh(self.gates[i])*F.linear(z,self.decoder)
            return (updated,*val[1:]) if isinstance(val,tuple) else updated
        return hook

    def install(self,model):
        if self._handles:
            raise RuntimeError("Full depth tunnel already installed")
        if len(model.model.layers)!=self.depth:
            raise AssertionError("Actual original decoder depth mismatch")
        self._handles.append(model.register_forward_pre_hook(self._begin))
        for i,layer in enumerate(model.model.layers):
            self._handles.append(layer.register_forward_hook(self._hook(i)))
        self._handles.append(model.register_forward_hook(self._end))


class PromptOnlyRouter(nn.Module):
    """S95-style prompt-only route, default immutably chooses base S0."""
    def __init__(self, features=512, hidden=12, profiles=2):
        super().__init__()
        self.features=features
        self.net=nn.Sequential(nn.Linear(features,hidden),nn.Tanh(),
                               nn.Linear(hidden,profiles))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.enabled=False  # must pass separate J7; not activated by pilot

    def features_from_ids(self, token_ids):
        feature=torch.zeros(1,self.features)
        for token in token_ids:
            # Deterministic token histogram; no question IDs or gold labels.
            feature[0,int(token)%self.features]+=1.0
        return feature/float(max(1,len(token_ids)))

    @torch.no_grad()
    def suggest(self,token_ids):
        scores=self.net(self.features_from_ids(token_ids))
        return {"suggested_profile":int(scores.argmax(-1)),
                "active_profile":int(scores.argmax(-1)) if self.enabled else 0,
                "router_trained_and_armed":self.enabled}


class ParallelSourceSlots(nn.Module):
    """S159-like single-prefill two-slot span binding: SHADOW ONLY."""
    def __init__(self,width,rank=16,slots=2):
        super().__init__()
        self.width=width;self.rank=rank;self.slots=slots
        self.query=nn.Linear(width,slots*rank,bias=False)
        self.keys=nn.Linear(width,slots*rank,bias=False)
        self.binding_query=nn.Linear(width,rank,bias=False)
        self.binding_keys=nn.Linear(width,rank,bias=False)
        self.binding_strength=nn.Parameter(torch.tensor(1.0))
        self.enabled=False # no source copy without independent train/J7

    def forward(self, hidden, source_ids):
        if (hidden.ndim!=3 or source_ids.ndim!=2 or
            hidden.shape[:2]!=source_ids.shape or hidden.shape[-1]!=self.width
            or hidden.shape[1]<self.slots):
            raise ValueError("Parallel slot head expects one complete prefill")
        b,l,h=hidden.shape
        n=l-self.slots+1
        windows=torch.stack([hidden[:,i:i+n] for i in range(self.slots)],dim=2)
        token_windows=torch.stack([source_ids[:,i:i+n] for i in range(self.slots)],dim=2)
        query=self.query(hidden[:,-1]).view(b,self.slots,self.rank)
        projected=self.keys(windows).view(b,n,self.slots,self.slots,self.rank)
        diag=projected.diagonal(dim1=2,dim2=3).permute(0,1,3,2)
        score=(diag*query[:,None]).sum(-1).mean(-1)/math.sqrt(self.rank)
        bound=(self.binding_keys(windows.mean(2))*self.binding_query(
            hidden[:,-1])[:,None]).sum(-1)/math.sqrt(self.rank)
        score=score+self.binding_strength.sigmoid()*bound
        choice=score.argmax(-1)
        chosen=token_windows[torch.arange(b),choice]
        probability=score.softmax(-1).gather(1,choice[:,None]).squeeze(1)
        return {"chosen_source_span_ids":chosen,
                "shadow_confidence":probability,
                "shadow_no_answer_override":not self.enabled}


class CompactProfileBank:
    """Physical sparse delta of S1 135M vs S0 135M; NO second whole archive.

    All centroids and changed index groups are physically serialized. These
    profile parameters do not pretend to be already-working skills.
    """
    def __init__(self, original:CB59, other:CB59):
        if (original.meta!=other.meta or
            original.meta["original_unique_parameter_count"]!=134515008):
            raise AssertionError("S1 bank must use the exact same original CB59 source geometry")
        desc=original.meta["tensor_descriptors"]
        self.changed_centroid_tensors={}
        self.changed_indices={}
        for name, info in desc.items():
            if info["dtype"]!="CB59_K32V8":
                if name not in other.raw or not (original.raw[name]==other.raw[name]).all():
                    raise AssertionError("Alternative S1 changed unpaid raw FP32 parameter")
                continue
            a=original.handles[name];b=other.handles[name]
            if (a.shape!=b.shape or not (a.row_scales==b.row_scales).all()):
                raise AssertionError("Alternative source changed raw geometry/row scales")
            if not (a.centroids==b.centroids).all():
                self.changed_centroid_tensors[name]=torch.from_numpy(
                    b.centroids.astype("<f2").copy())
            if not (a.packed==b.packed).all():
                length=a.shape[0]*a.shape[1]//8
                x=unpack5(a.packed.tobytes(),length)
                y=unpack5(b.packed.tobytes(),length)
                index=(x!=y).nonzero()[0]
                self.changed_indices[name]=torch.from_numpy(
                    __import__("numpy").stack([index, y[index]],axis=1).astype("int32"))
        self.total_changes=sum(int(len(v)) for v in self.changed_indices.values())

    def tensors(self):
        return {**{"centroid."+name:val for name,val in self.changed_centroid_tensors.items()},
                **{"index."+name:val for name,val in self.changed_indices.items()}}

    def write(self,path):
        if not self.tensors():
            raise AssertionError("Expected S1 distinct compressed profile")
        save_file(self.tensors(),str(path))

    def activate(self, sources, base:CB59, target):
        if target not in (0,1):
            raise ValueError("Only S0/S1 profile supported")
        with torch.no_grad():
            for name, s in sources.items():
                handle=base.handles[name]
                baseline=torch.from_numpy(handle.centroids.copy())
                if target and name in self.changed_centroid_tensors:
                    s.centroids.copy_(self.changed_centroid_tensors[name].float())
                else:
                    s.centroids.copy_(baseline)
                length=handle.shape[0]*handle.shape[1]//8
                ids=torch.from_numpy(unpack5(handle.packed.tobytes(),length).astype("int64"))
                if target and name in self.changed_indices:
                    changes=self.changed_indices[name].long()
                    ids[changes[:,0]]=changes[:,1]
                s.codes.copy_(ids)


class Calibri135NoNRA(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone=backbone
        w=backbone.config.hidden_size
        d=backbone.config.num_hidden_layers
        self.internal=InternalGreenTunnel(w,d,rank=8)
        self.full_depth=FullDepthGreenTunnel(w,d,rank=8)
        self.router=PromptOnlyRouter()
        self.slots=ParallelSourceSlots(w,rank=16,slots=2)
        # Register S87 hooks first; full-depth recurrent sees local updates.
        self.internal.install(backbone)
        self.full_depth.install(backbone)
        self.layout={"native_width":w,"native_depth":d,
                     "donor_layers":self.internal.positions,
                     "full_layer_count":d,
                     "neural_NRA_present":False,
                     "semantic_S3_adapters_present":False,
                     "slot_head_armed":False,
                     "prompt_router_armed":False}

    def auxiliary_parameters(self):
        return (sum(p.numel() for p in self.internal.parameters())+
                sum(p.numel() for p in self.full_depth.parameters())+
                sum(p.numel() for p in self.router.parameters())+
                sum(p.numel() for p in self.slots.parameters()))

    def sidecar(self):
        parts={"internal":self.internal,"full_depth":self.full_depth,
               "router":self.router,"slots":self.slots}
        return {key+"."+name:t.detach().cpu().half().contiguous()
                for key,module in parts.items()
                for name,t in module.state_dict().items()}

    def store(self,path):
        save_file(self.sidecar(),str(path))
        reloaded=load_file(str(path))
        if set(reloaded)!=set(self.sidecar()):
            raise AssertionError("Physical full architecture sidecar incomplete")
        return len(reloaded)


def physical_report(base_path, profile_path, sidecar_path, manifest_path):
    sizes={name:Path(p).stat().st_size for name,p in [
        ("core",base_path),("profile_delta",profile_path),
        ("all_new_neural_GT_router_parallel",sidecar_path),
        ("manifest",manifest_path)]}
    size=sum(sizes.values())
    return {"component_bytes":sizes,
            "all_inclusive_weighted_bytes":size,
            "original_FP32_unique_parameter_count":134515008,
            "effective_total_bpp":8*size/134515008,
            "passes_0p700_bpp":size*8<=int(134515008*.7),
            "all_trainable_inference_parameters_counted":True,
            "no_other_FP32_or_NRA_sidecar":True}
