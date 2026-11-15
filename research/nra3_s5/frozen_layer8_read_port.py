"""NRA2 3.0 S5: strictly read-only post-layer8 Calibri TRUE 43M CPU capture.

No gold labels / NRA outputs / learned weights enter frozen Calibri inference.
Port is registered AFTER existing Green Tunnel / S103 bridge hooks are
installed in the genuine S93 prefilling phase. Port is absent during the
original 456+128+32 J7 generation, and returns None (PyTorch output identity).

This module only defines the native CPU read ABI; it NEVER changes LIVE,
Supabase, Render, APK, original CALIBRI, Green Tunnel, or sealed EVAL.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenLayer8Sample:
    prompt_sha256: str
    token_ids: tuple[int, ...]
    hidden320_fp16: torch.Tensor


class FrozenLayer8ReadPort:
    LAYER_INDEX=7

    def __init__(self):
        self.samples:dict[str,FrozenLayer8Sample]={}
        self.observed_calls=0
        self.output_gate=0.
        self.candidate_accepted=False
        self.read_only=True
        self._active=False

    @contextmanager
    def capture(self,model:nn.Module,prompt:str,expected_ids:list[int]):
        """One train-only prefill; hook is uninstalled immediately.

        Called *inside* frozen S160 CPU phase and after GT/S103 installation.
        Never active during old J7 generation or during candidate decoding.
        """
        layers=getattr(getattr(model,"model",None),"layers",None)
        if (layers is None or len(layers)!=12 or
                not isinstance(prompt,str) or not prompt or self._active or
                not isinstance(expected_ids,list) or len(expected_ids)<2 or
                not all(type(i) is int and i>=0 for i in expected_ids) or
                self.output_gate!=0. or self.candidate_accepted):
            raise ValueError("S5 requires exact unmodified 12-layer CPU Calibri ABI")
        digest=_sha(prompt)
        if digest in self.samples:
            raise ValueError("S5 duplicate TRAIN-only prompt")
        self._active=True
        entered=0
        result=None
        def hook(_layer,_args,output):
            nonlocal entered,result
            entered+=1
            if entered!=1 or self.output_gate!=0. or self.candidate_accepted:
                raise RuntimeError("S5 forbidden duplicate or write activation")
            h=output[0] if isinstance(output,tuple) else output
            if (not isinstance(h,torch.Tensor) or h.ndim!=3 or
                    h.shape!=(1,len(expected_ids),320) or h.device.type!="cpu" or
                    not torch.isfinite(h).all()):
                raise ValueError("S5 real original 43M layer8 hidden [1,T,320] ABI drift")
            result=h.detach()[0].to(torch.float16).contiguous().clone()
            if not torch.isfinite(result).all():
                raise ValueError("S5 nonfinite FP16 storage")
            return None  # same object; no residual, mutation or candidate logits
        handle=layers[self.LAYER_INDEX].register_forward_hook(hook)
        try:
            yield self
            if entered!=1 or result is None:
                raise RuntimeError("S5 expected exactly one actual CPU layer8 forward")
            self.samples[digest]=FrozenLayer8Sample(digest,tuple(expected_ids),result)
            self.observed_calls+=1
        finally:
            handle.remove()
            self._active=False

    def frozen_batch(self,ordered_prompts:tuple[str,...]):
        """Physically charge only observed tokens and lengths, never gold."""
        sha=[_sha(s) for s in ordered_prompts]
        if (not sha or len(set(sha))!=len(sha)
                or set(sha)!=set(self.samples)
                or self.observed_calls!=len(sha) or self._active
                or self.output_gate!=0. or self.candidate_accepted):
            raise AssertionError("S5 missing/duplicate true CPU TRAIN12 hidden")
        lengths=torch.tensor(
            [len(self.samples[k].token_ids) for k in sha],dtype=torch.int32)
        maximum=int(lengths.max())
        hidden=torch.zeros((len(sha),maximum,320),dtype=torch.float16)
        ids=torch.full((len(sha),maximum),-1,dtype=torch.int32)
        valid=torch.zeros((len(sha),maximum),dtype=torch.bool)
        for j,k in enumerate(sha):
            sample=self.samples[k]
            n=len(sample.token_ids)
            if sample.hidden320_fp16.shape!=(n,320):
                raise AssertionError("S5 token length mismatches captured hidden")
            hidden[j,:n]=sample.hidden320_fp16
            ids[j,:n]=torch.tensor(sample.token_ids,dtype=torch.int32)
            valid[j,:n]=True
        if (not torch.equal(valid.sum(1).int(),lengths)
                or not torch.all(ids[~valid]==-1)
                or not torch.all(hidden[~valid]==0)):
            raise AssertionError("S5 native CPU sequence padding leak")
        return {
            "token_hidden320":hidden.contiguous(),
            "input_ids":ids.contiguous(),
            "token_mask":valid.contiguous(),
            "token_lengths":lengths.contiguous(),
        }
