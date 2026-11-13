"""NRA2 Stage 3A: genuine frozen Calibri final-layer contextual feature tap.

Read ONLY: first prefill token context after decoder layer 12.
A question SHA tags a feature row for TRAIN-only offline adapter research.
No answer, gold label, decoder change, hidden-state mutation, forward rewrite,
standalone trained NRA modification, QA/TEST, or LIVE access.

Stage3A is acquisition, NOT proof that the frozen standalone graph-trained NRA
can understand these 320D vectors. An alignment adapter must be TRAINED later.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Iterator
import torch
from torch import nn

WIDTH=320
DEPTH=12


def prompt_sha(question: str) -> str:
    if not isinstance(question,str) or not question:
        raise ValueError("Training prompt must be nonempty string")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


class FrozenFinalFeatures:
    """Capture first contextual prefill state for each tagged prompt SHA."""

    def __init__(self):
        self._handle=None
        self._active_prompt=None
        self.features:dict[str,torch.Tensor]={}
        self.forward_calls=0
        self.observed_tagged_calls=0
        self.unlabeled_capture_only=True
        self.read_layer=11

    @contextmanager
    def prompt_scope(self, question:str)->Iterator[None]:
        if self._active_prompt is not None:
            raise RuntimeError("Nested NRA2 feature tag would cross-contaminate examples")
        key=prompt_sha(question)
        self._active_prompt=key
        try:
            yield
        finally:
            self._active_prompt=None

    def install(self, model:nn.Module):
        if self._handle is not None:
            raise RuntimeError("Duplicate final-layer feature collector")
        if (not hasattr(model,"model") or not hasattr(model.model,"layers")
                or len(model.model.layers)!=DEPTH):
            raise AssertionError("Expected frozen Calibri 12-layer CPU decoder")

        def first_prefill(_layer,_args,output):
            self.forward_calls+=1
            if self._active_prompt is None:
                return None
            self.observed_tagged_calls+=1
            key=self._active_prompt
            if key in self.features:
                return None  # step1/step2 of same cached CPU generation
            h=output[0] if isinstance(output,tuple) else output
            if (not isinstance(h,torch.Tensor) or h.ndim!=3 or h.shape[0]!=1
                    or h.shape[-1]!=WIDTH or h.shape[1]<1
                    or h.device.type!="cpu" or not torch.isfinite(h).all()):
                raise AssertionError("NRA2 requires finite genuine [1,T,320] CPU state")
            self.features[key]=h.detach()[0,-1,:].float().cpu().to(torch.float16).clone()
            # Returning None is the original forward's exact output object.
            return None

        self._handle=model.model.layers[-1].register_forward_hook(first_prefill)
        return self

    def uninstall(self):
        if self._active_prompt is not None:
            raise RuntimeError("Cannot remove a collector with active input scope")
        if self._handle is not None:
            self._handle.remove()
            self._handle=None

    def select(self, ordered_train_hashes:list[str]) -> torch.Tensor:
        if (not ordered_train_hashes or len(set(ordered_train_hashes))!=len(ordered_train_hashes)
                or any(len(k)!=64 for k in ordered_train_hashes)):
            raise ValueError("Unique 64-byte SHA256 prompt identity list required")
        missing=set(ordered_train_hashes)-set(self.features)
        if missing:
            raise AssertionError(f"NRA2 missing {len(missing)} real CPU TRAIN hidden states")
        output=torch.stack([self.features[key] for key in ordered_train_hashes])
        if (output.shape!=(len(ordered_train_hashes),WIDTH)
                or output.dtype!=torch.float16
                or not torch.isfinite(output).all()):
            raise AssertionError("Physical NRA2 contextual features invalid")
        if len(ordered_train_hashes)>1 and int(torch.unique(output,dim=0).shape[0])!=len(ordered_train_hashes):
            raise AssertionError("NRA2 contextual features collapsed or duplicate")
        return output
