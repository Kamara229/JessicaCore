"""NRA2 S3E: read-only full contextual token sequence of genuine 43M CPU layer12.

Extends the original last-token S3C collector without changing its hook:
an independent second hook captures the exact same frozen layer output
across ALL prefill token positions. No labels, gold, decoder or LIVE.
"""
from __future__ import annotations

import torch
from research.nra2_bridge.frozen_features import FrozenFinalFeatures


class FrozenAllTokenFeatures(FrozenFinalFeatures):
    def __init__(self):
        super().__init__()
        self._all_handle=None
        self.token_sequences={}
        self.token_calls=0

    def install(self,model):
        super().install(model)
        def capture_all(_layer,_args,output):
            if self._active_prompt is None:
                return None
            key=self._active_prompt
            if key in self.token_sequences:
                return None
            h=output[0] if isinstance(output,tuple) else output
            if (not isinstance(h,torch.Tensor) or h.ndim!=3
                    or h.shape[0]!=1 or h.shape[-1]!=320
                    or h.shape[1]<2 or h.device.type!="cpu"
                    or not torch.isfinite(h).all()):
                raise AssertionError("S3E all-token CPU layer12 port requires finite [1,T>1,320]")
            self.token_sequences[key]=h.detach()[0].float().cpu().to(torch.float16).clone()
            self.token_calls+=1
            return None
        self._all_handle=model.model.layers[-1].register_forward_hook(capture_all)
        return self

    def select_tokens(self,ordered_sha):
        if (len(ordered_sha)!=len(set(ordered_sha))
                or set(ordered_sha)!=set(self.token_sequences)):
            raise AssertionError("S3E 48 ordered unique real prefill tokens mismatch")
        lengths=torch.tensor([len(self.token_sequences[k]) for k in ordered_sha],
                             dtype=torch.int32)
        maximum=int(lengths.max().item())
        tokens=torch.zeros((len(ordered_sha),maximum,320),dtype=torch.float16)
        mask=torch.zeros((len(ordered_sha),maximum),dtype=torch.bool)
        for i,k in enumerate(ordered_sha):
            n=int(lengths[i])
            tokens[i,:n]=self.token_sequences[k]
            mask[i,:n]=True
        if (torch.any(lengths<2) or not torch.isfinite(tokens).all()
                or not torch.equal(mask.sum(dim=1).to(torch.int32),lengths)
                or self.token_calls!=len(ordered_sha)):
            raise AssertionError("S3E token padding/length or duplicate capture")
        return tokens,mask,lengths

    def uninstall(self):
        if self._all_handle is not None:
            self._all_handle.remove()
            self._all_handle=None
        super().uninstall()
