"""S115 read-only temporal gating for S111/S112 Green Tunnel across KV steps.

S112 learned prompt-conditioned S93 dual GT gates, gained 1 full correct,
lost 7 full previously correct, gained one first-token correctness and
lost ZERO previously correct FIRST tokens. Hypothesis: some old complete
answers are damaged by applying the shared GT in cached continuation/EOS
steps rather than during the prompt prefill that determines first token.

Expose a contextmanager that enables the SAME already learned FP16
coefficients only at chosen stages of the genuine S101 KV-cache inference.
No new neural weights, no model training, no external QA; mask metadata
must be counted physically and every candidate checked on 456 original
authorized TRAIN full+EOS outputs with zero old forgetting.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import torch

VALID_PHASES = (
    "all_steps",
    "prefill_only",
    "prefill_and_second",
    "decode_only",
    "second_only",
)


def phase_enabled(phase: str, step: int) -> bool:
    if phase not in VALID_PHASES or step < 0:
        raise ValueError("Unknown decoder phase mask or negative step")
    if phase == "all_steps": return True
    if phase == "prefill_only": return step == 0
    if phase == "prefill_and_second": return step <= 1
    if phase == "decode_only": return step > 0
    return step == 1


@contextmanager
def frozen_cpu_phase_scope(
    model, bridge, frozen_profile: str | None, frozen_s95_features,
    *, phase: str,
) -> Iterator[dict]:
    """Execute exactly one S101 cached autoregressive rollout with step mask.

    Decoder pre-hook sees the original model.forward invocation before
    decoder-layer Green Tunnel hooks. First model call is prompt prefill;
    later calls carry a non-None KV cache and exactly one new token.
    There is NO answer-aware selection; the mask and frozen S95 route
    are selected before generating a single token.
    """
    if phase not in VALID_PHASES:
        raise ValueError("Not a registered causal phase ablation")
    if not getattr(bridge, "_handles", None):
        raise RuntimeError("Install frozen dual-GT bridge first")
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("S115 phase probes are inference-only")
    if frozen_profile == bridge.allowed_profile and frozen_s95_features is None:
        raise ValueError("S93 phase study needs original S95 prompt vector")
    state={"calls":0,"phase":phase,"route":frozen_profile}
    with bridge.prompt_scope(frozen_profile,frozen_s95_features):
        def pre_hook(_model,args,kwargs):
            step=state["calls"]
            cached=kwargs.get("past_key_values")
            if (step == 0 and cached is not None) or (
                    step > 0 and cached is None):
                raise AssertionError("S101 cached decoder phase ABI changed")
            if kwargs.get("use_cache") is not True:
                raise AssertionError("S115 must audit original KV-cache inference")
            if frozen_profile==bridge.allowed_profile and phase_enabled(phase,step):
                bridge._active_profile=bridge.allowed_profile
            else:
                bridge._active_profile=None
            state["calls"]+=1
            return None
        handle=model.register_forward_pre_hook(pre_hook,with_kwargs=True)
        try:
            yield state
        finally:
            handle.remove()
            bridge._active_profile=frozen_profile
    if state["calls"] < 1:
        raise AssertionError("S115 mask yielded no S101 model forward")
