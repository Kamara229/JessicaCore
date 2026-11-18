"""S69 actual 43M original GT vs 896-rank expansion decode-phase factor isolation.

S67 physically observed full-vs-KV2 H8 maxabs >1e-3 in 8/8 while
core+auxiliaries disabled had 1/8 (quantization at ~0.00195). S67's
control DISABLED the 43M S132 expansion as well as the GT, so it cannot
attribute root cause to Green Tunnel alone. S69 never changes the original
paid FP16 weights and observes the effects of activating the ORIGINAL
already-learned two residual modules on *cached decode steps*.

S68 separately proved original S95 feature algorithm uses complete future
text even for prefix features; this stage fixes the SAME full-text vector
for matched physical ablations, so it still DOES NOT certify online streaming.
No S66 held gold or adapter inference, no tuning/gradient/selector updates.
"""
from __future__ import annotations

from contextlib import contextmanager,ExitStack
import hashlib
import json

import torch
from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from research.nra3_s20.s67_original_cache_phase_isolation import (
    measured_native_triplet,require as physical_require,
)

MODES=(
    "s93_bridge_all_tunnel_all_expansion_all",
    "s93_bridge_all_tunnel_all_expansion_prefill",
    "s93_bridge_all_tunnel_prefill_expansion_all",
)
# Three modes vary ONLY whether frozen original full-layer Q2 GT and/or
# frozen original S132 expansion are active at incremental steps. Bridge
# route active in all three, so S132 route guard cannot silently bypass.
assert len(set(MODES))==3


@contextmanager
def frozen_original_expansion_all_steps(model,expansion):
    """Only flip S129 already-installed runtime _active flag; never weights.

    In contrast to S129.prefill_only, this is on for every forward with
    explicit KV and use_cache. It restores previous state on all exits.
    """
    physical_require(bool(expansion._handles)
                     and not expansion._active and not model.training
                     and torch.is_inference_mode_enabled(),
                     "S132 exact frozen CPU all-step ablation ABI")
    old=expansion._active
    visits=[]
    def pre(_model,args,kwargs):
        physical_require(not expansion._active, "leaked original expansion flag")
        physical_require("past_key_values" in kwargs
                         and kwargs.get("use_cache") is True,
                         "S132 requires original explicit KV use_cache")
        expansion._active=True
        visits.append(kwargs["past_key_values"] is None)
    def post(_model,args,kwargs,result):
        expansion._active=False
        return result
    hpre=model.register_forward_pre_hook(pre,with_kwargs=True)
    hpost=model.register_forward_hook(post,with_kwargs=True,always_call=True)
    try:
        yield visits
    finally:
        hpost.remove()
        hpre.remove()
        expansion._active=old
        expansion._visited=[]


@contextmanager
def original_phase_matrix(model,bridge,expansion,tunnel,prompt_vec,mode):
    physical_require(mode in MODES and not model.training
                     and torch.is_inference_mode_enabled(),
                     "S69 strictly CPU frozen phase matrix")
    # Order mirrors original S66 hooks: first progressive S132 prehook,
    # second S126 tunnel, third S115 bridge route enabling S132 route guard.
    with ExitStack() as stack:
        exp_all=(mode in (MODES[0],MODES[2]))
        tunnel_all=(mode in (MODES[0],MODES[1]))
        if exp_all:
            stack.enter_context(frozen_original_expansion_all_steps(
                model,expansion))
        else:
            stack.enter_context(expansion.prefill_only(model))
        if tunnel_all:
            stack.enter_context(tunnel.scope(True))
        else:
            stack.enter_context(prefill_only_tunnel(model,tunnel))
        state=stack.enter_context(frozen_cpu_phase_scope(
            model,bridge,"S93",prompt_vec,phase="all_steps"))
        yield
    physical_require(state["calls"]>=1 and not expansion._active
                     and not tunnel._active,
                     "S69 leaked core/GT active state")


def physical_spent_row(model,sources,bridge,expansion,tunnel,
                       prompt_vec,token_ids,cut):
    """All 3 ablations read exactly SAME 12-layer original model/IDs."""
    outs={}
    for mode in MODES:
        outs[mode]=measured_native_triplet(
            model,sources,bridge,expansion,tunnel,
            prompt_vec,token_ids,cut,mode,
            phase_handler=original_phase_matrix)
    physical_require(set(outs)==set(MODES)
                     and all(x["cache_position_trace"]==
                             [cut,cut+2,cut,cut+1,cut+2]
                             for x in outs.values()),
                     "S69 preserved native cache positions")
    return outs


def final_audit(rows):
    physical_require(len(rows)==8 and
                     all(set(x)==set(MODES) for x in rows),
                     "S69 all8 × all3 physical source only")
    for r in rows:
        for mode,d in r.items():
            physical_require(mode==d["mode"] and
                             d["output_gate"]==0 and
                             d["candidate_accepted"] is False and
                             d["activation_allowed"] is False and
                             d["normal_online_generation_certified"] is False,
                             "S69 research only gate")
    def n_mismatch(mode,key,tol=1.e-3):
        return sum(z[mode][key]["max_abs"]>tol for z in rows)
    return {
        "stage":"NRA3_S69_REAL43M_GENUINE_GT_S132_EXPANSION_KV_DECODE_PHASE_FACTOR_MATRIX",
        "original_model_and_GT_and_expansion_weights_unchanged":True,
        "original_616_baseline_J7_physically_replayed_first":True,
        "only_S66_previously_spent_native_inputs_no_gold":True,
        "s67_bare_core_also_disabled_progressive_expansion_so_not_sole_GT_attribution":True,
        "S95_full_source_features_still_not_causal_prefix_safe":True,
        "original_runtime_phase_only_ablation_unevaluated_for_new_J7":True,
        "root_cause_not_single_module_certified":True,
        "n_rows":len(rows),
        "mode_h8_KV2_mismatch_gt_1e3":{
            m:n_mismatch(m,"full_vs_KV2_H8_next2") for m in MODES},
        "mode_logits_KV2_mismatch_gt_1e3":{
            m:n_mismatch(m,"full_vs_KV2_logits_next2") for m in MODES},
        "mode_logits_KV2_vs_KV1plus1_mismatch_gt_1e3":{
            m:n_mismatch(m,"KV2_vs_KV1plus1_logits_next2") for m in MODES},
        "rows":rows,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "online_causality_certified":False,
    }
