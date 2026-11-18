"""S67: source-only ORIGINAL Calibri43M 12-layer / GT phase / KV cache ablation.

Uses ONLY previously spent S66 raw token IDs and the S66 physical model,
AFTER original 616 J7 baseline generation. No training, no gold, no update,
no model mutation. Diagnostic only; NOT evidence of active online causality.
A fixed full-prompt S95 vector can contain future-source information.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager

import torch
from v176.operators import clear

MODES=("core_only_gt_disabled", "s93_original_prefill_only",
       "s93_bridge_all_steps_tunnel_prefill_only")


def require(condition, reason):
    if not bool(condition):
        raise ValueError("S67 fail-closed original cache ablation: "+reason)


def cache_length(cache):
    require(cache is not None, "missing KV cache")
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    require(isinstance(cache,(tuple,list)) and len(cache)>0,
            "unknown KV type")
    return int(cache[0][0].shape[-2])


def delta(a,b):
    require(isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
            and a.shape==b.shape and a.numel()>0, "differing tensor shape")
    x=(a.float()-b.float()).abs()
    require(bool(torch.isfinite(x).all()), "nonfinite metric")
    return {"max_abs":float(x.max().item()),
            "mean_abs":float(x.mean().item()),
            "exact_fp16_or_logits":bool(torch.equal(a,b))}


@contextmanager
def _phase(model,bridge,expansion,tunnel,prompt_vec,mode):
    require(mode in MODES, "unknown phase mode")
    with ExitStack() as stack:
        if mode=="core_only_gt_disabled":
            # S93 compressed core and its frozen centroid route remain intact.
            # Disable BOTH prompt bridge and installed 12-layer GT residual.
            stack.enter_context(bridge.prompt_scope(None,None))
            stack.enter_context(tunnel.scope(False))
        else:
            stack.enter_context(expansion.prefill_only(model))
            stack.enter_context(__import__(
                "calibri2.s127.q2_prefill_gate_control",
                fromlist=["prefill_only_tunnel"]).prefill_only_tunnel(model,tunnel))
            from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope
            phase=("prefill_only" if mode=="s93_original_prefill_only"
                   else "all_steps")
            stack.enter_context(frozen_cpu_phase_scope(
                model,bridge,"S93",prompt_vec,phase=phase))
        yield


def measured_native_triplet(model,sources,bridge,expansion,tunnel,
                            prompt_vec,token_ids,cut,mode):
    """Compare full, full-prefix→KV2, and full-prefix→KV1→KV1.

    All three use the SAME immutable original native IDs. KV sequence lengths
    and H8/logits are read, not altered. No human labels or adapter outputs.
    """
    require(mode in MODES and len(model.model.layers)==12
            and not model.training and next(model.parameters()).device.type=="cpu",
            "12-layer original frozen CPU ABI")
    require(type(cut) is int and 4<=cut<=len(token_ids)-3
            and len(token_ids)>9
            and all(type(i) is int and i>=0 for i in token_ids),
            "native cut, ID or input length")
    require(not tunnel._active, "preexisting GT activation")
    observed=[]
    def hook(_layer,_args,output):
        h=output[0] if isinstance(output,tuple) else output
        require(isinstance(h,torch.Tensor) and h.ndim==3
                and h.shape[0]==1 and h.shape[-1]==320
                and bool(torch.isfinite(h).all()), "native H8 ABI")
        observed.append(h.detach()[0].to(torch.float16).clone())
        return None
    handle=model.model.layers[7].register_forward_hook(hook)
    def ids_tensor(v):
        return torch.tensor([v],dtype=torch.long)
    def run(v,cache):
        clear(sources)
        out=model(input_ids=ids_tensor(v),past_key_values=cache,
                  use_cache=True,output_hidden_states=True)
        require(out.past_key_values is not None
                and out.logits.shape[:2]==(1,len(v)), "native logits and KV")
        return out
    try:
        with torch.inference_mode():
            with _phase(model,bridge,expansion,tunnel,prompt_vec,mode):
                full=run(token_ids,None)
                full_logits=full.logits[0,cut:cut+2].detach().clone()
                full_at_cut_minus_1=full.logits[0,cut-1].detach().clone()
                require(cache_length(full.past_key_values)==len(token_ids),
                        "full forward cache length")
                del full
            with _phase(model,bridge,expansion,tunnel,prompt_vec,mode):
                pre=run(token_ids[:cut],None)
                first_logits=pre.logits[0,-1].detach().clone()
                cache=pre.past_key_values
                pre_len=cache_length(cache)
                require(pre_len==cut, "prefix cache length/position")
                del pre
                two=run(token_ids[cut:cut+2],cache)
                kv2_logits=two.logits[0].detach().clone()
                after_kv2=cache_length(two.past_key_values)
                require(after_kv2==cut+2, "KV2 cache position")
                del two,cache
            with _phase(model,bridge,expansion,tunnel,prompt_vec,mode):
                pre_again=run(token_ids[:cut],None)
                cache=pre_again.past_key_values
                require(cache_length(cache)==cut,"second-prefix position")
                del pre_again
                one=run([token_ids[cut]],cache)
                after_one=cache_length(one.past_key_values)
                one_logits=one.logits[0,0].detach().clone()
                require(after_one==cut+1,"KV1 position")
                cache=one.past_key_values
                del one
                next_one=run([token_ids[cut+1]],cache)
                after_next=cache_length(next_one.past_key_values)
                next_logits=next_one.logits[0,0].detach().clone()
                require(after_next==cut+2,"KV1+1 position")
                del next_one,cache
    finally:
        handle.remove()
    require(len(observed)==6, "six original physical layer8 forwards")
    lengths=(len(token_ids),cut,2,cut,1,1)
    require(tuple(x.shape[0] for x in observed)==lengths,
            "H8 forward token coverage")
    kv1plus1=torch.cat((observed[4],observed[5]),dim=0)
    split_logits=torch.stack((one_logits,next_logits),dim=0)
    return {
        "mode":mode, "native_token_count":len(token_ids),"cut":cut,
        "cache_position_trace":[pre_len,after_kv2,cut,after_one,after_next],
        "full_vs_prefix_H8_cut_minus_1":
            delta(observed[0][cut-1],observed[1][cut-1]),
        "full_vs_KV2_H8_next2":
            delta(observed[0][cut:cut+2],observed[2]),
        "KV2_vs_KV1plus1_H8_next2":
            delta(observed[2],kv1plus1),
        "full_vs_KV2_logits_next2":delta(full_logits,kv2_logits),
        "KV2_vs_KV1plus1_logits_next2":delta(kv2_logits,split_logits),
        "full_vs_prefix_nexttoken_logits":
            delta(full_at_cut_minus_1,first_logits),
        "frozen_full_text_prompt_vector_even_on_prefix":mode!="core_only_gt_disabled",
        "future_conditioning_not_ruled_out":True,
        "normal_online_generation_certified":False,
        "no_gradient_or_adapter_or_gold_or_generation_activation":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }


def diagnostic_result(rows, protected_616=True):
    require(protected_616 and len(rows)==8
            and all(set(x)==set(MODES) for x in rows), "complete source-only rows")
    for row in rows:
        for name,result in row.items():
            require(name==result["mode"] and result["output_gate"]==0
                    and result["candidate_accepted"] is False
                    and result["activation_allowed"] is False
                    and result["normal_online_generation_certified"] is False,
                    "fail-closed candidate")
    return {
        "stage":"NRA3_S67_GENUINE_ORIGINAL43M_GT_H8_KV_PHASE_ABLATION",
        "on_previously_spent_s66_raw_only":True,
        "616_original_baseline_J7_verified_BEFORE_capture":True,
        "root_cause_not_assigned_by_phase_probe":True,
        "no_new_adapter_parameters":True,
        "online_causality_not_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "rows":rows,
    }
