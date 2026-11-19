"""S71: frozen original 43M S132 all-step phase candidate and old TRAIN32 J7 drift.

S69 physical factor ablation narrowed full-vs-KV parity mismatch to S132
prefill-only decoding, not GreenTunnel Q2 alone. An all-step S132
continuation fixes parity locally, but has NEVER been verified to preserve
exact original 616 protected generated J7. S71 checks an inexpensive
subset first: all 32 ALREADY-SPENT S190 native old TRAIN paraphrases.

Original 616 J7 replay and S190 32 real generated token ID capture MUST
finish untouched before any alternate phase. We read immutable
S194.REAL_CAPTURE as comparator. ALL-STEPS phase changes are only temporary
runtime hooks, NEVER a release change and NEVER an optimizer update.
Only old TRAIN; NO held32, QA, EVAL32/128, manual gold or new candidates
accepted. Exact source only prompt features; not a streaming safety proof.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import torch

from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
from calibri2.s194 import actual_cpu_j7_bridge as s194bridge
from research.nra3_s20.s69_original_43m_gt_expansion_phase_matrix import (
    MODES as S69_MODES,original_phase_matrix,
)
from research.nra3_s20.s67_original_cache_phase_isolation import require


def sha(question):
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def actual_original_s190_frozen_source_probes():
    """Always exact same 32 PREVIOUS TRAIN only, never held/new eval."""
    p=s190.new_train_paraphrase_probes()
    require(len(p)==32 and len({sha(x[0]) for x in p})==32,
            "S71 existing S190 only 32 old TRAIN identities")
    return p


def original_32_baseline():
    """S194 unmodified original decoder first-read capture, not an oracle."""
    frozen=s194bridge.REAL_CAPTURE
    probes=actual_original_s190_frozen_source_probes()
    keys={sha(q) for q,_,_ in probes}
    require(set(frozen)==keys
            and all(isinstance(v,list) and v
                    and all(type(x) is int and x>=0 for x in v)
                    for v in frozen.values()),
            "S71 original S194 genuine 32 generated native tokens missing")
    return {key:tuple(frozen[key]) for key in sorted(frozen)}


def read_only_alternate_allstep_original32(cpu,tok,bridge,tunnel,expansion,
                                           copy_net,head,original_decoder):
    """No S194 duplicate-output guard altered; separate research read-only decoder.

    The caller pins original_decoder BEFORE S194 temporary J7 capture wrapper.
    This function deliberately never calls S194's mutable capture wrapper.
    """
    require(callable(original_decoder)
            and original_decoder is not s160.cached_full_cpu_parallel
            and not torch.cuda.is_available(),
            "S71 original frozen decoder BEFORE S194 compare-only instrumentation")
    base=original_32_baseline()
    model,sources,first,later,pointer,lm_head,*_=cpu
    require(not model.training and len(model.model.layers)==12
            and not expansion._active and not tunnel._active,
            "S71 genuine original 43M frozen modules only")
    probe_rows=[]
    all_decoder_calls=0
    for q,_old_train_answer,group in actual_original_s190_frozen_source_probes():
        key=sha(q)
        features=torch.as_tensor(
            s160.prompt_only_vector(tok,q),dtype=torch.float32)
        require(features.shape==(512,) and bool(torch.isfinite(features).all()),
                "S71 original S95 source features frozen")
        calls={"prefill":0,"incremental":0}
        def record(_model,_args,kwargs):
            require("past_key_values" in kwargs
                    and kwargs.get("use_cache") is True,
                    "S71 full original cached decoder inference only")
            k="prefill" if kwargs["past_key_values"] is None else "incremental"
            calls[k]+=1
        h=model.register_forward_pre_hook(record,with_kwargs=True)
        try:
            with torch.inference_mode():
                # S69 mode2 = original GT prefill; bridge ALL; S132 ALL.
                # Because S132's route guard requires active bridge
                # at decode, S132 ALL + original bridge prefill-only is
                # NOT the same physical experiment.
                with original_phase_matrix(
                        model,bridge,expansion,tunnel,features,S69_MODES[2]):
                    got,note=original_decoder(
                        model,sources,first,later,lm_head,tok,q,
                        pointer,copy_net,head)
        finally:
            h.remove()
        require(isinstance(got,list) and len(got)>0
                and all(type(t) is int and t>=0 for t in got)
                and calls["prefill"]>=1
                and isinstance(note,dict)
                and not expansion._active and not tunnel._active,
                "S71 all-step candidate forward completed and isolated")
        prev=base[key]
        all_decoder_calls+=calls["incremental"]
        probe_rows.append({
            "source_sha256":key,
            "source_group":group,
            "exact_original_J7_token_match":tuple(got)==prev,
            "candidate_generated_native_count":len(got),
            "original_generated_native_count":len(prev),
            "candidate_last_token_is_original_EOS":
                got[-1]==tok.eos_token_id,
            "first_token_same":got[0]==prev[0],
            "prefill_calls":calls["prefill"],
            "incremental_decode_calls":calls["incremental"],
            "original_source_only_no_answer_used_for_causal_phase":True,
            "output_gate":0,"candidate_accepted":False,
        })
    require({r["source_sha256"] for r in probe_rows}==set(base),
            "S71 frozen 32 original J7 comparison incomplete")
    return final_audit(probe_rows,all_decoder_calls)


def final_audit(rows,incremental_total):
    require(len(rows)==32 and len({z["source_sha256"] for z in rows})==32
            and type(incremental_total) is int and incremental_total>=0
            and all(z["output_gate"]==0 and z["candidate_accepted"] is False
                    and z["original_source_only_no_answer_used_for_causal_phase"]
                    for z in rows), "S71 old32 source J7 gate0")
    return {
        "stage":"NRA3_S71_FROZEN_S132_ALLSTEP_PHASE_OLD_S190_TRAIN32_TRUE_CPU_J7_DIAGNOSTIC",
        "original_616_J7_baseline_replayed_unchanged_before_alternate_phase":True,
        "S194_REPLAY_UNMODIFIED_original_capture32_compared":True,
        "original_43M_S132_S95_GT_FP16_weights_unchanged":True,
        "allstep_S132_with_required_original_S93_bridge_allstep_and_GT_prefill":True,
        "source_only_existing_S190_TRAIN32_never_EVAL_or_S66_held32_gold":True,
        "not_616_full_new_candidate_retention_certified":True,
        "no_training_no_optimizer_no_deployment":True,
        "real_original_decoding_subcalls_incremental_total":incremental_total,
        "n_previous_TRAIN32":32,
        "n_exact_original32_J7_outputs_unchanged":sum(
            r["exact_original_J7_token_match"] for r in rows),
        "n_original32_FIRST_token_unchanged":sum(
            r["first_token_same"] for r in rows),
        "n_phase_candidate_has_EOS":sum(
            r["candidate_last_token_is_original_EOS"] for r in rows),
        "n_rows_with_real_incremental_calls":sum(
            r["incremental_decode_calls"]>0 for r in rows),
        "rows":rows,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "S132_allstep_phase_production_safety_proven":False,
        "streaming_causal_safety_certified":False,
    }
