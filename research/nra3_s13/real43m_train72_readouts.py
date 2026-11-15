"""S13: 72 TRUE original Calibri43M CPU Russian TRAIN sentences only.

Train ONLY newly physically billed rank8 FP16 relation-conditioned
UPOS/role/observed-past-head readout. Prior original 6,106,154 packed
CB59 matrices and trained 39-FP16 S11 router are SHA-frozen/unaltered.
This is training fit, NOT independent EVAL or proven language improvement.

CRITICAL: S12c first heldout16 was consumed. This script MUST NEVER
import/use that 16, any earlier five consumed EVAL128, or sealed S3CI.
At word i, causal predictions are computed from original NRA features
AT THE WORD'S LAST TOKEN, without adding hidden of a later final
prefix (a previously misleading S12c metric construction).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config,UPOS,SEMANTIC_ROLES
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.real_ru72_source import (
    checked_real_train72,S10_FEATURE_SHA,S10_CAPTURE_RUN,
)
from research.nra3_s13.paid_relation_feedback import (
    PaidRelationFeedback,PARAMETERS,PHYSICAL_BYTES,
)

S8B_43_CB59_SHA="d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97"
S11_39_FP16_SHA="05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"
S8B_RUN=43731085988
S11_RUN=43734571081


def require(ok:bool,why:str):
    if not bool(ok):
        raise ValueError("NRA3 S13 authentic causal feedback TRAIN72 "+why)


@dataclass(frozen=True)
class FixedNative72:
    relation:torch.Tensor           # [72,T,128]
    upos:torch.Tensor               # [72,T,17]
    role:torch.Tensor               # [72,T,9]
    dep:torch.Tensor                # [72,T,T]
    valid:torch.Tensor              # [72,T]
    upos_gold:torch.Tensor          # [72,T] -100 ignore
    role_gold:torch.Tensor          # [72,T] -100 ignore
    dep_gold:torch.Tensor           # [72,T] -100 ignore
    candidate_heads:torch.Tensor    # [72,T,T] only words <= current
    num_words:int
    num_past_head_arcs:int


def gather_native_fixed_training(samples,native:NRA3NativePackedCPU)->FixedNative72:
    """Exactly 72 FULL original model physical native CPU forwards ONCE.

    Gold is used only in the output-side TRAIN masks after calling
    original unmodified native NRA3; no native model receives gold.
    """
    require(len(samples)==72,"S9 SHA-locked 72 TRAIN source ONLY")
    lengths=[int(x.hidden320.shape[0]) for x in samples]
    max_t=max(lengths)
    n=len(samples)
    rel=torch.zeros(n,max_t,128,dtype=torch.float32)
    up=torch.zeros(n,max_t,len(UPOS),dtype=torch.float32)
    role=torch.zeros(n,max_t,len(SEMANTIC_ROLES),dtype=torch.float32)
    dep=torch.full((n,max_t,max_t),-1e4,dtype=torch.float32)
    valid=torch.zeros(n,max_t,dtype=torch.bool)
    up_gold=torch.full((n,max_t),-100,dtype=torch.long)
    role_gold=torch.full((n,max_t),-100,dtype=torch.long)
    dep_gold=torch.full((n,max_t),-100,dtype=torch.long)
    candidates=torch.zeros(n,max_t,max_t,dtype=torch.bool)
    arcs=0; words=0
    with torch.inference_mode():
        for k,s in enumerate(samples):
            l=lengths[k]
            f=native(s.hidden320.unsqueeze(0))
            require(f.relations.shape==(1,l,128) and
                    f.upos_logits.shape==(1,l,len(UPOS)) and
                    f.semantic_role_logits.shape==(1,l,len(SEMANTIC_ROLES)) and
                    f.dependency_logits.shape==(1,l,l) and
                    f.parallel_hypothesis_gates is not None and
                    f.parallel_hypothesis_gates.shape==(1,l,3) and
                    hashlib.sha256(native.paid_router.physical_bytes()).hexdigest()==
                       S11_39_FP16_SHA,
                    "genuine frozen prior S11 three-hypothesis native CPU ABI drift")
            rel[k,:l]=f.relations[0]
            up[k,:l]=f.upos_logits[0]
            role[k,:l]=f.semantic_role_logits[0]
            dep[k,:l,:l]=f.dependency_logits[0]
            valid[k,:l]=True
            positions=[x.last_token for x in s.aligned.word_alignment]
            require(len(positions)==len(s.row.words) and
                    all(0<=p<l for p in positions) and
                    all(positions[j]<positions[j+1] for j in range(len(positions)-1)),
                    "genuine original 43M wordpiece alignment drift")
            for i,(p,w,semantic) in enumerate(zip(positions,s.row.words,s.row.roles)):
                up_gold[k,p]=UPOS.index(w.upos)
                role_gold[k,p]=SEMANTIC_ROLES.index(semantic)
                candidates[k,p,positions[:i+1]]=True
                # Strict causal syntactic arc: a head AFTER this word
                # cannot be observed yet. Such gold is ignored, never
                # leaked into any relation-head training step.
                if 0<w.head<=i:
                    h=positions[w.head-1]
                    require(h<=p and candidates[k,p,h],
                            "future syntactic head not eligible as TRAIN target")
                    dep_gold[k,p]=h
                    arcs+=1
                words+=1
    require(words==408 and arcs>=65 and arcs<408,
            "all genuine S9 annotated TRAIN72 words/past heads required")
    require(int((up_gold>=0).sum())==int((role_gold>=0).sum())==words,
            "TRAIN gold token count or original Calibri tokenizer offset wrong")
    require(bool(torch.isfinite(rel).all()) and
            bool(torch.isfinite(up).all()) and
            bool(torch.isfinite(role).all()) and
            bool(torch.isfinite(dep).all()),
            "genuine original native C 6-cycle features not finite")
    return FixedNative72(rel,up,role,dep,valid,up_gold,role_gold,dep_gold,
                         candidates,words,arcs)


def class_weights(target:torch.Tensor,n_classes:int)->torch.Tensor:
    actual=target[target>=0]
    counts=torch.bincount(actual,minlength=n_classes).float()
    present=counts>0
    weights=torch.zeros_like(counts)
    weights[present]=(
        counts.sum()/(float(present.sum())*counts[present])
    ).clamp(.12,8.)
    return weights


def objective(fb:PaidRelationFeedback,x:FixedNative72,
              upw:torch.Tensor,rolew:torch.Tensor, *,
              regularization:bool=True)->dict[str,torch.Tensor]:
    r=fb(x.relation,x.upos,x.role,x.dep,x.valid)
    up=F.cross_entropy(r.upos_logits.reshape(-1,len(UPOS)),
                       x.upos_gold.reshape(-1),weight=upw,ignore_index=-100)
    role=F.cross_entropy(r.semantic_role_logits.reshape(-1,len(SEMANTIC_ROLES)),
                         x.role_gold.reshape(-1),weight=rolew,ignore_index=-100)
    masked=r.dependency_logits.masked_fill(~x.candidate_heads,-1e4)
    deprel=F.cross_entropy(masked.reshape(-1,masked.shape[-1]),
                           x.dep_gold.reshape(-1),ignore_index=-100)
    penalty=(
        .00025*r.relation_delta.square().mean()+
        .0001*fb.output_weight.square().sum()+
        .0001*fb.dependency_diagonal.square().sum()
    ) if regularization else torch.tensor(0.)
    total=up+role+.35*deprel+penalty
    require(all(torch.isfinite(v) for v in (total,up,role,deprel,penalty)),
            "nonfinite causal paid readout TRAIN objective")
    return {"total":total,"UPOS_balanced_CE":up,
            "semantic_role_balanced_CE":role,
            "causally_observable_head_CE":deprel,
            "regularizer":penalty}


def train_real_72_only(x:FixedNative72,*,steps=250,lr=.03):
    require(x.relation.shape[0]==72 and x.num_words==408
            and x.num_past_head_arcs>=65 and 20<=steps<=450
            and 0<lr<=.1,
            "bounded real frozen TRAIN72-only optimizer")
    torch.manual_seed(31300133)
    torch.set_num_threads(2)
    fb=PaidRelationFeedback()
    upw=class_weights(x.upos_gold,len(UPOS))
    rolew=class_weights(x.role_gold,len(SEMANTIC_ROLES))
    opt=torch.optim.Adam(fb.parameters(),lr=lr)
    fb.eval()
    with torch.no_grad():
        initial=objective(fb,x,upw,rolew,regularization=False)
        base={k:float(v) for k,v in initial.items()}
    best_total=float(initial["total"])
    best=None
    trace=[]
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        z=objective(fb,x,upw,rolew)
        z["total"].backward()
        require(fb.output_weight.grad is not None and
                bool((fb.output_weight.grad!=0).any()) and
                fb.output_bias.grad is not None and
                fb.dependency_diagonal.grad is not None,
                "no paid rank8 relation gradient to all linguistic heads")
        torch.nn.utils.clip_grad_norm_(fb.parameters(),1.)
        opt.step()
        if (i+1)%25==0 or i==steps-1:
            with torch.no_grad():
                snap=objective(fb,x,upw,rolew,regularization=False)
                physical=fb.physical_bytes()
                loaded=PaidRelationFeedback.from_physical(physical)
                exact=objective(loaded,x,upw,rolew,regularization=False)
                score=float(exact["total"])
            trace.append({"step":i+1,
                          "roundtrip_fp16_train_objective":score,
                          "train_upos_balanced_CE":float(exact["UPOS_balanced_CE"]),
                          "train_roles_balanced_CE":float(exact["semantic_role_balanced_CE"]),
                          "train_known_past_head_CE":float(exact["causally_observable_head_CE"])})
            if score<best_total:
                best_total=score
                best=physical
    require(best is not None and best_total<float(initial["total"])-.02,
            "physically loaded paid S13 source did not improve even TRAIN fit")
    physical=PaidRelationFeedback.from_physical(best)
    with torch.no_grad():
        after=objective(physical,x,upw,rolew,regularization=False)
        end={k:float(v) for k,v in after.items()}
        original=PaidRelationFeedback().eval()
        p0=original(x.relation,x.upos,x.role,x.dep,x.valid)
        p1=physical(x.relation,x.upos,x.role,x.dep,x.valid)
        def accuracy(logits,labels):
            m=labels>=0
            return {"correct":int((logits.argmax(-1)[m]==labels[m]).sum()),
                    "total":int(m.sum()),"accuracy":float(
                      (logits.argmax(-1)[m]==labels[m]).float().mean())}
        base_up=accuracy(p0.upos_logits,x.upos_gold)
        new_up=accuracy(p1.upos_logits,x.upos_gold)
        base_role=accuracy(p0.semantic_role_logits,x.role_gold)
        new_role=accuracy(p1.semantic_role_logits,x.role_gold)
        bdep=p0.dependency_logits.masked_fill(~x.candidate_heads,-1e4)
        adep=p1.dependency_logits.masked_fill(~x.candidate_heads,-1e4)
        base_dep=accuracy(bdep,x.dep_gold)
        new_dep=accuracy(adep,x.dep_gold)
        changed={k:bool(((a-b).abs()>1e-5).any()) for k,a,b in (
           ("upos",p1.upos_logits,p0.upos_logits),
           ("roles",p1.semantic_role_logits,p0.semantic_role_logits),
           ("head",p1.dependency_logits,p0.dependency_logits))}
        require(all(changed.values()),
                "learned S13 forward is not physically changing all three heads")
    report={
      "S13_real_calibri43M_train72_rows":72,
      "S13_train_only_words":x.num_words,
      "S13_causal_previous_head_supervision_arcs":x.num_past_head_arcs,
      "S13_optimizer_steps":steps,
      "S13_optimizer_learning_rate":lr,
      "S13_physical_train_only_loss_before":base,
      "S13_physical_train_only_loss_after":end,
      "S13_paid_fp16_train_only_loss_reduction":base["total"]-end["total"],
      "S13_physical_fp16_roundtrip_only":True,
      "S13_original_direct_causal_UPOS_train_accuracy":base_up,
      "S13_feedback_direct_causal_UPOS_train_accuracy":new_up,
      "S13_original_direct_causal_role_train_accuracy":base_role,
      "S13_feedback_direct_causal_role_train_accuracy":new_role,
      "S13_original_known_past_head_train_accuracy":base_dep,
      "S13_feedback_known_past_head_train_accuracy":new_dep,
      "S13_all_three_real_linguistic_outputs_modified":changed,
      "S13_no_future_head_gold_used_in_train":True,
      "S13_no_last_chat_prefix_hidden_added_to_earlier_word_prediction":True,
      "S13_training_only_trace":trace,
    }
    return best,report


def main():
    ap=argparse.ArgumentParser()
    for name in ("s10-hidden","s10-report","s8-cb59","s11-router","out"):
        ap.add_argument("--"+name,required=True,type=Path)
    a=ap.parse_args()
    require(not a.out.exists() or not any(a.out.iterdir()),
            "existing S13 physical run cannot be overwritten")
    original,s10=checked_real_train72(a.s10_hidden,a.s10_report)
    x=a.s8_cb59.read_bytes()
    prev=a.s11_router.read_bytes()
    sha_x=hashlib.sha256(x).hexdigest()
    sha_prev=hashlib.sha256(prev).hexdigest()
    require(sha_x==S8B_43_CB59_SHA and len(x)==653798 and
            len(CB59(x).handles)==43 and
            CB59(x).meta["original_unique_parameter_count"]==6_106_154,
            "old physical 6.1M K32V8 native C matrices changed")
    from research.nra3_s11.paid_router import PaidThreeWayRouter
    require(sha_prev==S11_39_FP16_SHA and len(prev)==123 and
            PaidThreeWayRouter.from_physical(prev).physical_bytes()==prev,
            "original post-S11 39param FP16 router changed or was reselected")
    a.out.mkdir(parents=True,exist_ok=True)
    lib=a.out/"original_s13_CB59_C_only.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(lib)],check=True)
    torch.set_num_threads(2)
    native=NRA3NativePackedCPU(x,NRA3Config(
        parallel_hypotheses=3,parallel_selector_version=3),lib,
        paid_router_physical=prev).eval()
    require(native.source_sha256==S8B_43_CB59_SHA
            and native.paid_router.physical_bytes()==prev
            and not native.native_cpu_weights_dense_resident
            and len(native.source.handles)==43
            and native.output_gate==0 and native.candidate_accepted is False,
            "true original 43 physical native C model not cold-loaded intact")
    saved=gather_native_fixed_training(original,native)
    assert not any(t.requires_grad for t in (
        saved.relation,saved.upos,saved.role,saved.dep))
    adapter,trained=train_real_72_only(saved,steps=250,lr=.03)
    require(len(adapter)==PHYSICAL_BYTES==2581 and
            hashlib.sha256(adapter).hexdigest()==
                hashlib.sha256(PaidRelationFeedback.from_physical(adapter).physical_bytes()).hexdigest(),
            "new physically paid S13 rank8 identity invalid")
    name="NRA3_S13_RANK8_CAUSAL_RELATION_UPOS_ROLE_DEP_FROZENTRAIN72_FP16.n3s13"
    (a.out/name).write_bytes(adapter)
    require((a.out/name).read_bytes()==adapter,"S13 exported FP16 file drift")
    # Full native inference rule: original 43 C operators and 39param paid
    # router are frozen, new 1,266 FP16 params applied only to outputs.
    physical_count=6_106_154+39+PARAMETERS
    physical_bytes=653798+123+len(adapter)
    report={
      "stage":"NRA3_S13_REAL_43M_S10_TRAIN72_1266_PAID_FP16_CAUSAL_RELATION_LINGUISTIC_READOUT_SHADOW",
      "S10_genuine_original43M_post_GT_layer8_sha256":S10_FEATURE_SHA,
      "S10_origin_success_run":S10_CAPTURE_RUN,
      "S8b_original_43_native_CB59_matrices_sha256":sha_x,
      "S8b_original_6M_unique_parameters":6_106_154,
      "S8b_original_native_C_43_matrix_source_unchanged":True,
      "S11_prior_39_FP16_selector_sha256":sha_prev,
      "S11_prior_39_selector_in_six_recurrent_relation_cycles":True,
      "S13_new_rank8_feedback_physical_filename":name,
      "S13_new_rank8_feedback_sha256":hashlib.sha256(adapter).hexdigest(),
      "S13_new_rank8_feedback_unique_parameters":PARAMETERS,
      "S13_new_rank8_feedback_physical_bytes":len(adapter),
      "S13_total_area_plus_selector_plus_feedback_parameters":physical_count,
      "S13_total_raw_area_plus_selector_plus_feedback_bytes":physical_bytes,
      "S13_raw_area_plus_selector_plus_feedback_bpp":8*physical_bytes/physical_count,
      "S13_new_readout_affects_predicted_UPOS_roles_and_observed_head":True,
      "S13_real_feedback_applied_to_earlier_grammar_semantic_layers":False,
      "S13_real_feedback_applied_to_causal_linguistic_output_heads":True,
      "S13_native_all_43_matrices_still_physically_packed_inference_C":True,
      "S13_FP16_new_2d_operators_explicitly_billed_not_5bit":True,
      "S13_uses_no_external_parser_gold_at_inference":True,
      "S13_no_original_calibri43M_or_GT_or_6M_or_S11_backprop":True,
      "S13_model_selection_used_only_preexisting_S9_train72":True,
      "S13_original_protected616_unchanged_S10_origin_only":True,
      "S13_original_protected616_with_active_candidate_not_tested":True,
      "S13_heldout16_S12c_now_CONSUMED_and_not_loaded":True,
      "S13_any_independent_new_eval_or_full_answer_EOS_measured":False,
      "S13_whole_system_less_equal_point7_bpp_certified":False,
      "S13_whole_original43M_GT_NRA_CPU_speed_at_least_90percent_certified":False,
      "five_prior_EVAL128_not_loaded":True,
      "sixth_reserved_S3CI_EVAL128_not_loaded":True,
      "no_LIVE_Render_Supabase_APK_Jessica_chat":True,
      "output_gate":0,
      "candidate_accepted":False,
      "activation_allowed":False,
      **trained,
    }
    path=a.out/"NRA3_S13_PAID_RANK8_CAUSAL_RELATION_LINGUISTIC_TRAIN72_NO_PROMOTION.json"
    path.write_text(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
                      if k!="S13_training_only_trace"},
                      ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
