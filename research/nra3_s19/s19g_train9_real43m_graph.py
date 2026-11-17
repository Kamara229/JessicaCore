"""S19g: genuine Calibri43M+Green Tunnel, frozen S19d, FULL TRAIN9 only.

One tiny, 9-sentence **TRAIN-resubstitution** optimization verifies that a
real gold-blind neural multihead can backpropagate event, directed role,
participant rejection, coreference and inter-event relations *separately*.
It is NOT a heldout metric, generalization proof or independent EVAL.
No candidate activation or NRA2 production integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph, semantic_parameter_count, full_parameter_count,
    NODE_INPUT, STATUS, ROLE, ARG_STATUS, COREF_STATUS, EVENT_RELATION,
)
from research.nra3_s19.train_only_event_graph_source import (
    TRAIN_ONLY_FROZEN_SHA256,
)

S19D_FULL_PHYSICAL_SHA = "928f735a2018604db2ffe649799778df3805162b79a989a36e282dc3cb61c774"
S19F_REAL_SOURCE_SHA = "213fe770868e5b6ad76c1f3955200ee4e648581a3371e6c53b3fc707ff764c08"
S19E_TRUE_CPU = 44132334034
S19F_PHYSICAL_CI = 44133991642
S19D_TRUE_CPU = 44128707585

STEPS=240
LR=.012
SEED=319107
NONE_WEIGHT=.06


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S19g genuine43M-GT 9-TRAIN-only semantic graph: "+msg)


def load_segregated_real_source(features:Path, gold:Path, manifest:Path):
    for f in (features,gold,manifest):
        require(Path(f).is_file(),"separate S19f real physical source file missing")
    data=json.loads(Path(manifest).read_text(encoding="utf-8"))
    require(data["stage"]=="NRA3_S19F_TRUE_ORIGINAL43M_GT_TRAIN9_TYPED_GRAPH_TARGETS_ONLY"
            and data["TRAIN9_source_gold_SHA256"]==TRAIN_ONLY_FROZEN_SHA256
            and data["original43M_GT_new9_real_hidden_SHA256"]==S19F_REAL_SOURCE_SHA
            and data["parent_616_J7_generation_digest_unchanged"]
            and data["all_model_features_from_true43M_GT_hidden320_and_raw_text_only"]
            and data["all_human_gold_event_role_scope_coref_relations_target_only"]
            and data["no_optimizer_no_new_model_scoring"]
            and data["no_spent_independent_EVAL32_EVAL128_reused"]
            and data["new9_realCPU_full_tensor_roundtrip"]
            and data["optimizer_steps"]==0
            and data["output_gate"]==0 and not data["candidate_accepted"]
            and not data["activation_allowed"]
            and data["gold_blind_only_feature_file"]==features.name
            and data["source_human_targets_filename"]==gold.name
            and hashlib.sha256(features.read_bytes()).hexdigest()==data["gold_blind_only_feature_sha256"]
            and hashlib.sha256(gold.read_bytes()).hexdigest()==data["source_human_targets_sha256"],
            "source verified original 43M CPU word features/human gold SHA drift")
    x=load_file(str(features),device="cpu")
    y=load_file(str(gold),device="cpu")
    require(set(x)=={"raw_text_frozen_word338","valid_word_mask"}
            and set(y)=={"event_status_TRAIN_ONLY","predicate_argument_role_TRAIN_ONLY",
                        "argument_polarity_TRAIN_ONLY","coreference_TRAIN_ONLY",
                        "event_relation_TRAIN_ONLY"},
            "STRICT physically separate gold-free model input and human TRAIN labels")
    raw=x["raw_text_frozen_word338"]
    valid=x["valid_word_mask"]
    require(raw.ndim==3 and raw.shape[0]==9 and raw.shape[-1]==338
            and raw.dtype==torch.float32 and valid.dtype==torch.bool
            and valid.shape==raw.shape[:2]
            and bool(valid[:,0].all()) and bool(torch.isfinite(raw).all())
            and bool((raw[~valid]==0).all())
            and bool(((raw[:,:,-18:]==0)|(raw[:,:,-18:]==1)).all()),
            "gold-blind real CPU original + raw18 physical tensor corrupt")
    b,w=valid.shape
    pairs=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(w,dtype=torch.bool).unsqueeze(0))
    require(y["event_status_TRAIN_ONLY"].shape==(b,w)
            and all(t.shape==(b,w,w) for k,t in y.items()
                    if k!="event_status_TRAIN_ONLY")
            and int((y["event_status_TRAIN_ONLY"]>0).sum())==15
            and int((y["predicate_argument_role_TRAIN_ONLY"]>0).sum())==40
            and int((y["argument_polarity_TRAIN_ONLY"]==1).sum())==1
            and int((y["coreference_TRAIN_ONLY"]==1).sum())==5
            and int((y["event_relation_TRAIN_ONLY"]>0).sum())==6
            and bool((y["event_status_TRAIN_ONLY"][~valid]==-1).all())
            and all(bool((t[~pairs]==-1).all()) for k,t in y.items()
                    if k!="event_status_TRAIN_ONLY"),
            "sealed human TRAIN-only event graph loss targets invalid")
    return raw,valid,y,data


def pair_targets(y):
    role=y["predicate_argument_role_TRAIN_ONLY"]
    p=y["argument_polarity_TRAIN_ONLY"]
    # Original S19f human role edge polarity is 0=POSITIVE, 1=REJECTED;
    # all nonedges become the explicit classifier NONE=0. Pure TARGETS.
    polarity=torch.where(p<0,0,p+1)
    require(int((polarity==2).sum())==int((p==1).sum())
            and int((polarity==1).sum())==int((p==0).sum())
            and int((polarity>0).sum())==int((role>0).sum())
            and bool((p[role>0]>=0).all())
            and bool((p[role<=0]==-1).all()),
            "rejected/positive participants must be supervised separately, even if rare class absent in TRAIN subset")
    return polarity


def fixed_multitask_loss(out,y,valid):
    b,w=valid.shape
    mask=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(w,dtype=torch.bool).unsqueeze(0))
    e=F.cross_entropy(out.event_status[valid],y["event_status_TRAIN_ONLY"][valid],
                      weight=torch.tensor([.15,1.1,2.,2.,3.,1.]))
    role=F.cross_entropy(out.role[mask],
                         y["predicate_argument_role_TRAIN_ONLY"][mask],
                         weight=torch.tensor([.05]+[1.5]*(len(ROLE)-1)))
    pol=F.cross_entropy(out.argument_polarity[mask],pair_targets(y)[mask],
                        weight=torch.tensor([.06,1.,10.]))
    coref=F.cross_entropy(out.coreference[mask],y["coreference_TRAIN_ONLY"][mask],
                          weight=torch.tensor([.07,3.]))
    rel=F.cross_entropy(out.event_relation[mask],y["event_relation_TRAIN_ONLY"][mask],
                        weight=torch.tensor([.07]+[4.]*(len(EVENT_RELATION)-1)))
    return e+.8*role+.6*pol+.5*coref+.6*rel


def task_train_reuse_report(out,y,valid):
    b,w=valid.shape
    pair_mask=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(w,dtype=torch.bool).unsqueeze(0))
    tasks=(
        ("event",out.event_status,y["event_status_TRAIN_ONLY"],valid),
        ("directed_role",out.role,y["predicate_argument_role_TRAIN_ONLY"],pair_mask),
        ("argument_polarity",out.argument_polarity,pair_targets(y),pair_mask),
        ("coreference",out.coreference,y["coreference_TRAIN_ONLY"],pair_mask),
        ("event_relation",out.event_relation,y["event_relation_TRAIN_ONLY"],pair_mask),
    )
    result={}
    for name,logits,gold,valid_task in tasks:
        guess=logits.argmax(-1)[valid_task]
        actual=gold[valid_task]
        tp=int(((guess==actual)&(actual>0)).sum())
        fp=int(((guess>0)&((guess!=actual)|(actual==0))).sum())
        fn=int(((actual>0)&((guess!=actual)|(guess==0))).sum())
        result[name]={
            "TRAIN_ONLY_supervised_labels_nonNONE":int((actual>0).sum()),
            "TP":tp,"FP":fp,"FN":fn,
            "TRAIN_RESUBSTITUTION_micro_F1_NOT_independent":
                2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "TRAIN_RESUBSTITUTION_exact_class_accuracy_NOT_independent":
                float((guess==actual).float().mean()),
        }
    return result


def main():
    parser=argparse.ArgumentParser()
    for n in ("features","gold","manifest","s19d-parent","out"):
        parser.add_argument("--"+n,required=True,type=Path)
    a=parser.parse_args()
    require(not a.out.exists(),"must not overwrite any prior research artifacts")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw,valid,y,source=load_segregated_real_source(
        a.features,a.gold,a.manifest)
    parent_physical=a.s19d_parent.read_bytes()
    require(hashlib.sha256(parent_physical).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "cannot substitute original frozen S19d full TRAIN96 paid parent")
    parent=PaidContextualEventScope.from_physical(parent_physical)
    require(parent.output_gate==0 and not parent.candidate_accepted
            and not parent.activation_allowed and
            all(not p.requires_grad for p in parent.parameters()),
            "frozen S19d contextual parent must never train")
    torch.manual_seed(SEED)
    model=PaidSemanticEventGraph(parent).train()
    require(semantic_parameter_count(model)==3049
            and full_parameter_count(model)==12381
            and model.output_gate==0 and not model.candidate_accepted
            and not model.activation_allowed,
            "physically billed event/role/COREF/relations graph ABI invalid")
    optimizer=torch.optim.AdamW(
        list(model.semantic_params()),lr=LR,weight_decay=.001)
    with torch.inference_mode():
        first=fixed_multitask_loss(model(raw,valid),y,valid).item()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        score=model(raw,valid)
        loss=fixed_multitask_loss(score,y,valid)
        require(bool(torch.isfinite(loss)),"nonfinite graph TRAIN loss")
        loss.backward()
        require(all(p.grad is None for p in model.parent.parameters()),
                "forbidden gradient to original frozen grammar/context parent")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.semantic_params()),
                "semantic graph head gradient missing")
        torch.nn.utils.clip_grad_norm_(list(model.semantic_params()),1.)
        optimizer.step()
    physical=model.physical_bytes()
    readback=PaidSemanticEventGraph.from_physical(physical).eval()
    require(not any(p.requires_grad for p in readback.parameters())
            and readback.head_physical()==model.head_physical()
            and readback.output_gate==0
            and not readback.candidate_accepted
            and not readback.activation_allowed,
            "paid joint parent/graph FP16 storage not verifiable")
    with torch.inference_mode():
        after=fixed_multitask_loss(readback(raw,valid),y,valid).item()
        metrics=task_train_reuse_report(readback(raw,valid),y,valid)
    require(after<first and not torch.isnan(torch.tensor(after)),
            "TRAIN-only graph head failed to decrease fixed multihead loss")
    a.out.mkdir(parents=True,exist_ok=False)
    fn="NRA3_S19G_REAL43M_GT_NINE_TRAIN_SEMANTIC_GRAPH_12381_FP16.n3s19g"
    dest=a.out/fn
    dest.write_bytes(physical)
    require(dest.read_bytes()==physical,
            "FP16 graph model file did not persist exact SHA bytes")
    report={
      "stage":"NRA3_S19G_REAL43M_GT_NEW9_TRAIN_ONLY_SEMANTIC_GRAPH_SHADOW",
      "original_source_S19e_CPU_616_J7":S19E_TRUE_CPU,
      "original_REAL_S19f_graph_physical_join":S19F_PHYSICAL_CI,
      "frozen_contextual_parent_S19d_CPU_run":S19D_TRUE_CPU,
      "S19d_full_TRAIN96_parent_paid_SHA":S19D_FULL_PHYSICAL_SHA,
      "S19f_real_original43M_GT_CPU_source_sha":S19F_REAL_SOURCE_SHA,
      "S19_graph_new9_human_gold_SHA":TRAIN_ONLY_FROZEN_SHA256,
      "S19f_physically_separated_input_SHA":source["gold_blind_only_feature_sha256"],
      "S19f_physically_separated_human_label_SHA":source["source_human_targets_sha256"],
      "source_sealed_9_TRAIN_sentences":9,
      "loss_before_training_TRAIN_ONLY":first,
      "loss_after_physical_FP16_training_TRAIN_ONLY":after,
      "train_only_full9_resubstitution_NOT_generalization":metrics,
      "train_optimizer_steps_on_9_human_TRAIN_sentences_only":STEPS,
      "all_predicate_event_role_rejection_coreference_disourse_predictions_NO_gold_inputs":True,
      "trained_event_graph_joint_outputs":list(metrics),
      "frozen_S19d_parent_FP16_parameters":9332,
      "learned_graph_head_FP16_parameters":3049,
      "all_physically_included_FP16_parameters":12381,
      "full_FP16_joint_model_sha256":hashlib.sha256(physical).hexdigest(),
      "full_FP16_joint_model_filename":fn,
      "original_43M_and_Green_Tunnel_weights_not_updated":True,
      "original_parent_616_protected_J7_digest_preceded_original_train_capture":True,
      "new_candidate_active_616_J7_NOT_certified":True,
      "new_candidate_all_in_bpp_le07_NOT_certified":True,
      "new_candidate_CPU_throughput_ge90pct_CB59_NOT_certified":True,
      "strict_online_generation_NOT_supported_or_measured":True,
      "overfit_on_nine_training_examples_NOT_independent_evaluation":True,
      "no_old_or_spent_EVAL32_EVAL128_S17c_S15c_S12c_loaded":True,
      "NRA2_decoding_gate_connected":False,
      "candidate_accepted":False,"activation_allowed":False,"output_gate":0,
      "LIVE_Render_Supabase_Jessica_APK_untouched":True,
    }
    (a.out/"NRA3_S19G_REAL43M_GT_NEW9_TRAIN_ONLY_GRAPH_SHADOW_REPORT.json").write_text(
       json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
       encoding="utf-8")
    print("S19G_TRAIN9_ONLY_GRAPH_LOSS",first,after,flush=True)
    print("S19G_TRAIN9_RESUBSTITUTION_NOT_EVAL",json.dumps(metrics,sort_keys=True),flush=True)
    print("S19G_FULL_PHYSICAL_SHA",report["full_FP16_joint_model_sha256"],flush=True)


if __name__=="__main__":
    main()
