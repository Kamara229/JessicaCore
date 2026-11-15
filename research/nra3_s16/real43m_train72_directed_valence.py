"""NRA3 S16b: real original43M+GT 72 TRAIN, true 43 native-C CB59 matrices.

Train ONLY the S16 directionally correct 2,309-param FP16 rank8
predicate -> argument bilinear operator, preserving *all* frozen
original43M+GT, original 43 CB59 C matrices and S11 39param router.
Input is S10 genuine 43M post-GT layer8 72 old TRAIN safetensors.
Labels originate ONLY from immutable S9 TRAIN72 216 directed links.

All PREDICATE/ARGUMENT candidates are enumerated in BOTH directions
for every distinct pair of observed original wordpieces, regardless
of gold POS/role. The scorer gets no label, target or future hidden.
Gold is used only for the supervised TRAIN loss and posthoc reports.
Do not claim 'understanding Russian' or independent generalization.
No consumed S15c EVAL24/S12c EVAL16 or old EVAL128 can be imported.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s11.real_ru72_source import (
    checked_real_train72,S10_FEATURE_SHA,S10_CAPTURE_RUN,
)
from research.nra3_s13.real43m_train72_readouts import (
    S8B_43_CB59_SHA,S11_39_FP16_SHA
)
from research.nra3_s16.direct_train72_edges import (
    build_directed_train72,sealed_directed_train72,S16_TRAIN_SOURCE_SHA,
)
from research.nra3_s16.paid_directed_valence import (
    ROLES,PARAMETERS,PHYSICAL_BYTES,PaidDirectedValence,
)

PRECOMMITTED_STEPS=200
CHECKPOINT_STEPS=(0,25,50,75,100,125,150,175,200)
LEARNING_RATE=.022
SOURCE_ONLY_S16_LOCKED_CI=43744621643
S16_TRAIN_LINKS=216
_ORIENT=("predicate_now","argument_now")


def require(condition:bool,reason:str):
    if not bool(condition):
        raise ValueError("NRA3 S16b original43M frozen native C directed TRAIN72 "+reason)


@dataclass(frozen=True)
class DirectedPairTrain:
    predicate:torch.Tensor       # [pairs,128] native C relation at its own last token
    argument:torch.Tensor        # [pairs,128] native C relation at its own last token
    predicate_positions:torch.Tensor # [pairs] real original Calibri token indices
    argument_positions:torch.Tensor  # [pairs] same, no future
    current_positions:torch.Tensor   # [pairs] max of the two
    target:torch.Tensor          # [pairs], ONLY in train loss
    orientation:torch.Tensor     # 0: predicate-now, 1: argument-now
    source_index:torch.Tensor    # [pairs], original 72 TRAIN row index
    rows:int
    positive_links:int

    @property
    def total_pairs(self):
        return int(self.target.numel())


def build_all_paired_native_train72(samples,native:NRA3NativePackedCPU)->DirectedPairTrain:
    """One real original full-43-native-C forward per original TRAIN sentence.

    The oracle TRAIN gold informs only label construction. It does not
    restrict which candidate links are scored: BOTH directed orientations
    of ALL distinct observed original Calibri words are included.
    """
    labels=build_directed_train72()
    require(len(samples)==len(labels)==72 and
            sealed_directed_train72()[
                "S16_complete_predicate_to_argument_gold_source_sha256"
            ]==S16_TRAIN_SOURCE_SHA,
            "original 72 TRAIN directed label SHA not hard pinned")
    pvec=[];avec=[];pos_p=[];pos_a=[];pos_cur=[]
    target=[];orient=[];source=[]
    hits=0; native_relation_sha=[]
    with torch.inference_mode():
        for k,(item,stamp) in enumerate(zip(samples,labels)):
            require(item.row.sha256()==stamp.text_sha256 and
                    item.row.construction==stamp.construction and
                    len(item.row.words)==stamp.word_count and
                    item.row.lexical_family==stamp.lexical_family,
                    "old S9 TRAIN row or original tokenizer physical provenance drift")
            t=int(item.hidden320.shape[0])
            out=native(item.hidden320.unsqueeze(0))
            require(out.relations.shape==(1,t,128) and
                    out.parallel_hypothesis_gates is not None and
                    out.parallel_hypothesis_gates.shape==(1,t,3) and
                    bool(torch.isfinite(out.relations).all()) and
                    native.output_gate==0 and native.candidate_accepted is False,
                    "actual 43 original CB59 C and frozen S11 3way not executed")
            relation=out.relations[0].detach().contiguous()
            truepos=[int(w.last_token) for w in item.aligned.word_alignment]
            require(len(truepos)==stamp.word_count and
                    all(0<=j<t for j in truepos) and
                    truepos==sorted(set(truepos)),
                    "native original43M Calibri wordpiece positions not ordered")
            gold={(z.predicate_word-1,z.argument_word-1):ROLES.index(z.role)
                  for z in stamp.links}
            seen_positive=set()
            for i in range(len(truepos)):
                for j in range(i+1,len(truepos)):
                    # Every unordered pair appears twice, once per direction.
                    for pred_idx,arg_idx in ((i,j),(j,i)):
                        ps=truepos[pred_idx];az=truepos[arg_idx]
                        now=max(ps,az)
                        require(ps!=az and now<t and
                                0<=min(ps,az)<now,
                                "no causal current-word pair available")
                        label=gold.get((pred_idx,arg_idx),0)
                        if label:
                            hits+=1
                            seen_positive.add((pred_idx,arg_idx))
                        pvec.append(relation[ps])
                        avec.append(relation[az])
                        pos_p.append(ps);pos_a.append(az);pos_cur.append(now)
                        target.append(label)
                        orient.append(0 if pred_idx>arg_idx else 1)
                        source.append(k)
            require(seen_positive==set(gold),
                    "missing directional gold edge in all-pairs candidate space")
            native_relation_sha.append(hashlib.sha256(
                relation.numpy().astype("<f4",copy=False).tobytes()).hexdigest())
    require(hits==S16_TRAIN_LINKS and len(pvec)>hits*5,
            "true 216 positive links and ALL negative two-way candidates required")
    make_long=lambda x:torch.tensor(x,dtype=torch.long)
    corpus=DirectedPairTrain(torch.stack(pvec).float().contiguous(),
                            torch.stack(avec).float().contiguous(),
                            make_long(pos_p),make_long(pos_a),make_long(pos_cur),
                            make_long(target),make_long(orient),make_long(source),
                            72,hits)
    require(corpus.predicate.shape==corpus.argument.shape==
            (corpus.total_pairs,128) and
            corpus.total_pairs==len(pos_p) and
            int((corpus.target!=0).sum())==hits and
            all(int((corpus.target==k).sum())==
                int(sealed_directed_train72()["S16_gold_directed_role_counts"][ROLES[k]])
                for k in range(1,len(ROLES))) and
            set(corpus.orientation.tolist())=={0,1} and
            bool((torch.maximum(corpus.predicate_positions,
                corpus.argument_positions)==corpus.current_positions).all()),
            "original wordpiece pair tensor target/direction/source not authentic")
    return corpus


def frozen_paired_logits(model:PaidDirectedValence,x:DirectedPairTrain):
    return model.score_pairs(
        x.predicate,x.argument,
        x.predicate_positions,x.argument_positions,x.current_positions)


def metrics(logits:torch.Tensor,x:DirectedPairTrain)->dict:
    require(logits.shape==(x.total_pairs,len(ROLES)) and
            bool(torch.isfinite(logits).all()),
            "all-source 5-way physical score invalid")
    y=x.target
    guess=logits.argmax(-1)
    def compute(mask):
        a=y[mask];b=guess[mask]
        tp=int(((b==a)&(a!=0)).sum())
        fp=int(((b!=a)&(b!=0)).sum())
        fn=int(((b!=a)&(a!=0)).sum())
        pre=tp/(tp+fp) if tp+fp else 0.
        rec=tp/(tp+fn) if tp+fn else 0.
        f1=2*pre*rec/(pre+rec) if pre+rec else 0.
        return {
            "pairs":int(mask.sum()),
            "gold_links":int((a!=0).sum()),
            "predicted_positive":int((b!=0).sum()),
            "exact_directed_role_true_positive":tp,
            "false_positive":fp,"false_negative":fn,
            "precision":pre,"recall":rec,"f1":f1,
            "hard_class_correct":int((a==b).sum())
        }
    on=torch.ones_like(y,dtype=torch.bool)
    all_pair=compute(on)
    byorient={name:compute(x.orientation==i) for i,name in enumerate(_ORIENT)}
    perclass={}
    for cls in range(1,len(ROLES)):
        yes=y==cls;pred=guess==cls
        tp=int((yes&pred).sum());fp=int((~yes&pred).sum())
        fn=int((yes&~pred).sum())
        denom=(2*tp+fp+fn)
        perclass[ROLES[cls]]={
            "gold":int(yes.sum()),"predicted":int(pred.sum()),
            "tp":tp,"precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "f1":2*tp/denom if denom else 0.}
    return {
        "all_two_direction_wordpair_exact_role":all_pair,
        "by_surface_orientation":byorient,
        "per_role":perclass,
        "train_ce_unweighted":float(F.cross_entropy(logits,y))
    }


def score_weighted_train(model:PaidDirectedValence,x:DirectedPairTrain,
                         weights:torch.Tensor):
    scores=frozen_paired_logits(model,x)
    ce=F.cross_entropy(scores,x.target,weight=weights)
    # Discourage extreme FP16 embedding values without unbilled scaling.
    penalty=1e-5*(
        model.predicate_weight.square().mean()+
        model.argument_weight.square().mean()+
        model.role_bilinear.square().mean())
    total=ce+penalty
    require(bool(torch.isfinite(total)),"nonfinite TRAIN-only edge objective")
    return total


def train_only(x:DirectedPairTrain)->tuple[bytes,dict]:
    torch.manual_seed(3160002)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model=PaidDirectedValence().eval()
    require(sum(z.numel() for z in model.parameters())==PARAMETERS,
            "new 2309 physical FP16 fields must be all charged")
    # Default NONE prior dominates because ALL directed pairs were
    # included. Moderate downweighting prevents trivial all-NONE while
    # retaining the real many-negative problem.
    weight=torch.tensor([.20,1.,1.,1.,1.5],dtype=torch.float32)
    snapshots=[]
    def snapshot(step):
        physical=model.physical_bytes()
        loaded=PaidDirectedValence.from_physical(physical).eval()
        require(loaded.physical_bytes()==physical and
                all(not q.requires_grad for q in loaded.parameters()),
                "train score must use exact physically reloaded FP16, not optimizer FP32")
        with torch.inference_mode():
            l=frozen_paired_logits(loaded,x)
            m=metrics(l,x)
            m["weighted_train_CE"]=float(F.cross_entropy(
                l,x.target,weight=weight))
        return {"step":step,"physical":physical,
                "physical_SHA256":hashlib.sha256(physical).hexdigest(),
                "TRAIN_only":m}
    snapshots.append(snapshot(0))
    optim=torch.optim.AdamW(model.parameters(),lr=.025,weight_decay=.0002)
    for step in range(1,PRECOMMITTED_STEPS+1):
        optim.zero_grad(set_to_none=True)
        loss=score_weighted_train(model,x,weight)
        loss.backward()
        require(all(q.grad is not None and bool(torch.isfinite(q.grad).all())
                    for q in model.parameters()),
                "all charged projection and four directed role fields must get gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if step in CHECKPOINT_STEPS:
            snapshots.append(snapshot(step))
    require(len(snapshots)==len(CHECKPOINT_STEPS)==9,
            "all precommitted physical checkpoint FP16 sources required")
    # Deliberately TRAIN-ONLY ranking of PAID PHYSICAL bytes.
    # An increase in in-sample F1 is NOT evidence of generalization.
    def rank(z):
        m=z["TRAIN_only"]["all_two_direction_wordpair_exact_role"]
        return (m["f1"],m["recall"],
                -z["TRAIN_only"]["weighted_train_CE"],-z["step"])
    selected=max(snapshots,key=rank)
    report={
        "S16b_train_steps":PRECOMMITTED_STEPS,
        "S16b_physical_step_schedule":list(CHECKPOINT_STEPS),
        "S16b_all_pairs_were_scored_in_BOTH_directions":True,
        "S16b_only_original_72_TRAIN_gold_used":True,
        "S16b_no_oracle_POS_or_gold_link_candidates_at_inference":True,
        "S16b_no_future_wordpiece_token_readout":True,
        "S16b_training_examples_sentence_rows":x.rows,
        "S16b_total_directed_wordpair_candidates":x.total_pairs,
        "S16b_positive_directed_role_gold":x.positive_links,
        "S16b_actual_direction_counts":{
            "predicate_now":int((x.orientation==0).sum()),
            "argument_now":int((x.orientation==1).sum())},
        "S16b_untrained_seed_fp16_TRAIN_metrics":snapshots[0]["TRAIN_only"],
        "S16b_selected_TRAIN_only_FP16_step":selected["step"],
        "S16b_selected_TRAIN_only_FP16_SHA":selected["physical_SHA256"],
        "S16b_selected_TRAIN_only_metrics":selected["TRAIN_only"],
        "S16b_selected_because_of_TRAIN_only":True,
        "S16b_all_precommitted_FP16_checkpoints":[{
          "step":z["step"],"sha256":z["physical_SHA256"],
          "TRAIN_only":z["TRAIN_only"]} for z in snapshots],
    }
    return selected["physical"],report


def main():
    arg=argparse.ArgumentParser()
    for name in ("s10-hidden","s10-report","s8-cb59","s11-router","out"):
        arg.add_argument("--"+name,type=Path,required=True)
    a=arg.parse_args()
    require(not a.out.exists(),
            "do not overwrite any previous genuine S16b physical run or model")
    samples,original_s10=checked_real_train72(a.s10_hidden,a.s10_report)
    require(len(samples)==72 and
            original_s10["all_616_native_CPU_generated_token_digests_unchanged"] is True,
            "original 43M+GT original J7 true CPU TRAIN72 checksum failed")
    original=a.s8_cb59.read_bytes()
    paid39=a.s11_router.read_bytes()
    require(len(original)==653798 and
            hashlib.sha256(original).hexdigest()==S8B_43_CB59_SHA and
            len(CB59(original).handles)==43 and
            CB59(original).meta["original_unique_parameter_count"]==6_106_154,
            "all 43 original packed CB59 C 5bit matrices must be SHA-locked")
    require(len(paid39)==123 and
            hashlib.sha256(paid39).hexdigest()==S11_39_FP16_SHA and
            PaidThreeWayRouter.from_physical(paid39).physical_bytes()==paid39,
            "old 39 physical S11 selector must be immutable")
    a.out.mkdir(parents=True,exist_ok=False)
    library=a.out/"s16b_original_43_CB59_native.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
        "research/calibri/direct_kernel_v59.c","-o",str(library)],check=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native=NRA3NativePackedCPU(
        original,NRA3Config(parallel_hypotheses=3,
                            parallel_selector_version=3),library,
        paid_router_physical=paid39).eval()
    require(native.source_sha256==S8B_43_CB59_SHA and
            native.paid_router.physical_bytes()==paid39 and
            not native.native_cpu_weights_dense_resident and
            native.output_gate==0 and native.candidate_accepted is False and
            all(not q.requires_grad for q in native.parameters()),
            "S16b must execute original frozen 43 matrix C native weights")
    gold=sealed_directed_train72()
    require(gold["S16_complete_predicate_to_argument_gold_source_sha256"]==
            S16_TRAIN_SOURCE_SHA and
            gold["S16_gold_directed_links"]==S16_TRAIN_LINKS,
            "S16 label source all 216 PREDICATE→ARGUMENT links not pinned")
    x=build_all_paired_native_train72(samples,native)
    physical,m=train_only(x)
    require(len(physical)==PHYSICAL_BYTES==4665 and
            PaidDirectedValence.from_physical(physical).physical_bytes()==physical,
            "S16b new parameter physical reload is incomplete or unbilled")
    output_name="NRA3_S16B_2309FP16_DIRECTED_VALENCE_REAL43M_TRAIN72_ONLY.n3s16"
    target=a.out/output_name
    target.write_bytes(physical)
    require(target.read_bytes()==physical,"physically paid learned source changed")
    result={
        "stage":"NRA3_S16B_TRUE_ORIGINAL43M_GT_RU72_216_DIRECTED_LINKS_NATIVE_C_TRAIN_ONLY",
        "S10_true_original43M_post_GT_layer8_train72_FP16_SHA":S10_FEATURE_SHA,
        "S10_original_43M_616_J7_provenance_run":S10_CAPTURE_RUN,
        "S8b_original_43_packed_5bit_native_C_SHA":S8B_43_CB59_SHA,
        "S11_original_39_FP16_six_cycle_router_SHA":S11_39_FP16_SHA,
        "S16_original_directed_training_gold_complete_SHA":S16_TRAIN_SOURCE_SHA,
        "S16_original_directed_training_gold_source_success_CI":SOURCE_ONLY_S16_LOCKED_CI,
        "S16b_NEW_2309_paid_FP16_filename":output_name,
        "S16b_NEW_2309_paid_FP16_SHA":hashlib.sha256(physical).hexdigest(),
        "S16b_NEW_2309_paid_FP16_physical_bytes":PHYSICAL_BYTES,
        "S16b_NEW_2309_paid_FP16_params":PARAMETERS,
        "S16b_original_43_CB59_native_C_matrices_unchanged":True,
        "S16b_original_Calibri43M_GT_and_S11_router_unchanged":True,
        "S16b_original_43_CB59_cpu_executed_on_all_72_real_hidden":True,
        "S16b_original_parent_616_J7_baseline_unchanged_S10":True,
        "S16b_active_candidate_616_J7_or_fullEOS_measured":False,
        "S16b_separate_future_NEW_independent_Russian_EVAL_performed":False,
        "S16b_previously_consumed_S15c_EVAL24_not_loaded":True,
        "S16b_previously_consumed_S12c_EVAL16_not_loaded":True,
        "S16b_old_five_EVAL128_and_reserved_sixth_S3CI_not_loaded":True,
        "S16b_whole_original43M_GT_NRA_CPU_speed_ge90pct_measured":False,
        "S16b_whole_system_inclusive_bpp_le07_certified":False,
        "S16b_new_S16_operator_raw_standalone_FP16_bpp":
            8*PHYSICAL_BYTES/PARAMETERS,
        "S16b_original_NRA6M_plus_S11_plus_S16_direct_raw_bytes":
            len(original)+len(paid39)+len(physical),
        "S16b_raw_NRA6M_plus_S11_plus_S16_direct_bpp":
            8*(len(original)+len(paid39)+len(physical))/
                (6_106_154+39+PARAMETERS),
        "S16b_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
        "S16b_research_quality_acceptance":False,
        "S16b_candidate_accepted":False,
        "S16b_output_gate":0,
        "S16b_activation_allowed":False,
        **m,
    }
    path=a.out/"NRA3_S16B_TRUE43M_GOLD216_DIRECTED_VALENCE_FP16_TRAIN_ONLY_REPORT.json"
    path.write_text(json.dumps(result,sort_keys=True,indent=2,
        ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items()
        if k not in ("S16b_all_precommitted_FP16_checkpoints",)},
        sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
