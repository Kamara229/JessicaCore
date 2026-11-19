"""NRA3 S115: fixed SynTagRus-A CC BY-NC-SA NONCOMMERCIAL TRAIN-only.

Physically protected Calibri CB59+GreenTunnel all12 after original616 J7.
Only after S114 original12 source SHA check open separately precommitted
NEW SynTagRus TRAIN-a UD gold, fully disjoint from prior GSD/Taiga.
Freeze all S112 53,743 trained in-Calibri neural weights, original NRA2
eight minis, 12layer router, memory16, and original core/GT.
Train exactly EIGHT newly appended bounded neural trust scalars.
The S112 forward enforces |native-BPE argument correction|<=1 logit.
NO internal developer scoring (not even once) on S115: fixed TRAIN ONLY,
leave new SynTagRus last16/64 entirely UNREAD by diagnosis for later
strict preregistered one-shot firstread; all earlier dev and TaigaC burnt.
No candidate integration, LIVE, production, bpp/CPU or independent QA proof.
NONCOMMERCIAL license: NEVER integrate commercially without direct license.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,MAX_LOGIT_CORRECTION,
    TOTAL,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    OriginalBPEAllVerbsEdges,offline_all_verb_multi_positive_relation_loss,
)
from research.calibri_nra3.s115_SynTagRusNC_original12_GOLD_AFTER_S114_SHA import (
    load_source_VERIFY_sha_THEN_OPEN_gold,UNKNOWN,
)

STAGE="CALIBRI_NRA3_S115_NC_SYNTAGRUS_A_NEW8_BOUNDED_NEURAL_TRUST_TRAIN_ONLY_GATE0"
SOURCE_SHA="e9972e4f2204887cd1a507289dc0e4936f18c9f8ec9e18a0a2630e1009a1f209"
GOLD_SHA="a7d41b9b6925c2ad9b2051294813a7af91c963dfae534e3ceca7bf3e4eeb7395"
SEED=1151120
EPOCHS=24
LR=0.010
WD=0.02
PREFIX="model.nra3_neural_weights."
OLD=53743
EXTRA=8
CHILD="NRA3_S114_IN_CALIBRI_ZERO_TRUST_53751_FP16.safetensors"
TRAINED="NRA3_S115_SYNTAGRUS_NC_NEW8_BOUNDED_TRUST_53751_IN_CALIBRI_FP16.safetensors"
REPORT="NRA3_S115_SYNTAGRUS_NC_NEW8_TRAIN_ONLY_NO_DEV_REPORT.json"

def require(x,why):
    if not bool(x):
        raise ValueError("S115 NC original12 true Calibri 8 neural trust TRAIN-only: "+why)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def batches_fixed(source_ids,train_mask):
    cohorts=[]
    for i in range(4):
        chosen=torch.where((source_ids>=64*i)&(source_ids<64*i+48)&train_mask)[0]
        require(len(chosen)>=16,">=16 exact BPE source rows per fixed train stratum")
        cohorts.append(chosen)
    n=min(len(x)//2 for x in cohorts)
    require(n>=8,"at least 8 fixed balanced batches each epoch")
    return [torch.cat([g[2*j:2*j+2] for g in cohorts]) for j in range(n)]


def select_targets(target:OriginalBPEAllVerbsEdges,indices):
    return OriginalBPEAllVerbsEdges(
        native_BPE_valid=target.native_BPE_valid[indices],
        argument_gold_positive=target.argument_gold_positive[indices],
        event_role_supervised=target.event_role_supervised[indices],
        event_verb_positive_only=target.event_verb_positive_only[indices],
    )


@torch.inference_mode()
def teacher_anchored_diagnostic(net,hidden,aligned,subset):
    """Only AFTER forward score annotated argument POSITIVE token; no oracle forward."""
    names=("upos","case","number","voice")
    m={k:{"correct":0,"support":0} for k in names}
    r={k:{"correct":0,"role_slots":0,"positive_edges":0} for k in ROLE_NAMES}
    rows=torch.where(subset)[0].tolist()
    for x in rows:
        g=net(hidden[x:x+1].float().detach(),aligned.role.native_BPE_valid[x:x+1])
        for k in names:
            truth=aligned.morphology[k][x]
            mask=truth!=UNKNOWN
            correct=int((getattr(g,k+"_logits")[0][mask].argmax(-1)==truth[mask]).sum())
            m[k]["correct"]+=correct
            m[k]["support"]+=int(mask.sum())
        for j,name in enumerate(ROLE_NAMES):
            slots=torch.where(aligned.role.event_role_supervised[x,j])[0]
            for p in slots.tolist():
                positive=aligned.role.argument_gold_positive[x,j,p]
                prediction=int(g.role_arc_logits[0,j,p].argmax())
                r[name]["role_slots"]+=1
                r[name]["positive_edges"]+=int(positive.sum())
                r[name]["correct"]+=int(bool(positive[prediction]))
    for d in m.values():
        d["accuracy"]=d["correct"]/d["support"] if d["support"] else None
    for d in r.values():
        d["teacher_anchored_ANY_positive_top1"]=d["correct"]/d["role_slots"] if d["role_slots"] else None
    return {
        "n_source_rows":len(rows),
        "morphology_original_native_BPE":m,
        "multi_verb_role_teacher_anchored":{k:v for k,v in r.items() if v["role_slots"]},
        "not_autonomous_event_detection_or_factuality_QA":True,
        "not_independent_prospective_firstread":True,
    }





def train(source12,source_manifest,old_s114,s113_gold_folder,out):
    require(not out.exists(),"never overwrite neural shadow training artifacts")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    # Exact original S114 true CB59+GT source SHA MUST be physical and
    # immutable BEFORE opening official SynTagRus licensed UD GOLD.
    source,labels,audit=load_source_VERIFY_sha_THEN_OPEN_gold(
        source12,source_manifest,s113_gold_folder)
    m=json.loads(source_manifest.read_text(encoding="utf8"))
    require(old_s114.name==CHILD
            and sha(old_s114)==m["IN_CALIBRI_53751_MORPH_FP16_WEIGHTS_SHA256"]
            and m["prior_S110_53743_and_new_8_zero_trust_exactly_restored_from_S112"]
            and m["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and m["noncommercial_research_only_CC_BY_NC_SA_4_license"]
            and m["output_gate"]==0 and not m["candidate_accepted"],
            "only actual S114 53,751 in-Calibri untrained zero trust checkpoint")
    h=source["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    require(h.dtype==torch.float16 and h.ndim==4
            and h.shape[1]==12 and h.shape[-1]==320
            and not h.requires_grad and torch.isfinite(h).all()
            and h.shape[0]==len(labels.original_source_row_indices),
            "genuine S114 original CB59+GT fixed all12 native BPE")
    prior=load_file(str(old_s114),device="cpu")
    require(sum(v.numel() for v in prior.values())==TOTAL==53751
            and all(k.startswith(PREFIX) and v.dtype==torch.float16
                    and torch.isfinite(v).all() for k,v in prior.items())
            and torch.count_nonzero(prior[PREFIX+"structural_trust_gate"])==0,
            "exact previous S112 neural child zero-trust FP16 initial condition")
    net=NRA3BoundedStructuralTrustInCalibriWeights()
    net.load_state_dict({k[len(PREFIX):]:v.float().clone() for k,v in prior.items()},
                        strict=True)
    require(net.budget()["new_neural_parameters"]==53751
            and net.output_gate==0 and not net.candidate_accepted
            and not net.activation_allowed and not net.is_trained
            and MAX_LOGIT_CORRECTION==1,
            "actual bounded neural trust model inside research Calibri gate0")
    for name,p in net.named_parameters():
        p.requires_grad_(name=="structural_trust_gate")
    paid=[p for p in net.parameters() if p.requires_grad]
    require(sum(p.numel() for p in paid)==EXTRA==8
            and sum(p.numel() for p in net.parameters() if not p.requires_grad)==OLD,
            "only eight in-Calibri neural trust scalars get TRAIN gradients")
    batches=batches_fixed(labels.original_source_row_indices,labels.train)
    optimizer=torch.optim.AdamW(paid,lr=LR,weight_decay=WD)
    loss_trace=[]
    for ep in range(EPOCHS):
        net.train()
        batchloss=[]
        for ix in batches:
            require(bool(labels.train[ix].all())
                    and not bool(labels.internal_dev[ix].any()),
                    "S115 only first48 per original SynTagRus strata; NEVER internal16")
            optimizer.zero_grad(set_to_none=True)
            graph=net(h[ix].float().detach(),labels.role.native_BPE_valid[ix])
            loss,_=offline_all_verb_multi_positive_relation_loss(
                graph,select_targets(labels.role,ix))
            require(torch.isfinite(loss),"fixed positive-only UD relation loss")
            loss.backward()
            require(net.structural_trust_gate.grad is not None
                    and torch.isfinite(net.structural_trust_gate.grad).all(),
                    "train only new eight trust-weight gradient channels")
            nn.utils.clip_grad_norm_(paid,1.0)
            optimizer.step()
            batchloss.append(float(loss.detach()))
        if ep in (0,5,11,17,EPOCHS-1):
            loss_trace.append({"epoch":ep,"ONLY_GENUINE_SYNTAGRUS_TRAIN_loss":
                               sum(batchloss)/len(batchloss)})
    net.eval()
    net.is_trained=True
    result={PREFIX+k:v.detach().half().cpu().contiguous()
            for k,v in net.state_dict().items()}
    require(sum(v.numel() for v in result.values())==53751
            and all(torch.equal(v,result[k]) for k,v in prior.items()
                    if k!=PREFIX+"structural_trust_gate")
            and not torch.equal(prior[PREFIX+"structural_trust_gate"],
                                result[PREFIX+"structural_trust_gate"])
            and torch.isfinite(result[PREFIX+"structural_trust_gate"]).all(),
            "all previous 53,743 FP16 bit-identical, only 8 inmodel neural trust weights updated")
    require(sha(old_s114)==m["IN_CALIBRI_53751_MORPH_FP16_WEIGHTS_SHA256"],
            "true source-only research child still pinned")
    # TRAIN-labelled diagnostic ONLY; not same-corpus internal/dev.
    # No helper call with labels.internal_dev and NO report with hidden metrics.
    old=NRA3BoundedStructuralTrustInCalibriWeights().eval()
    old.load_state_dict({k[len(PREFIX):]:v.float().clone() for k,v in prior.items()},
                       strict=True)
    baseline=teacher_anchored_diagnostic(old,h,labels,labels.train)
    trained=teacher_anchored_diagnostic(net,h,labels,labels.train)

    out.mkdir(parents=True,exist_ok=False)
    save_file(result,str(out/TRAINED))
    roundtrip=load_file(str(out/TRAINED))
    require(set(roundtrip)==set(result)
            and all(torch.equal(roundtrip[k],v) for k,v in result.items()),
            "physical 53,751 complete FP16 in-Calibri child checksum roundtrip")
    manifest={
        "stage":STAGE,
        "S114_physical_original12_source_SHA256":m["ALL12_CPU_SAFETENSORS_SHA256"],
        "S114_previous_original12_53751_neural_child_SHA256":sha(old_s114),
        "S115_trained_IN_CALIBRI_53751_FP16_child_SHA256":sha(out/TRAINED),
        "official_SynTagRus_SOURCE_SHA256":SOURCE_SHA,
        "official_SynTagRus_SEPARATE_GOLD_SHA256":GOLD_SHA,
        "original_616_J7_CB59_GT_source_protected":True,
        "S113_teacher_gold_opened_ONLY_AFTER_S114_original12_source_SHA":True,
        "S115_ONLY_NEW_EIGHT_neural_trust_weights_trained":True,
        "ALL_PREVIOUS_53743_NRA2_mini8_memory16_router12_structural_FP16_EXACT":True,
        "maximum_native_BPE_logit_residual_PER_candidate":1.,
        "source_only_structural_no_external_UD_parser_at_inference":True,
        "original_SynTagRus_TRAINa_NC_LICENSE":"CC BY-NC-SA 4.0",
        "NONCOMMERCIAL_research_only_NO_model_licensing_commercial_use":True,
        "old_S79_S86_firstreads_SPENT_not_loaded":True,
        "S96_S99_S102_S105_internal_devs_SPENT_not_loaded":True,
        "S110_GSD_internal_dev_SPENT_not_loaded":True,
        "TaigaC_one_shot_firstread_ALREADY_BURNED_NOT_USED":True,
        "SynTagRus_INTERNAL_16per64_NEVER_SCORED_OR_TUNED":True,
        "no_independent_transfer_or_autonomous_event_accuracy_claim":True,
        "no_new_active_Calibri_online_J7_or_allin_CPU_BPP_certification":True,
        "fixed_training":{"seed":SEED,"epochs":EPOCHS,
                          "lr":LR,"weight_decay":WD,
                          "neural_trainable_params":8,
                          "other_in_calibri_frozen_params":53743,
                          "fixed_batches":len(batches)},
        "native_original_BPE_alignment":audit,
        "TRAIN_only_loss_trace":loss_trace,
        "frozen_S112_BASELINE_on_THIS_TRAIN_only":baseline,
        "S115_EIGHT_NEURAL_TRUST_on_THIS_TRAIN_only":trained,
        "learned_neural_tanh_trust_per_role":[float(z) for z in
                                           torch.tanh(net.structural_trust_gate).tolist()],
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(json.dumps(
        manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA3_S115_TRUE12_NONCOMMERCIAL_ONLY_EIGHT_INCALIBRI_TRUST_NEURAL_TRAIN",
          json.dumps({"rows":audit["counts"]["full_UD_BPE_rows"],
                      "positive_relation_edges":audit["counts"]["positive_event_argument_edges"],
                      "TRAIN_only_baseline":baseline["multi_verb_role_teacher_anchored"],
                      "TRAIN_only_trained":trained["multi_verb_role_teacher_anchored"],
                      "eight_neural_trust":manifest["learned_neural_tanh_trust_per_role"],
                      "internal_developer_scores":"NOT_READ",
                      "net_neural":53751,"trained":8,
                      "output_gate":0},ensure_ascii=False,sort_keys=True),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    for name in ("source12","source_manifest","old_s114","s113_gold_folder","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    train(a.source12,a.source_manifest,a.old_s114,a.s113_gold_folder,a.out)

if __name__=="__main__":
    main()
