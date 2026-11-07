"""CALIBRI v230: original paid-211 codebooks, first-error answer restoration.
Precommitted TRAIN-only optimization; source is the SHA-verified physically
packed v228 step-12 checkpoint, never an unsealed optimizer state.
Unlike v226/v228, no repeat or surrogate-only selection objective.
The ONLY acceptance criterion is MORE complete greedy TRAIN answers
than the original 31/48 with NONE of the original 31 lost.
No public DEV/QA/VALID/TEST use; no inference-time sidecars; no LIVE.
"""
import argparse
from pathlib import Path
import json

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from v174.run import save
from v176.operators import clear
from v185.codec import replace_discrete
from v186.fit import paired_loss
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import (probe_greedy_train, summarize_sequences,
                          sequence_no_regression_gate)
from v227.run import correct_train_samples, already_correct_full_answer_loss
from v229.run import sha_checked, ACCEPTED_V228_STEP12_SHA256

SEED=230
STEPS=20
LR=3.0e-5
MARGIN=0.30
GRAD_CLIP=1.0
ANCHOR=0.04
CHECKPOINT_STEPS=(2,4,6,8,12,16,20)
TARGET_WEIGHT=0.55
PRESERVE_WEIGHT=0.35
OPPOSING_WEIGHT=0.10
# Fixed solely from independent sealed-v228 original TRAIN v229 first-error
# diagnostic; ordered nearest gold logit: first 6 wrong at token 1,
# next 2 wrong at token 2. Never select on QA or public DEV.
FROZEN_TARGETS=((4,"left"),(0,"left"),(3,"left"),(1,"left"),
                (2,"left"),(5,"left"),(12,"right"),(14,"right"))

def first_wrong_context(pair,side,generated):
    sample=pair[side]
    prompt=sample["prompt_ids"].detach().long().clone().cpu()
    gold=sample["targets"].detach().long().clone().cpu()
    predicted=generated[pair["pair_id"]]["sides"][side]["generated_tokens"]
    k=0
    while k<min(len(predicted),len(gold)) and predicted[k]==int(gold[k]):
        k+=1
    if k>=len(predicted) or k>=len(gold):
        raise AssertionError("Frozen first incorrect token not found")
    inp=torch.cat((prompt,torch.as_tensor(predicted[:k],dtype=torch.long)))
    return {"case":[pair["pair_id"],side],"input_ids":inp[None],
            "gold_token":int(gold[k]),"wrong_token":int(predicted[k]),
            "first_error_position":k+1}

def run(v228_path,v229_audit_path,out):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    audit=json.loads(Path(v229_audit_path).read_text())
    if (not audit.get("completed") or not audit.get("read_only") or
        audit.get("actual_reloaded_original_v220_TRAIN_full_correct")!=31 or
        audit.get("actual_reloaded_sealed_v228_TRAIN_full_correct")!=31 or
        audit.get("lost_original_correct_complete_TRAIN_answers") or
        not audit.get("no_public_DEV_VALID_TEST_QA_used")):
        raise AssertionError("Independent sealed v229 TRAIN audit not reproduced")
    source=sha_checked(v228_path,ACCEPTED_V228_STEP12_SHA256)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    chosen,pairs,valid,origin=training_inputs(tokenizer)
    if len(chosen)!=20 or len(pairs)!=24 or len(valid)!=P.VALID_PAIRS:
        raise AssertionError("Original 48 opposing TRAIN protocol drift")
    mapping={p["pair_id"]:p for p in pairs}
    if len(mapping)!=24 or any(p["split"]!="TRAIN" for p in pairs):
        raise AssertionError("Non TRAIN pair detected")
    student,paid=build_trainable_from_paid_source(source)
    student.cpu().eval()
    if len(paid)!=211:
        raise AssertionError("Missing one of the original paid 211 dictionaries")
    baseline=probe_greedy_train(student,paid,pairs,max_new_tokens=12)
    old=summarize_sequences(baseline)
    if (len(old["exact"])!=31 or old["prefix_tokens"]!=74 or
        old["repetition_excess"]!=3):
        raise AssertionError("Sealed v228 48 TRAIN answers not reproduced")
    prev_correct=correct_train_samples(
        {"all_48_opposing_TRAIN_rows":[
            {"pair_id":pid,"side":side,"v220_exact_answer":True}
            for pid,side in sorted(old["exact"])]},mapping)
    if len(prev_correct)!=31:
        raise AssertionError("Frozen 31 original correct full TRAIN answers changed")
    fragile=next((s for s in prev_correct
                  if s["pair_id"]==15 and s["side"]=="left"),None)
    if fragile is None:
        raise AssertionError("Previously regressed fragile correct TRAIN answer missing")
    rows={(r["pair_id"],r["side"]):r for r in
          audit["v228_first_wrong_gold_margin_nearest_top1"]}
    fixed=[]
    for key in FROZEN_TARGETS:
        r=rows.get(key)
        if (r is None or r.get("first_bad_1based") not in (1,2) or
            not (-0.65<float(r["gold_logit_minus_wrong_top1"])<-0.15)):
            raise AssertionError("Frozen original v229 first-error margin drift")
        x=first_wrong_context(mapping[key[0]],key[1],baseline)
        if x["first_error_position"]!=r["first_bad_1based"]:
            raise AssertionError("First wrong token differs from sealed v229")
        fixed.append(x)
    original_centroids={n:s.centroids.detach().clone()
                        for n,s in paid.items()}
    original_codes={n:s.codes.detach().clone()
                    for n,s in paid.items()}
    save(out/"PRECOMMITTED_v230_eight_first_error_TRAIN_cases.json",{
        "sealed_v228_SHA256":ACCEPTED_V228_STEP12_SHA256,
        "independent_original_v229_audit_consumed":True,
        "eight_frozen_near_gold_first_wrong_TRAIN_pairs":list(FROZEN_TARGETS),
        "only_original_211_FP16_codebooks_updated":True,
        "original_5bit_indices_unchanged":True,
        "steps":STEPS,"learning_rate":LR,"checkpoints":CHECKPOINT_STEPS,
        "margin":MARGIN,"loss_weights":{
          "first_wrong_gold":TARGET_WEIGHT,
          "previously_correct_full_answers":PRESERVE_WEIGHT,
          "opposing_TRAIN_pair":OPPOSING_WEIGHT,
          "source_centroid_anchor":ANCHOR},
        "precommitted_acceptance":"at least 32/48 complete TRAIN correct and no original correct answers lost",
        "repeat_counts_not_a_quality_target":True,
        "no_DEV_VALID_QA_TEST_used":True,
        "no_Jessica_tests":True,"no_LIVE_main_Render_Supabase_APK":True
    })
    optimizer=torch.optim.Adam([s.centroids for s in paid.values()],lr=LR)
    checkpoints=[]
    training=[]
    best=None
    best_correct=31
    for step in range(1,STEPS+1):
        clear(paid)
        optimizer.zero_grad(set_to_none=True)
        target=fixed[(step-1)%len(fixed)]
        keep=prev_correct[(step-1)%len(prev_correct)]
        keep2=prev_correct[(step+9)%len(prev_correct)]
        pair=pairs[(step-1)%len(pairs)]
        logits=student(input_ids=target["input_ids"],use_cache=False).logits[0,-1].float()
        gold=target["gold_token"]
        wrong=target["wrong_token"]
        correct_token_loss=F.cross_entropy(logits[None],torch.tensor([gold]))
        margin_loss=F.relu(MARGIN+logits[wrong]-logits[gold])
        targeted=correct_token_loss+0.5*margin_loss
        preserved=(already_correct_full_answer_loss(student,fragile)+
                   already_correct_full_answer_loss(student,keep)+
                   already_correct_full_answer_loss(student,keep2))/3
        opposing,_=paired_loss(student,None,pair)
        anchor=torch.stack([
            (s.centroids-original_centroids[n]).square().mean()
            for n,s in paid.items()]).mean()
        loss=(TARGET_WEIGHT*targeted+PRESERVE_WEIGHT*preserved+
              OPPOSING_WEIGHT*opposing+ANCHOR*anchor)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite first-error answer restoration")
        loss.backward()
        active=sum(s.centroids.grad is not None for s in paid.values())
        if active!=211:
            raise AssertionError("First-error restoration missed paid dictionaries")
        norm=float(torch.nn.utils.clip_grad_norm_(
            [s.centroids for s in paid.values()],GRAD_CLIP,error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in paid.values():
                s.centroids.clamp_(-16,16)
        clear(paid)
        training.append({"step":step,"target":target["case"],
            "preserve_cases":[[fragile["pair_id"],fragile["side"]],
              [keep["pair_id"],keep["side"]],[keep2["pair_id"],keep2["side"]]],
            "all_211_paid_gradients":active,
            "loss_diagnostic_only":float(loss.detach()),
            "gradient_norm":norm})
        print(json.dumps({"stage":"v230_first_wrong_TRAIN_gradient",
                          "step":step,"target":target["case"],
                          "all_211_paid":active}),flush=True)
        if step not in CHECKPOINT_STEPS:
            continue
        output=probe_greedy_train(student,paid,pairs,max_new_tokens=12)
        summ=summarize_sequences(output)
        gate=sequence_no_regression_gate(baseline,output)
        missing=sorted(old["exact"]-summ["exact"])
        clear(paid)
        packed,seal=replace_discrete(source,paid,strict=False)
        if (len(packed)!=P.SOURCE_BYTES or seal["complete_bpp"]>0.7 or
           seal["actual_changed_5bit_group_indices"]!=0 or
           seal["all_2d_original_matrices"]!=211 or
           any(not torch.equal(s.codes,original_codes[n])
               for n,s in paid.items())):
            raise AssertionError("Paid CB59 source altered physical geometry")
        accept=(len(summ["exact"])>best_correct and not missing and
                seal["actually_changed_FP16_codebook_cells"]>0)
        row={"checkpoint":step,"whole_TRAIN_exact":len(summ["exact"]),
             "source_whole_TRAIN_exact":31,
             "new_correct_whole_TRAIN":sorted(summ["exact"]-old["exact"]),
             "lost_original_correct_whole_TRAIN":missing,
             "gold_prefix_total":summ["prefix_tokens"],
             "repetitions_diagnostic_only":summ["repetition_excess"],
             "physical_bpp":seal["complete_bpp"],
             "FP16_paid_cells_changed":seal["actually_changed_FP16_codebook_cells"],
             "unchanged_original_5bit_codes":True,"accepted":accept,
             "sha256":seal["sha256"]}
        checkpoints.append(row)
        print(json.dumps({"stage":"v230_ACTUAL_48_full_TRAIN_answers",**row}),flush=True)
        if accept:
            best_correct=len(summ["exact"])
            best=(packed,seal,step)
    result={"completed":True,"source_full_TRAIN_exact":31,
            "checkpoints":checkpoints,"accepted":best is not None,
            "no_DEV_VALID_TEST_QA_selection":True,
            "no_Jessica_test":True,
            "not_independent_FP32_parity_evidence":True}
    if best is not None:
        packed,seal,step=best
        (out/"CALIBRI_v230_ACCEPTED_paid_first_wrong_TRAIN.cb59").write_bytes(packed)
        result["accepted_step"]=step
        result["accepted_sha256"]=seal["sha256"]
        result["physical_bpp"]=seal["complete_bpp"]
    else:
        result["rejected"]="No full-TRAIN-answer improvement over 31/48 without regression"
    save(out/"ACTUAL_v230_full_TRAIN_48_answers_7_checkpoints.json",result)
    save(out/"ACTUAL_v230_all_211_paid_TRAIN_gradient_steps.json",{
      "gradients":training,"original_5bit_indices_unchanged":True})
    print(json.dumps({"stage":"v230_FINAL_full_answer_gate",
       "accepted":result["accepted"],"accepted_step":result.get("accepted_step"),
       "max_48_full_TRAIN_correct":max(x["whole_TRAIN_exact"] for x in checkpoints),
       "original_48_full_TRAIN_correct":31,
       "each_original_31_preserved_if_accepted":True,
       "FP32_QA_parity_not_tested":True}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("v228_paid",type=Path)
    p.add_argument("v229_audit",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.v228_paid,a.v229_audit,a.out)
