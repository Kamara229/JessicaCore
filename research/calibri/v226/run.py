"""CALIBRI v226: all 211 paid-codebook TRAIN-only anti-loop restoration.

Frozen v225 shows 5 repeated token decisions in 4 TRAIN examples,
none selected by authentic FP32 or v206 when using the SAME v220
prefix. Train ALL original 211 paid FP16 dictionaries jointly to
restore gold next tokens and opposing whole TRAIN answers. Evaluate
real bounded TRAIN greedy full sequences before accepting physical
source. No new runtime processor, no extra weights, no changing 5-bit
indices, no QA/DEV/VALID/TEST optimizer or selection and no LIVE.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v176.operators import clear
from v185.codec import replace_discrete
from v186.fit import paired_loss
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import (probe_greedy_train, summarize_sequences,
                          sequence_no_regression_gate)
from v212.run import SOURCE_V206_SHA256
from v221.run import V220_SOURCE_SHA256

SEED=226
STEPS=16
LEARNING_RATE=0.00008
MARGIN=1.5
GRAD_CLIP=1.0
PAIR_WEIGHT=0.55
REPEAT_WEIGHT=0.45

def authenticate(v220_path,v222_path,v224_path,v225_path):
    raw=Path(v220_path).read_bytes()
    if len(raw)!=P.SOURCE_BYTES or sha256(raw).hexdigest()!=V220_SOURCE_SHA256:
        raise AssertionError("Not original immutable native v220 paid archive")
    rollout=json.loads(Path(v222_path).read_text())
    audit=json.loads(Path(v224_path).read_text())
    cause=json.loads(Path(v225_path).read_text())
    if (not rollout.get("completed")
            or not rollout.get("never_train_or_update_source")
            or not rollout.get("public_DEV16_not_reused")
            or rollout.get("source_hashes")!={"v206":SOURCE_V206_SHA256,
                                             "v220":V220_SOURCE_SHA256}):
        raise AssertionError("Not the verified TRAIN-only v222 output")
    if (not audit.get("completed")
            or audit.get("sources")!={"v206":0,"v220":5}
            or audit.get("v220_repeat_position_histogram_1based")!={"3":4,"4":1}):
        raise AssertionError("Not the original five v224 repeated positions")
    if (not cause.get("completed") or not cause.get("strict_TRAIN_only")
            or cause.get("summary",{}).get("repeat_choice_on_v220_prefix_counts")
               != {"FP32":0,"v206":0,"v220":5}
            or cause.get("immutable_original_v206_v220_archive_sha256")
               != {"v206":SOURCE_V206_SHA256,"v220":V220_SOURCE_SHA256}):
        raise AssertionError("Original v225 isolated cause study not completed")
    originals=rollout["all_48_opposing_TRAIN_rows"]
    if len(originals)!=48 or len({
        (r["pair_id"],r["side"]) for r in originals})!=48:
        raise AssertionError("v222 frozen TRAIN rows incomplete")
    return CB59(raw),rollout,audit,cause


def build_frozen_pairs(rollout):
    baseline={}
    for r in rollout["all_48_opposing_TRAIN_rows"]:
        pid=r["pair_id"]
        if pid not in baseline:
            baseline[pid]={"domain":r["domain"],"sides":{}}
        if baseline[pid]["domain"]!=r["domain"]:
            raise AssertionError("TRAIN domain changed")
        generated=r["v220_generated_token_ids"]
        gold=r["gold_token_ids_including_EOS"]
        baseline[pid]["sides"][r["side"]]={
            "gold_tokens_with_EOS":gold,
            "generated_tokens":generated,
            "correct_prefix_tokens":r["v220_gold_prefix"],
            "exact_with_EOS":r["v220_exact_answer"],
            "adjacent_repetition_excess":r["v220_repeat_excess"],
            "stopped_at_EOS":bool(generated and generated[-1]==gold[-1]),
        }
    if len(baseline)!=24 or any(set(r["sides"])!={"left","right"}
                                    for r in baseline.values()):
        raise AssertionError("Not original 24 paired TRAIN baseline")
    if (summarize_sequences(baseline)["repetition_excess"]!=5 or
            len(summarize_sequences(baseline)["exact"])!=31):
        raise AssertionError("Original v220 TRAIN metrics drifted")
    return baseline


def five_repeat_cases(audit,cause,pairs):
    contexts=[]
    for entry in audit["v220_repeat_details"]:
        key=(entry["pair_id"],entry["side"])
        pair=pairs[entry["pair_id"]]
        if pair["split"]!="TRAIN" or key[1]!="left":
            raise AssertionError("Repeat example leaked outside original TRAIN")
        sample=pair[key[1]]
        expected=sample["prompt_ids"].detach().clone().long()[None]
        targets=[int(t) for t in sample["targets"]]
        generated=entry["candidate_first_12_token_ids"]
        for pos in entry["positions"]:
            i=pos["position_1based"]
            prev=int(generated[i-2])
            if i not in (3,4) or prev!=int(generated[i-1]):
                raise AssertionError("Invalid original repeated token event")
            context=torch.cat((expected,expected.new_tensor([generated[:i-1]])),1)
            gold=int(targets[i-1]) if len(targets)>=i else int(targets[-1])
            if gold==prev:
                # A legitimate gold token repeat must never be forcibly
                # penalized, even when v220 happened to repeat it.
                raise AssertionError("Precommitted anti-loop target repeats gold")
            contexts.append({
                "TRAIN_pair_id":entry["pair_id"],"side":key[1],
                "repeat_position_1based":i,"input_ids":context,
                "wrong_repeated_token_id":prev,
                "gold_next_token_id":gold,
            })
    if len(contexts)!=5 or Counter(x["repeat_position_1based"]
        for x in contexts)!=Counter({3:4,4:1}):
        raise AssertionError("Frozen five repetition cases lost")
    pairs_in_cause={(d["pair_id"],d["side"],d["repeat_position_1based"])
                    for d in cause["details"]}
    if {(x["TRAIN_pair_id"],x["side"],x["repeat_position_1based"])
             for x in contexts}!=pairs_in_cause:
        raise AssertionError("Different repeat contexts from v225")
    return contexts


def fit(student,paid,pairs,cases,out):
    frozen_indices={name:src.codes.detach().clone()
                    for name,src in paid.items()}
    optimizer=torch.optim.Adam(
        [s.centroids for s in paid.values()],lr=LEARNING_RATE)
    records=[]
    for step in range(STEPS):
        clear(paid)
        optimizer.zero_grad(set_to_none=True)
        # All 24 frozen TRAIN opposing pairs rotate without dynamic QA/VALID
        # feedback or hyperparameter adjustment. Four passes over 5 cases.
        pair=pairs[step%len(pairs)]
        context=cases[step%len(cases)]
        pair_ce,info=paired_loss(student,None,pair)
        logits=student(input_ids=context["input_ids"],use_cache=False
                       ).logits[0,-1].float()
        gold=context["gold_next_token_id"]
        wrong=context["wrong_repeated_token_id"]
        gold_term=F.cross_entropy(logits[None,:],
                                  logits.new_tensor([gold],dtype=torch.long))
        contrast=F.relu(MARGIN+logits[wrong]-logits[gold])
        repeat_loss=gold_term+0.5*contrast
        loss=PAIR_WEIGHT*pair_ce+REPEAT_WEIGHT*repeat_loss
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite TRAIN-only answer restoration")
        loss.backward()
        active=sum(s.centroids.grad is not None for s in paid.values())
        if active!=211:
            raise AssertionError("Not all original 211 paid codebooks trained")
        norm=float(torch.nn.utils.clip_grad_norm_(
            [s.centroids for s in paid.values()],GRAD_CLIP,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in paid.values():
                s.centroids.clamp_(-16,16)
        clear(paid)
        row={
            "step":step+1,"TRAIN_pair_id":pair["pair_id"],
            "TRAIN_repeat_pair_id":context["TRAIN_pair_id"],
            "TRAIN_repeat_position":context["repeat_position_1based"],
            "all_211_paid_codebooks_gradients":active,
            "actual_opposing_answer_loss_diagnostic":float(pair_ce.detach()),
            "original_repeated_position_gold_loss_diagnostic":float(gold_term.detach()),
            "original_repeated_position_margin_diagnostic":float(contrast.detach()),
            "gradient_norm":norm,"no_QA_or_VALID_optimizer":True,
        }
        records.append(row)
        print(json.dumps({"stage":"v226_paid_TRAIN_restoration_step",
                          "step":step+1,"all_source_matrices":active,
                          "TRAIN_repeat":context["TRAIN_pair_id"],
                          "repeat_position":context["repeat_position_1based"]}),
              flush=True)
    if any(not torch.equal(s.codes,frozen_indices[n])
           for n,s in paid.items()):
        raise AssertionError("Original physical 5-bit row index mutation")
    save(out/"ACTUAL_v226_16step_TRAIN_fit_only.json",{
        "steps":STEPS,"completed":True,
        "every_step_all_211_paid_codebooks_optimized":True,
        "unchanged_original_5bit_indices":True,
        "metrics_are_TRAIN_diagnostics_NOT_quality_gate":True,
        "steps_log":records,
        "never_fit_or_select_on_public_DEV_VALID_TEST":True,
    })


def run(v220_path,v222_path,v224_path,v225_path,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    archive,rollout,audit,cause=authenticate(
        v220_path,v222_path,v224_path,v225_path)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    selected,pairs,valid_pairs,origin=training_inputs(tok)
    if len(selected)!=20 or len(pairs)!=24 or len(valid_pairs)!=P.VALID_PAIRS:
        raise AssertionError("Frozen TRAIN protocol changed")
    if any(p["split"]!="TRAIN" for p in pairs):
        raise AssertionError("No VALID pairs can be used for training")
    cases=five_repeat_cases(audit,cause,{p["pair_id"]:p for p in pairs})
    baseline=build_frozen_pairs(rollout)
    save(out/"PRECOMMITTED_v226_TRAIN_only_paid_anti_repeat_protocol.json",{
        "physical_initial_v220_sha256":V220_SOURCE_SHA256,
        "new_sealed_source_max_bpp":0.7,
        "native_original_5bit_indices_unchanged":True,
        "model_operators":211,"steps":STEPS,"learning_rate":LEARNING_RATE,
        "gradient_clip":GRAD_CLIP,"repeat_margin":MARGIN,
        "frozen_original_TRAIN_repeat_positions":len(cases),
        "TRAIN_opp_pairs":len(pairs),
        "objective_weights":{"paired":PAIR_WEIGHT,"specific_repeat":REPEAT_WEIGHT},
        "full_TRAIN_answer_evaluation_before_acceptance":True,
        "original_v220_TRAIN_exact_answers":len(summarize_sequences(
            baseline)["exact"]),
        "original_v220_TRAIN_adjacent_repeats":
            summarize_sequences(baseline)["repetition_excess"],
        "no_DEV_VALID_TEST_QA_optim_or_source_selection":True,
        "no_new_runtime_decoder_or_unpaid_sidecar":True,
        "no_LIVE_Jessica_main_Render_Supabase_APK_changes":True,
    })
    student,paid=build_trainable_from_paid_source(archive)
    student.eval()
    if len(paid)!=211:
        raise AssertionError("Original source model lost paid matrices")
    fit(student,paid,pairs,cases,out)
    # Physical codec is verified even for candidates later rejected on TRAIN.
    clear(paid)
    packed,seal=replace_discrete(archive,paid,strict=False)
    if (len(packed)!=P.SOURCE_BYTES or seal["complete_bpp"]>0.7
            or seal["actual_changed_5bit_group_indices"]!=0
            or seal["actually_changed_FP16_codebook_cells"]<=0
            or seal["sha256"]==V220_SOURCE_SHA256):
        raise AssertionError("Training failed to generate a paid packed model")
    save(out/"FROZEN_v226_physical_candidate_BEFORE_TRAIN_acceptance.json",{
        "sha256":seal["sha256"],"whole_physical_bpp":seal["complete_bpp"],
        "all_211_operators":seal["all_2d_original_matrices"],
        "FP16_codebook_cells_changed":
            seal["actually_changed_FP16_codebook_cells"],
        "original_5bit_indices_changed":
            seal["actual_changed_5bit_group_indices"],
        "no_approximate_dense_inference":True,"no_unpaid_sidecar":True,
        "no_public_QA_access_or_selection":True,
    })
    # The original 12-token full answer correctness and repeats, not
    # CE/KL/NLL, are the sole in-TRAIN candidate acceptance gate.
    candidate=probe_greedy_train(student,paid,pairs,max_new_tokens=12)
    gate=sequence_no_regression_gate(baseline,candidate)
    summ=summarize_sequences(candidate)
    result={
        "completed":True,
        "original_v220_TRAIN_exact":len(summarize_sequences(baseline)["exact"]),
        "v226_candidate_TRAIN_exact":len(summ["exact"]),
        "original_v220_TRAIN_adjacent_repeats":
            summarize_sequences(baseline)["repetition_excess"],
        "v226_candidate_TRAIN_adjacent_repeats":summ["repetition_excess"],
        "original_v220_TRAIN_prefix_tokens":
            summarize_sequences(baseline)["prefix_tokens"],
        "v226_candidate_TRAIN_prefix_tokens":summ["prefix_tokens"],
        "did_not_lose_any_previously_correct_TRAIN_answers":
            not gate["lost_original_complete_TRAIN_answers"],
        "TRAIN_full_sequence_no_regression_gate":gate,
        "physical_candidate_sha256":seal["sha256"],
        "physical_candidate_bpp":seal["complete_bpp"],
        "QA_DEV_VALID_TEST_run":False,
        "native_CPU_speed_NOT_MEASURED":True,
        "FP32_DEV_quality_parity_NOT_CLAIMED":True,
        "Jessica_tests_ran":False,
    }
    save(out/"ACTUAL_v226_TRAIN_greedy_48_gate.json",result)
    print(json.dumps({"stage":"v226_actual_TRAIN_full_answer_gate",
        "v220_exact":result["original_v220_TRAIN_exact"],
        "v226_exact":result["v226_candidate_TRAIN_exact"],
        "v220_repeats":result["original_v220_TRAIN_adjacent_repeats"],
        "v226_repeats":result["v226_candidate_TRAIN_adjacent_repeats"],
        "gate_accept":gate["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"],
        "bpp":seal["complete_bpp"],
        "original_5bit_indices_unchanged":True}),flush=True)
    if gate["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"]:
        target=out/"CALIBRI_v226_ACCEPTED_TRAIN_only_paid_211.cb59"
        target.write_bytes(packed)
        print(json.dumps({"stage":"v226_TRAIN_ACCEPTED_physical_source",
                          "sha256":seal["sha256"],
                          "not_a_public_QA_or_native_CPU_result":True}),
              flush=True)
    else:
        # Rejected candidate is not written as a usable archive. TRAIN
        # diagnostics remain for research, not used as FP32 quality evidence.
        print(json.dumps({"stage":"v226_REJECTED_TRAIN_no_distributable_source",
                          "reason":"previous_correct_answer_or_prefix_or_repetition_regression",
                          "no_model_source_exported":True}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("sealed_v220",type=Path)
    ap.add_argument("v222_original_TRAIN_rollouts",type=Path)
    ap.add_argument("v224_original_TRAIN_repeat_audit",type=Path)
    ap.add_argument("v225_original_TRAIN_logits",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    run(a.sealed_v220,a.v222_original_TRAIN_rollouts,
        a.v224_original_TRAIN_repeat_audit,a.v225_original_TRAIN_logits,a.out)
