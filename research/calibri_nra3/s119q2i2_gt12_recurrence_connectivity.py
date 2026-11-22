"""S119q2i2: physical 43M GT12 recurrent connection to Core/NRA3/NPA1.

Prior q2i1 actual Russian multi-BPE+EOS joint training learned a
247,040-param native head and 120,999-param new Core/NRA3/NPA1/rank8
child, but all models had zero complete Russian DEV answers. The GT
physical architecture is installed, however its genuine S142 GT12
gates are all ZERO and the prompt-conditioned S111 coefficients are
exactly ZERO/frozen; new neural source features are injected AFTER the
GT layer5 hook. Therefore no evidence yet that the new regions benefit
from recurrent full-layer GT state.

This stage makes the link ACTUALLY trainable, in ISOLATED RESEARCH:
start with a verified identical physically trained q2i1 contextual FP16
child and original S142 Core+GT, freeze original Core, original LM head,
and q2i1 child. Optimizer changes ONLY the 12 full-layer GT gates.
Train 3 complete passes over exactly 12 distinct authored Russian TRAIN
rows (two per six domains); full native answer BPE+native EOS with
causal teacher forcing. TWO matched 12-param candidate routes:
  RECURRENT: original S126 mix_logits=0 (sigmoid 0.5 cross-layer carry)
  LOCAL: frozen mix_logits=-20 (no meaningful cross-layer carry)
Both have the same 12 trainable gate params, identical initial q2i1
neural weights, source, target BPE, optimizer, epochs and budget.
The LOCAL condition is a deliberately diagnostic state reset, not a
new improved GT architecture; it stays restricted to original existing
S126 shared low-rank projected vectors. Neither re-trains original GT
encoder/decoder nor original trained 43M or 247k q2i1 language head.

Evaluate actual original full-native OWN-TOKEN greedy DEV on the
previously scored 12 q2i1 DEV questions: baseline gate0, trained
recurrent GT12, trained local no-carry GT12, each with real EOS. DEV
was already scored, so no independent-blind generalization claim.
All hooks, GT12 mix/gate tensor states restored in memory after this
experiment; only isolated research FP16 GT12 gate sidecars are saved.
No old S118-B GOLD/EVAL128/J7, Jessica chat, LIVE/Render/Supabase/main.
Never promote; output_gate=0, candidate_accepted=false.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CONTEXT,LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT,VOCAB_SIZE,OLD_PREFIX,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CONTEXT,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as ru_rows,provenance as ru_provenance,KINDS,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    JointRuNative,NativeHeadAlignment,STAGE as Q2I1_STAGE,
    CONTEXT_CKPT as Q2I1_CTX,REPORT as Q2I1_REPORT,
    verified_native_answer,real_joint_native,
)
from research.calibri_nra3.s119q2g_fresh_own_token_source import normalized

STAGE="CALIBRI_S119Q2I2_REAL43M_GT12_RECURRENCE_LINK_NRA3_NPA1_NATIVE_EOS"
REPORT="S119Q2I2_GT12_LINK_3ROUNDS_TRAIN_AND_OLDDEV_OWN_BPE_REPORT.json"
RECURRENT_WEIGHTS="S119Q2I2_GT12_RECURRENT_GATES_FP16_G0.safetensors"
LOCAL_WEIGHTS="S119Q2I2_GT12_LOCAL_NOCARRY_GATES_FP16_G0.safetensors"
SEED=119141
TRAIN_PER_KIND=2
TRAIN_ROWS=12
DEV_ROWS=12
ROUNDS=3
MAX_OWN_NATIVE=12
GT_GATE_LR=0.025
MINUS_20=-20.0
GT_DEPTH=12
GT_RANK=4
GT_WIDTH=320
CONDITIONS=("GT_ZERO_BASELINE","GT_RECURRENT","GT_NOCARRY_LOCAL")


def native_full_gt_gate_contract(full,bridge):
    check((full.width,full.depth,full.rank)==(GT_WIDTH,GT_DEPTH,GT_RANK)
          and full.gates.shape==(GT_DEPTH,)
          and bool(torch.equal(full.gates.detach(),
                               torch.zeros_like(full.gates)))
          and full.mix_logits.shape==(GT_DEPTH,)
          and bool(torch.equal(full.mix_logits.detach(),
                               torch.zeros_like(full.mix_logits)))
          and bridge.coefficients.shape[0]==2
          and int(torch.count_nonzero(bridge.coefficients.detach()))==0
          and not bridge.coefficients.requires_grad,
          "Original S142 GT12 has 12 ZERO gates; S111 frozen bridge ZERO: no active learned new-area GT gradient")
    return {
        "original_S142_GT12_12_gates_exact_ZERO":True,
        "original_S142_GT12_mix_logits_exact_ZERO":True,
        "original_S111_conditional_bridge_ALL_340_coefficients_ZERO_frozen":True,
        "actual_GreenTunnel_and_new_regions_share_model_forward":True,
        "BUT_GT12_original_route_has_no_active_recurrent_gain":True,
    }


def load_q2i1_physical_context(file,metadata,prior_child):
    check(file.name==Q2I1_CTX
          and sha(file)==metadata["sha256"]
          and file.stat().st_size==metadata["file_bytes"],
          "q2i1 ORIGINAL native Russian-language trained child FP16 provenance")
    state=load_file(str(file),device="cpu")
    check(state and all(k.startswith(OLD_PREFIX)
                        and v.dtype==torch.float16 for k,v in state.items())
          and sum(v.numel() for v in state.values())==
              metadata["saved_total_child_FP16_params"],
          "physical Q2I1 complete trained q2f + native full head tensors")
    m=JointRuNative(prior_child)
    rekey={k[len(OLD_PREFIX):]:v.float() for k,v in state.items()}
    check(set(rekey)==set(m.state_dict()),
          "entire physical fitted Core+NRA3+NPA1+GT-input child+language head")
    m.load_state_dict(rekey,strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    # Q2i1 child is FROZEN in this first GT-only causal study. This
    # requires-grad flag ensures S119q1 original hook delta.requires_grad,
    # but its parameter is not in the optimizer and NEVER updated.
    m.child.child.midlayer_gates.requires_grad_(True)
    check(m.output_gate==m.child.output_gate==0 and
          not m.candidate_accepted and not m.child.candidate_accepted,
          "NO candidate activation, full pretrained neural child frozen")
    return m


def set_gt_route(full,condition,gate_values):
    check(condition in CONDITIONS and gate_values.shape==(GT_DEPTH,)
          and bool(torch.isfinite(gate_values).all())
          and not full._active and full._carry is None,
          "set only research physical GT route between original model passes")
    with torch.no_grad():
        full.gates.copy_(gate_values)
        full.mix_logits.fill_(
            MINUS_20 if condition=="GT_NOCARRY_LOCAL" else 0.)
    check(not full._active and full._carry is None
          and bool(torch.allclose(full.gates.detach(),gate_values))
          and (bool(torch.all(full.mix_logits.detach()==MINUS_20))
               if condition=="GT_NOCARRY_LOCAL"
               else bool(torch.count_nonzero(full.mix_logits.detach())==0)),
          "local stage use only off-line fixed original GT12 mixing")


def original_native_gt_step(model,sources,bridge,full,expansion,
                            child,entry,prefix,target,*,contextual):
    check(len(prefix)<160 and 0<=target<VOCAB_SIZE
          and entry["split"] in ("TRAIN","DEV"),
          "original native prefix only on allowed q2i1 source, no unseen labels")
    # Q2I1 learned output head and NRA3/NPA/Core child remain IDENTICAL
    # frozen FP16-in-memory across recurrent vs local gate-only rounds.
    logits=real_joint_native(
        model,sources,bridge,full,expansion,child,
        prefix,entry["prompt_features"],contextual=contextual)
    check(logits.shape==(1,VOCAB_SIZE) and
          full.last_forward_coverage==tuple(range(GT_DEPTH))
          and not full._active and full._carry is None,
          "full original GT12 hooks actually covered all decoder layers this pass")
    loss=F.cross_entropy(
        logits,torch.tensor([int(target)],dtype=torch.long))
    return loss,logits


def train_gate_rounds(model,sources,bridge,full,expansion,
                      child,train,*,condition):
    check(len(train)==TRAIN_ROWS
          and all(e["split"]=="TRAIN" for e in train)
          and condition in ("GT_RECURRENT","GT_NOCARRY_LOCAL"),
          "original-new RU TRAIN-only 2x6 examples; no DEV gradient")
    check(sum(p.numel() for p in (full.gates,))==12,
          "exact 12 active learned GT gates only")
    for name,p in full.named_parameters():
        p.requires_grad_(name=="gates")
    child_gates_snapshot=child.child.child.midlayer_gates.detach().clone()
    opt=torch.optim.Adam([full.gates],lr=GT_GATE_LR)
    hist=[];real_calls=0;optimizations=0;grad_norm_first=None
    for round_idx in range(ROUNDS):
        ce=[];eos_ce=[];native_targets=0
        for e in train:
            gold=e["answer_native_with_eos"]
            opt.zero_grad(set_to_none=True)
            for t,token in enumerate(gold):
                native_prefix=e["prefix"]+gold[:t]
                check(native_prefix[:len(e["prefix"])]==e["prefix"]
                      and native_prefix[len(e["prefix"]):]==gold[:t]
                      and token==gold[t],
                      "strict no current/future target in original GT decoder prefix")
                loss,logits=original_native_gt_step(
                    model,sources,bridge,full,expansion,child,e,
                    native_prefix,token,contextual=True)
                (loss/len(gold)).backward()
                native_targets+=1;real_calls+=1
                ce.append(float(loss.detach()))
                if t==len(gold)-1:
                    eos_ce.append(float(loss.detach()))
                del logits,loss
            if grad_norm_first is None:
                g=full.gates.grad
                check(g is not None and g.shape==(GT_DEPTH,)
                      and bool(torch.isfinite(g).all())
                      and float(torch.linalg.vector_norm(g))>1.e-15,
                      "REAL full native 49152 CE backward reached GT12 gate from NRA3/NPA1 forward")
                grad_norm_first=float(torch.linalg.vector_norm(g))
            torch.nn.utils.clip_grad_norm_([full.gates],1.)
            opt.step();optimizations+=1
            check(bool(torch.isfinite(full.gates).all())
                  and bool((full.gates.detach().abs()<1.).all()),
                  "research GT12 gates bounded by independent strict cap")
        hist.append({
            "round":round_idx+1,
            "physical_original_12layer_Core_GT_NATIVE_forwards":native_targets,
            "GT12_gate_ADAM_updates":TRAIN_ROWS,
            "TRAIN_mean_full_native_answer_BPE_CE_and_EOS":
                sum(ce)/len(ce),
            "TRAIN_mean_true_EOS_native_target_CE":
                sum(eos_ce)/len(eos_ce),
            "GT12_gate_max_absolute":
                float(full.gates.detach().abs().max()),
            "same_original_12_train_source_and_frozen_child":True,
        })
    check(optimizations==TRAIN_ROWS*ROUNDS and real_calls>TRAIN_ROWS*ROUNDS
          and grad_norm_first is not None
          and float(full.gates.detach().abs().max())>1.e-8
          and torch.equal(
              child_gates_snapshot,child.child.child.midlayer_gates.detach()),
          "physically trained 12 GT gates three rounds, not new q2i1 weights")
    return {
        "rounds":hist,
        "condition":condition,
        "physical_original_decoder_forwards":real_calls,
        "actual_GT12_only_optimizer_updates":optimizations,
        "number_trainable_GT12_parameters":12,
        "first_true_full_head_12_GT_gate_gradient_norm":grad_norm_first,
        "result_GT12_gates_12":full.gates.detach().cpu().tolist(),
        "full_original_output_head_and_child_not_trainable":True,
    }


def own_token_dev(model,sources,bridge,full,expansion,child,example,*,condition):
    check(example["split"]=="DEV" and condition in CONDITIONS
          and MAX_OWN_NATIVE<=16,
          "old known source q2i1 DEV only, never used in optimization")
    ids=[];prefix=list(example["prefix"]);emitted_eos=False
    from transformers import AutoTokenizer
    for _ in range(MAX_OWN_NATIVE):
        check(prefix==example["prefix"]+ids
              and len(prefix)<160,
              "each physical GT native model prefill uses only its OWN emitted tokens")
        logits=real_joint_native(
            model,sources,bridge,full,expansion,child,
            prefix,example["prompt_features"],contextual=True)
        check(logits.shape==(1,VOCAB_SIZE)
              and full.last_forward_coverage==tuple(range(GT_DEPTH))
              and not full._active and full._carry is None,
              "one real original Core12 GT12 LM head per native generated token")
        predicted=int(torch.argmax(logits.detach(),dim=-1))
        ids.append(predicted);prefix.append(predicted)
        del logits
        if predicted==example["eos_native_id"]:
            emitted_eos=True
            break
    decoded=example["tokenizer"].decode(ids,skip_special_tokens=True)
    correct_text=normalized(decoded)==normalized(
        example["gold_answer_for_scoring_ONLY"])
    return {
        "source_id":example["id"],
        "domain":example["kind"],
        "variant":condition,
        "original_full_decoder_calls":len(ids),
        "generated_own_native_BPE_ids":ids,
        "decoded_text":decoded,
        "native_EOS":emitted_eos,
        "truncated_without_eos":not emitted_eos,
        "text_exact_even_if_not_finished":correct_text,
        "strict_finished_full_answer_exact":
            correct_text and emitted_eos,
        "teacher_forcing_in_DEV":False,
        "all_core_NRA3_NPA1_language_head_child_weights_FROZEN":True,
    }


def safe_gt_gate_sidecar(path,gates,condition):
    check(not path.exists() and
          gates.shape==(GT_DEPTH,)
          and bool(torch.isfinite(gates).all())
          and condition in ("GT_RECURRENT","GT_NOCARRY_LOCAL"),
          "isolated research-only GT12 learned gates not original model")
    data={"research.s119q2i2.fullGT12_gates_fp16":
          gates.detach().cpu().half().contiguous()}
    save_file(data,str(path))
    read=load_file(str(path),device="cpu")
    check(set(read)==set(data)
          and all(torch.equal(read[k],v) for k,v in data.items()),
          "true physical GT12 FP16 sidecar roundtrip exact")
    return {"file_name":path.name,"sha256":sha(path),
            "file_bytes":path.stat().st_size,
            "12_GT12_trainable_only":True,"condition":condition}


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,out):
    check(not out.exists(),"isolated unaccepted GT sidecars, no overwrites")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source_meta=ru_provenance()
    source_train=ru_rows("TRAIN")
    source_dev=ru_rows("DEV")
    check(source_meta["TRAIN_rows"]==36 and
          source_meta["DEV_rows"]==DEV_ROWS,
          "only new handwritten q2i1 source, no reserved EVAL")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CONTEXT,q2d_dir/Q2D_LOCAL,
        s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE and
          q2i1["candidate_accepted"] is False and
          q2i1["output_gate"]==0 and
          q2i1["original_S142_43M_GT_SHA256"]==sha(s142) and
          q2i1["original_S93_SHA256"]==sha(s93) and
          q2i1["prior_physically_trained_q2f_report_sha256"]==
              sha(q2f_dir/Q2F_REPORT) and
          q2i1["original_author_new_Russian_source_provenance"]==source_meta
          and q2f["parent_q2d_training_report_SHA256"]==
              sha(q2d_dir/Q2D_REPORT),
          "true prior physical trained q2i1 child+source provenance exact")
    base_q2d,_=load_q2d_trained_child(
        q2d_dir/Q2D_CONTEXT,
        q2d["new_research_checkpoint_contextual"])
    prior_q2f=load_physical_q2f_child(
        q2f_dir/Q2F_CONTEXT,
        q2f["projector_contextual_child_checkpoint"],base_q2d)
    joint=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],prior_q2f)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "true ORIGINAL 49152 native BPE and EOS only")
    train=[verified_native_answer(tok,x) for x in source_train
           if int(x["id"].split("_")[-1])<TRAIN_PER_KIND]
    dev=[verified_native_answer(tok,x) for x in source_dev]
    check(len(train)==TRAIN_ROWS and len(dev)==DEV_ROWS
          and {x["prompt_sha256"] for x in train}.isdisjoint(
              x["prompt_sha256"] for x in dev)
          and Counter(x["kind"] for x in train)=={k:2 for k in KINDS},
          "exact original 2x6 train source and already scored q2i1 2x6 DEV")
    for e in dev:
        e["eos_native_id"]=int(tok.eos_token_id)
        e["tokenizer"]=tok
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",joint)
    model.eval()
    frozen_digest=old_original_param_digest(model)
    child_state={k:v.detach().cpu().clone()
                 for k,v in joint.state_dict().items()}
    original_full_gt={
        k:v.detach().cpu().clone()
        for k,v in full.state_dict().items()}
    gt_initial=native_full_gt_gate_contract(full,bridge)
    original_12_zero=full.gates.detach().clone()
    original_12_mix=full.mix_logits.detach().clone()
    check(all(not p.requires_grad for name,p in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "every ORIGINAL S142 43M Core12 and native LM head frozen")
    for name,p in full.named_parameters():
        p.requires_grad_(name=="gates")
    reports={};gate_sidecars={}
    gt_forwards=0;dev_forwards=0
    trained_gates={}
    out.mkdir(parents=True)
    try:
        # Real identical pre-training q2i1 child + zero GT gates A/B.
        for condition in ("GT_RECURRENT","GT_NOCARRY_LOCAL"):
            set_gt_route(full,condition,original_12_zero)
            result=train_gate_rounds(
                model,sources,bridge,full,expansion,joint,
                train,condition=condition)
            trained_gates[condition]=full.gates.detach().clone()
            reports[condition]=result
            gt_forwards+=result["physical_original_decoder_forwards"]
            check(old_original_param_digest(model)==frozen_digest
                  and all(torch.equal(v,joint.state_dict()[k].detach().cpu())
                          for k,v in child_state.items()),
                  "joint neural child and original 43M HEAD NOT trained via GT optimizer")
        check(reports["GT_RECURRENT"][
                  "number_trainable_GT12_parameters"]==
              reports["GT_NOCARRY_LOCAL"][
                  "number_trainable_GT12_parameters"]==12,
              "matched exact 12 effective GT trainables and round budget")
        # Save learned GT sidecars before eventual original restore.
        for condition,file in (
            ("GT_RECURRENT",RECURRENT_WEIGHTS),
            ("GT_NOCARRY_LOCAL",LOCAL_WEIGHTS),
        ):
            gate_sidecars[condition]=safe_gt_gate_sidecar(
                out/file,trained_gates[condition],condition)
        # Original baseline is q2i1 trained child with true S142 GT12 0.
        dev_results={}
        for condition in CONDITIONS:
            gg=(original_12_zero if condition=="GT_ZERO_BASELINE"
                else trained_gates[condition])
            set_gt_route(full,condition,gg)
            rows=[]
            for entry in dev:
                generated=own_token_dev(
                    model,sources,bridge,full,expansion,joint,
                    entry,condition=condition)
                rows.append(generated)
                dev_forwards+=generated["original_full_decoder_calls"]
            dev_results[condition]=rows
    finally:
        # Reset every modified S142 GT12 tensor and external optimizer flag;
        # the only learned params reside in NEW isolated research sidecars.
        with torch.no_grad():
            full.gates.copy_(original_12_zero)
            full.mix_logits.copy_(original_12_mix)
        for p in full.parameters():p.requires_grad_(False)
    check(all(torch.equal(v,full.state_dict()[k].detach().cpu())
              for k,v in original_full_gt.items()) and
          old_original_param_digest(model)==frozen_digest and
          all(torch.equal(v,joint.state_dict()[k].detach().cpu())
              for k,v in child_state.items()),
          "all original S142 GT12 tensors/core plus Q2I1 neural child restored")
    check(len(dev_results)==3 and all(len(x)==DEV_ROWS
                                    for x in dev_results.values()),
          "three complete 12DEV actual original native own-token versions")
    scored={}
    for condition,trace in dev_results.items():
        scored[condition]={
            "strict_full_answer_with_real_EOS":
                sum(x["strict_finished_full_answer_exact"] for x in trace),
            "text_exact_even_without_EOS":
                sum(x["text_exact_even_if_not_finished"] for x in trace),
            "EOS_finished_count":
                sum(x["native_EOS"] for x in trace),
            "total_real_original_decoder_forwards":
                sum(x["original_full_decoder_calls"] for x in trace),
            "answers_by_domain":{
                k:sum(x["strict_finished_full_answer_exact"]
                      for x in trace if x["domain"]==k)
                for k in KINDS,
            },
        }
    report={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "prior_q2i1_physical_original_fullanswer_EOS_SHA256":
            sha(q2i1_dir/Q2I1_REPORT),
        "new_RU_TRAIN_SHA256":source_meta["TRAIN_sha256"],
        "already_spent_DEV_SHA256":source_meta["DEV_sha256"],
        "original_GT12_provenance":gt_initial,
        "three_rounds_each_condition":ROUNDS,
        "matched_GT12_learnable_gates_PER_CANDIDATE":12,
        "GT12_original_pretrained_encoder_decoder_AND_CORE_GreenTunnel_unchanged":True,
        "new_q2i1_neural_output_head_and_NRA3_NPA1_weights_NOT_UPDATED":True,
        "original_original12_shared12_state_is_REAL_GT12_ACROSS_ALL_LAYERS":True,
        "recurrent_GT_carry_sigmoid_0p5_vs_LOCAL_nocarry_sigmoid_minus20":True,
        "original_native_vocab":VOCAB_SIZE,
        "train_q2i1_author_original_source_examples":TRAIN_ROWS,
        "already_scored_OLD_DEV_autoregressive_examples":DEV_ROWS,
        "GT12_original_native_TRAIN_decoder_forwards":gt_forwards,
        "GT12_original_native_OLD_DEV_own_token_decoder_forwards":dev_forwards,
        "total_original_native_decoder_forwards":gt_forwards+dev_forwards,
        "GT12_recurrent_train":reports["GT_RECURRENT"],
        "GT12_local_control_train":reports["GT_NOCARRY_LOCAL"],
        "GT12_physical_FP16_new_unaccepted_sidecars":gate_sidecars,
        "oldDEV_own_native_generated_data":dev_results,
        "oldDEV_own_native_EOS_scoring":scored,
        "no_new_independent_BLIND_or_opened_RESERVED_EVAL":True,
        "not_evidence_of_russian_quality_jump_without_new_sealed_test":True,
        "no_full_616_J7_EOS_replay_or_physical_packed_BPP_CPU_KV":True,
        "no_GOLD_S118_B_spent_EVAL128_or_Jessica_chat":True,
        "no_main_LIVE_Render_Supabase_APK_secrets":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"GT12_PHYSICAL_COUPLING_RESEARCH_ONLY_NO_PROMOTION",
    }
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I2_TRUE_GT12_43M_NRA3_NPA1_LINK_3ROUNDS "+
          json.dumps({
              "GT12_original_gates_exact_zero":True,
              "true_original_decoder_forwards":gt_forwards+dev_forwards,
              "GT_recurrent_first_round_CE":
                  reports["GT_RECURRENT"]["rounds"][0][
                      "TRAIN_mean_full_native_answer_BPE_CE_and_EOS"],
              "GT_recurrent_final_round_CE":
                  reports["GT_RECURRENT"]["rounds"][-1][
                      "TRAIN_mean_full_native_answer_BPE_CE_and_EOS"],
              "GT_local_final_round_CE":
                  reports["GT_NOCARRY_LOCAL"]["rounds"][-1][
                      "TRAIN_mean_full_native_answer_BPE_CE_and_EOS"],
              "OLD_DEV_strict_native_answers":{
                  key:value["strict_full_answer_with_real_EOS"]
                  for key,value in scored.items()},
              "OLD_DEV_visible_text_correct":{
                  key:value["text_exact_even_without_EOS"]
                  for key,value in scored.items()},
              "true_model_pretrained_original_unchanged":True,
              "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    x=p.parse_args()
    run(x.s93,x.side,x.s142,x.q2d_dir,x.q2f_dir,x.q2i1_dir,x.out)


if __name__=="__main__":
    main()
