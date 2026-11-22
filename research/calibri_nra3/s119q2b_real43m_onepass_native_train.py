"""S119q2b: physically train tiny EARLY-CAUSAL one-pass NRA3/NPA1 on native 43M.

Single small TRAIN-only optimization microblock, NOT general intelligence:
- exactly SIX newly authored synthetic TRAIN question->first-native-BPE targets
  (one for each of six types) generated and SHA-pinned by S119q2a;
- SIX disjoint same-template TRAIN-only source rows for *diagnostic* CE,
  NEVER read CALIBRATION answer labels in optimizer/evaluation;
- both structured-context and equal-parameter token-local ablation trained
  with same INIT, optimizer, full original native 49152 head, 2 epochs;
- original Core12+GreenTunnel and old NRA3 S115 53,751 FP16 frozen;
- source question prefix only: no completion token enters input;
- one original 12-layer decoder forward for EACH candidate train step;
- old full12-trained parent checkpoint now actually updated using early-only
  h0..h5 view (later pseudo-slots = copies of h5, NOT real h6..h11).

No heldout GOLD, spent EVAL128, S119n six prompts, S101/J7 replay, full
CPU KV benchmark, packing/bpp acceptance, LIVE, Render, Supabase or Jessica.
All checkpoint outputs are research-only gate0 unaccepted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    load_parent,prompt_features,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)
from research.calibri_nra3.s119q2a_train_source_manifest import (
    build_rows,validate_rows,build_manifest,canonical_bytes,EXPECTED_MANIFEST_SHA256,
    KINDS,STAGE as SOURCE_STAGE,EXPECTED_COUNTS,
)

STAGE="CALIBRI_S119Q2B_ORIGINAL43M_GT_ONEPASS_EARLY_NATIVE_SIX_TRAIN_MATCHED"
REPORT="S119Q2B_ONEPASS_43M_GT_NATIVE_6TRAIN_6PROBE_MATCHED_SOURCE_ONLY.json"
CONTEXT_CHECKPOINT="S119Q2B_EARLY_ONEPASS_CONTEXTUAL_TRAIN_ONLY_UNACCEPTED_FP16.safetensors"
LOCAL_CHECKPOINT="S119Q2B_EARLY_ONEPASS_LOCAL_MATCHED_UNACCEPTED_FP16.safetensors"
SEED=119131
EPOCHS=2
LR=0.001
TRAIN_INDEX=0
PROBE_INDEX=1


def prepare_microblock(rows):
    """Predeclared microblock only; never select by result or CAL outcomes."""
    audit=validate_rows(rows)
    check(audit["rows_total"]==90 and
          audit["split_counts"]==EXPECTED_COUNTS,
          "only immutable exact S119q2a original generator dataset")
    picked={}
    for kind in KINDS:
        series=[r for r in rows if r["kind"]==kind and r["split"]=="TRAIN"]
        check(len(series)==12,"only 12 canonical TRAIN rows for "+kind)
        for name,index in (("train",TRAIN_INDEX),("probe",PROBE_INDEX)):
            key=(name,kind)
            picked[key]=series[index]
            check(series[index]["id"].endswith("_%02d"%index),
                  "index pre-declared, not model-selected")
    train=[picked[("train",k)] for k in KINDS]
    probe=[picked[("probe",k)] for k in KINDS]
    check(len(train)==len(probe)==len(KINDS)==6 and
          set(x["prompt"] for x in train).isdisjoint(
          x["prompt"] for x in probe) and
          all(x["split"]=="TRAIN" for x in train+probe),
          "only source-split TRAIN with strictly disjoint probe prompts")
    check(all(x["source_kind"]=="original_deterministic_author_created_synthetic"
              and x["ground_truth_kind"]=="generator_symbolic_rule_NOT_independent_QA"
              for x in train+probe),
          "original authored synthetic labels are NOT independent answers")
    return train,probe


def verify_q2a_artifact(source_dir:Path):
    manifest_path=source_dir/"S119Q2A_MANIFEST.json"
    train_path=source_dir/"S119Q2A_TRAIN.jsonl"
    cal_path=source_dir/"S119Q2A_CALIBRATION.jsonl"
    check(source_dir.is_dir() and
          {x.name for x in source_dir.iterdir()} ==
          {train_path.name,cal_path.name,manifest_path.name},
          "exact physical SOURCE-only artifact files, no extra sealed GOLD")
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    check(manifest["stage"]==SOURCE_STAGE
          and manifest["synthetic_rows_SHA256"]==EXPECTED_MANIFEST_SHA256
          and manifest["metrics"]["no_training_or_gradients"] is True
          and manifest["candidate_accepted"] is False
          and manifest["output_gate"]==0,
          "S119q2a original canonical provenance and unaccepted source only")
    rows=build_rows()
    expected=build_manifest(rows)
    check(manifest==expected,
          "entire physical source manifest must equal frozen original generator")
    all_splits={
        "TRAIN":b"".join(canonical_bytes(row) for row in rows
                        if row["split"]=="TRAIN"),
        "CALIBRATION":b"".join(canonical_bytes(row) for row in rows
                        if row["split"]=="CALIBRATION"),
    }
    check(train_path.read_bytes()==all_splits["TRAIN"] and
          cal_path.read_bytes()==all_splits["CALIBRATION"],
          "physical SHA-bound original bytes only; don't access CAL labels")
    return rows,manifest


def true_native_first_target(tok,row):
    """Exact completion's first native BPE class, no handcrafted key-space."""
    check(row["split"]=="TRAIN" and
          row["ground_truth_kind"]==
          "generator_symbolic_rule_NOT_independent_QA",
          "S119q2b TRAIN-only authorship, never independently graded answer")
    completion=row["completion"]
    all_ids=list(tok.encode(completion,add_special_tokens=False))
    check(1<=len(all_ids)<40 and
          all(type(t) is int and 0<=t<tok.vocab_size for t in all_ids),
          "genuine source authored answer native BPE must be decodable")
    restored=tok.decode(all_ids,skip_special_tokens=True)
    check(restored.strip()==completion.strip(),
          "original native target completion roundtrip, no BPE ambiguity")
    return int(all_ids[0]),all_ids


def pack_native_rows(tok,rows):
    packed=[]
    for row in rows:
        prefix=strict_chat_source_ids(tok,row["prompt"])
        target,answer_ids=true_native_first_target(tok,row)
        check(target not in prefix[-1:] and
              len(prefix)<160 and
              all(0<=x<49152 for x in prefix),
              "answer first native BPE never injected into causal prompt")
        packed.append({
            "id":row["id"],"kind":row["kind"],"prefix":prefix,
            "features":prompt_features(tok,row["prompt"]),
            "target_first_native_bpe_id":target,
            "answer_native_token_count":len(answer_ids),
            "question_SHA256":hashlib.sha256(
                row["prompt"].encode("utf8")).hexdigest(),
        })
    return packed


def physical_native_loss(model,sources,bridge,full,expansion,adapter,
                         record,*,contextual):
    logits,witness=onepass_full_native_logits(
        model,sources,bridge,full,expansion,adapter,
        record["prefix"],record["features"],contextual=contextual)
    target=torch.tensor([record["target_first_native_bpe_id"]],
                        dtype=torch.long)
    check(logits.shape==(1,49152) and
          0<=int(target[0])<49152 and
          witness["candidate_forward_calls"]==1,
          "original native full 49152 classes after EXACT one decoder pass")
    return F.cross_entropy(logits,target),int(logits.detach().argmax(-1)),witness


def train_variant(model,sources,bridge,full,expansion,adapter,rows,
                  *,contextual):
    check(len(rows)==6 and all(r["id"].split("_")[-1]=="00" for r in rows),
          "only six PRE-DECLARED original source TRAIN targets")
    trainables=list(adapter.trainables())
    check(len({id(x) for x in trainables})==len(trainables) and
          all(x.requires_grad for x in trainables),
          "exact unique trainable model child params, old S115 frozen")
    optimizer=torch.optim.Adam(trainables,lr=LR)
    old_parent={
        k:v.detach().half().cpu().clone()
        for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()
    }
    trace=[]
    grad_roles=None
    for epoch in range(EPOCHS):
        for source in rows:
            optimizer.zero_grad(set_to_none=True)
            loss,pred,witness=physical_native_loss(
                model,sources,bridge,full,expansion,adapter,
                source,contextual=contextual)
            check(bool(torch.isfinite(loss)) and
                  math.isfinite(float(loss)),"finite true physical original 49k NLL")
            loss.backward()
            gate_grad=adapter.midlayer_gates.grad
            check(gate_grad is not None and
                  bool(torch.isfinite(gate_grad).all()) and
                  float(gate_grad.abs().sum())>1.e-12,
                  "native head CE must backprop via original layers6..11")
            if grad_roles is None and epoch==1:
                observed={
                    "trainable_core_rank4":adapter.shadow.unified.core_growth.up.grad,
                    "trainable_structured_interface":
                        adapter.shadow.unified.structured_interface.fuse.weight.grad,
                    "trainable_planning_NPA1":
                        adapter.shadow.unified.planning_action_area.state_update.weight_ih.grad,
                }
                grad_roles={}
                for key,grad in observed.items():
                    check(grad is not None and bool(torch.isfinite(grad).all())
                          and float(grad.abs().sum())>1.e-18,
                          "native CE reaches actual early unified "+key)
                    grad_roles[key]=float(torch.linalg.vector_norm(grad).item())
            torch.nn.utils.clip_grad_norm_(trainables,1.0)
            optimizer.step()
            check(all(bool(torch.isfinite(p).all()) for p in trainables),
                  "no NaN from tiny TRAIN-only early-state optimizer")
            trace.append({"epoch":epoch+1,"kind":source["kind"],
                          "question_SHA256":source["question_SHA256"],
                          "native_first_BPE_loss":float(loss.detach()),
                          "native_first_predicted_id_before_update":pred,
                          "native_first_target_id":
                              source["target_first_native_bpe_id"]})
    check(grad_roles is not None and len(trace)==EPOCHS*len(rows),
          "actual deep gradients to three new regions and exact 12 steps")
    check(all(torch.equal(
        v,adapter.shadow.unified.nra3_neural_weights.state_dict()[k].detach().half().cpu())
        for k,v in old_parent.items()),
        "original old 53751 FP16 neural weights unchanged after training")
    return trace,grad_roles,sum(p.numel() for p in trainables)


def probe_variant(model,sources,bridge,full,expansion,adapter,rows,*,contextual):
    check(len(rows)==6 and all(x["id"].split("_")[-1]=="01" for x in rows),
          "only six disjoint original same-template TRAIN-SOURCE probes")
    trace=[]
    for source in rows:
        loss,pred,wit=physical_native_loss(
            model,sources,bridge,full,expansion,adapter,
            source,contextual=contextual)
        trace.append({
            "kind":source["kind"],
            "native_first_BPE_loss":float(loss.detach()),
            "native_first_predicted_id":pred,
            "native_first_target_id":source["target_first_native_bpe_id"],
            "equals_symbolic_first_BPE":pred==source["target_first_native_bpe_id"],
        })
        del loss
    return {
        "source_probe_count":len(trace),
        "source_only_mean_native_first_BPE_CE":
            sum(x["native_first_BPE_loss"] for x in trace)/len(trace),
        "source_only_first_BPE_exact_matches":
            sum(x["equals_symbolic_first_BPE"] for x in trace),
        "not_blind_or_independent":True,
        "probes":trace,
    }


def persist_half(adapter,path):
    check(not path.exists(),"never overwrite research-only physical trained child")
    state={"model.nra3_calibri_joint.s119q2b."+k:
           v.detach().cpu().half().contiguous()
           for k,v in adapter.state_dict().items()}
    save_file(state,str(path))
    reread=load_file(str(path),device="cpu")
    check(set(reread)==set(state)
          and all(torch.equal(reread[k],v) for k,v in state.items()),
          "complete original and new trained FP16 physical child roundtrip")
    return {"sha256":sha(path),"file_bytes":path.stat().st_size,
            "saved_total_child_FP16_params":sum(x.numel() for x in state.values())}


def run(s93,side,s142,parent,parent_report,source_dir,out):
    check(not out.exists(),"new isolated q2b research outputs only")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    all_rows,manifest=verify_q2a_artifact(source_dir)
    train_raw,probe_raw=prepare_microblock(all_rows)
    _,ctx,old53751=load_parent(s93,side,s142,parent,parent_report)
    loc=copy.deepcopy(ctx)
    check(all(torch.equal(a,b) for a,b in zip(
        ctx.state_dict().values(),loc.state_dict().values())),
        "matched new Core12/NRA3/NPA1 architecture AND initial FP16 weights")
    ctx.eval();loc.eval()
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==49152,"real original full native BPE head only")
    tr=pack_native_rows(tok,train_raw)
    pr=pack_native_rows(tok,probe_raw)
    check(set(x["question_SHA256"] for x in tr).isdisjoint(
          x["question_SHA256"] for x in pr),
          "future probe prompts not in optimizer TRAIN")
    model.model.add_module("nra3_s119m_onpath",ctx)
    model.eval()
    digest=old_original_param_digest(model)
    check(all(not p.requires_grad for n,p in model.named_parameters()
              if "nra3_s119m_onpath" not in n) and
          ctx.output_gate==loc.output_gate==0 and
          not ctx.candidate_accepted and not loc.candidate_accepted,
          "entire original 43M+GT frozen and both children gate0")
    ctx_train_start=copy.deepcopy(ctx.state_dict())
    local_train_start=copy.deepcopy(loc.state_dict())
    check(all(torch.equal(ctx_train_start[k],local_train_start[k])
              for k in ctx_train_start),"exact initial physical same parameters")
    timer=time.perf_counter()
    trace_ctx,grad_ctx,ntrain_ctx=train_variant(
        model,sources,bridge,full,expansion,ctx,tr,contextual=True)
    ctx_train_seconds=time.perf_counter()-timer
    timer=time.perf_counter()
    trace_local,grad_local,ntrain_local=train_variant(
        model,sources,bridge,full,expansion,loc,tr,contextual=False)
    local_train_seconds=time.perf_counter()-timer
    check(ntrain_ctx==ntrain_local and
          ntrain_ctx>0 and
          old_original_param_digest(model)==digest,
          "exact matched original trainable budget and unchanged Core12/GT")
    ctx_probe=probe_variant(
        model,sources,bridge,full,expansion,ctx,pr,contextual=True)
    local_probe=probe_variant(
        model,sources,bridge,full,expansion,loc,pr,contextual=False)
    check(old_original_param_digest(model)==digest and
          ctx.output_gate==loc.output_gate==0 and
          not ctx.candidate_accepted and not loc.candidate_accepted,
          "all original pretrained weights exact, no acceptance")
    for oldkey,old in old53751.items():
        key=oldkey.split("model.nra3_calibri_joint.s119m_onpath.",1)[-1]
        check(torch.equal(ctx.state_dict()[key].detach().half().cpu(),old)
              and torch.equal(loc.state_dict()[key].detach().half().cpu(),old),
              "all original 53751 trained FP16 physically unchanged in A/B")
    out.mkdir(parents=True)
    ctx_info=persist_half(ctx,out/CONTEXT_CHECKPOINT)
    local_info=persist_half(loc,out/LOCAL_CHECKPOINT)
    report={
        "stage":STAGE,
        "parent_original_S93_sha256":sha(s93),
        "parent_original_S142_sha256":sha(s142),
        "parent_S119m_checkpoint_sha256":sha(parent),
        "q2a_original_authored_synthetic_rows_sha256":
            manifest["synthetic_rows_SHA256"],
        "q2a_full_physical_manifest_SHA256":
            sha(source_dir/"S119Q2A_MANIFEST.json"),
        "original_full_native_BPE_vocabulary":49152,
        "original_12_layer_onepass_earlycausal_h0_to_h5_only":True,
        "later_virtual_input_slots_are_h5_copies_NOT_real_later_states":True,
        "original_later_layers6_11_and_GreenTunnel_physically_executed":True,
        "one_real_original_decoder_forward_per_candidate_train_step":True,
        "six_source_train_example_SHA256":[x["question_SHA256"] for x in tr],
        "six_disjoint_SOURCE_TRAIN_only_probe_SHA256":
            [x["question_SHA256"] for x in pr],
        "q2a_CALIBRATION_labels_NOT_loaded_in_training_or_probe":True,
        "six_task_categories":list(KINDS),
        "real_original_49152_class_next_first_BPE_loss_ONLY":True,
        "native_long_answer_multi_token_not_trained":True,
        "two_epochs_same_optimizer_order_lr":True,
        "learning_rate":LR,
        "epochs":EPOCHS,
        "same_total_trainable_parameters":ntrain_ctx,
        "original_core_GT_unchanged":True,
        "old_original_53751_NRA3_FP16_unchanged":True,
        "contextual_train_trace":trace_ctx,
        "matched_local_train_trace":trace_local,
        "contextual_three_neural_gradient_norms":grad_ctx,
        "local_three_neural_gradient_norms":grad_local,
        "contextual_SOURCE_TRAIN_probe":ctx_probe,
        "matched_local_SOURCE_TRAIN_probe":local_probe,
        "contextual_MINUS_local_SOURCE_TRAIN_first_BPE_CE":
            ctx_probe["source_only_mean_native_first_BPE_CE"]-
            local_probe["source_only_mean_native_first_BPE_CE"],
        "contextual_win_ONLY_synthetic_source_probe":
            ctx_probe["source_only_mean_native_first_BPE_CE"]<
            local_probe["source_only_mean_native_first_BPE_CE"],
        "contextual_training_wall_seconds_NOT_CPU_throughput":
            ctx_train_seconds,
        "local_training_wall_seconds_NOT_CPU_throughput":
            local_train_seconds,
        "contextual_child_artifact":ctx_info,
        "local_child_artifact":local_info,
        "no_new_blind_heldout_or_independent_quality_score":True,
        "no_S101_full616_J7_EOS_replay":True,
        "no_final_runtime_packed_0_7_bpp_cert":True,
        "no_paired_CPU_KV_generation_speed_cert":True,
        "protected_S118_B_GOLD_and_spent_EVAL128_unopened":True,
        "no_Jessica_chat_LIVE_Render_Supabase_main_APK":True,
        "output_gate":0,"candidate_accepted":False,
        "decision":"SMALL_SYNTHETIC_TRAIN_ONLY_NATIVE_ONEPASS_MECHANICAL_LEARNING_NO_ACCEPTANCE",
    }
    (out/REPORT).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2B_REAL_43M_GT_EARLY_NATIVE_TRAIN "+json.dumps({
        "train_rows":len(tr),"probe_rows":len(pr),
        "train_steps_per_variant":len(trace_ctx),
        "equal_param":ntrain_ctx==ntrain_local,
        "contextual_train_end_CE":trace_ctx[-1]["native_first_BPE_loss"],
        "local_train_end_CE":trace_local[-1]["native_first_BPE_loss"],
        "contextual_probe_CE":
            ctx_probe["source_only_mean_native_first_BPE_CE"],
        "local_probe_CE":
            local_probe["source_only_mean_native_first_BPE_CE"],
        "source_only_win":report[
            "contextual_win_ONLY_synthetic_source_probe"],
        "all_old_weights_exact":True,
        "real_native_full_vocabulary":49152,
        "accepted":False,
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","parent","parent_report",
                "source_dir","out"):
        p.add_argument("--"+key.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.parent,a.parent_report,a.source_dir,a.out)


if __name__=="__main__":
    main()
