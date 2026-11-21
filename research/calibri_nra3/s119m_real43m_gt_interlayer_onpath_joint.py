"""S119m: real ON-PATH mid-decoder semantic/planning intervention, TRAIN only.

After S119l proved that an off-path full-vocabulary shadow did NOT change a
single generated token on three chat prompts, S119m tests a **real early
decoder intervention**: FIRST CPU prefill captures immutable original12
hidden on the SAME native BPE prefix; frozen physical S119k trained NRA3
(ALL original 53,751), NPA1 planner, rank4 core and causal structured
interface compute a 320D residual; SECOND original CPU prefill injects
this learned residual AFTER layer 5 and BEFORE original layers 6..11,
Green Tunnel and native 49,152-class LM head. Hooks remain confined to
isolated S119m research, with zero initialization ensuring the exact
original native logits. Both FULL original 43M/GT and original old NRA3
are frozen. Gradients from actual original FULL native LM vocabulary
loss propagate through actual remaining original transformer decoder
layers and Green Tunnel into new in-weight areas; never into old weights.

This is NOT efficient: each source prefix is processed TWICE, and
the first view uses all12 old original states; it is NOT a single-pass
pre-core neural area. This is an architecture and backpropagation feasibility
gate, not an accepted CPU-speed or universal-response model.

Paired contextual-versus-token-local control with matching parameter count,
identical source-only native token targets, original full vocabulary and
identical steps. Tiny TWO TRAIN-A rows and TWO same-TRAIN-A corpus probes
are insufficient for quality claims. No independent QA, no new J7 replay.
Do not touch S118 SynTagRus-B sealed GOLD, spent EVAL, LIVE, Jessica,
Render, Supabase, main, deployment or user data. Source license
SynTagRus-A CC BY-NC-SA 4.0 noncommercial research.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow, load_actual_original,
    real_cpu_prefix, sha, S142_SHA)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    parse_parent,
    STAGE as PARENT_STAGE,
    REPORT as PARENT_REPORT,
)
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    CHECKPOINT as PARENT_CHECKPOINT,
)
from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import (
    PARENT_H as SOURCE_FILE, PARENT_H_SHA as SOURCE_SHA,
    PARENT_REPORT as SOURCE_REPORT,
)
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

STAGE="CALIBRI_S119M_REAL_S142_43M_GT_INTERLAYER5_ONPATH_JOINT_TRAIN_ONLY"
CHECKPOINT="S119M_REAL_ONPATH_CORE_NRA3_NPA1_TRAINONLY_FP16_UNACCEPTED.safetensors"
REPORT="S119M_REAL_ONPATH_NATIVE_FULL_VOCAB_ABLATION_AND_FROZEN_43M_REPORT.json"
HOOK_LAYER=5
SEED=119126
TRAIN_ROWS=(0,1)
PROBE_ROWS=(128,129)
PREFIXES=(12,18)


def old_original_param_digest(model):
    """All originally trained CB59/GT/S142 model tensors, NOT new child."""
    hasher=hashlib.sha256()
    for name,p in model.named_parameters():
        if "nra3_s119m_onpath" in name:continue
        value=p.detach().cpu().contiguous()
        hasher.update(name.encode()+b"\0"+value.numpy().tobytes())
    return hasher.hexdigest()


class LearnedOnPathInterlayer(nn.Module):
    """Learned joint old-NRA3/NPA/core/RS residual injected after layer5.

    The 3 output gates are new learnable parameters in one S119k child.
    Zero initial gate preserves original final native logits; this child
    has NO independent vocab output, no separate LLM, no outside tool.
    """
    def __init__(self,shadow:NativeFullHeadShadow):
        super().__init__()
        self.shadow=shadow
        self.midlayer_gates=nn.Parameter(torch.zeros(3))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        for p in self.shadow.unified.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        # Old S119k output residual is off-path and cannot affect new loss.
        self.shadow.readout_gates.requires_grad_(False)
    def delta_components(self,original12,*,contextual:bool):
        check(self.output_gate==0 and not self.candidate_accepted
              and not self.activation_allowed and
              original12.ndim==4 and original12.shape[1]==12 and
              original12.shape[0]==1 and original12.shape[-1]==320
              and original12.shape[2]>=2,
              "ONPATH only on original frozen source prefix h320")
        valid=torch.ones((1,original12.shape[2]),dtype=torch.bool)
        graph,rel,plan,planned=self.shadow.unified(
            original12,valid,contextual=contextual)
        graph_parent=self.shadow.unified.nra3_neural_weights(
            original12,valid)
        adapted=self.shadow.unified.structured_interface(
            original12[:,0],graph_parent,valid,contextual=contextual)
        structure_delta=adapted[:,-1]-original12[:,0,-1]
        core_delta=self.shadow.unified.core_growth(
            original12)[:,-1,-1]-original12[:,-1,-1]
        rel6=F.pad(rel[:,-1],(0,314))
        check(structure_delta.shape==core_delta.shape==
              rel6.shape==planned.shape==(1,320),
              "all old full learned modules meet original native H320")
        return torch.stack(
            [core_delta,structure_delta+0.01*rel6,planned],dim=1)
    def forward(self,original12,*,contextual:bool):
        c=self.delta_components(original12,contextual=contextual)
        gates=self.midlayer_gates.tanh()
        delta=torch.einsum("bnh,n->bh",c,gates)
        check(delta.shape==(1,320) and bool(torch.isfinite(delta).all()),
              "one differentiable active learned residual for middle original layer")
        return delta
    def trainables(self):
        return [self.midlayer_gates,*
                self.shadow.unified.new_joint_trainables()]


@contextmanager
def research_only_midlayer_injection(model,adapter,delta,*,layer=HOOK_LAYER):
    """TRUE layer5's returned hidden is changed BEFORE original layer6.

    Installed AFTER legacy original GT3/9, FullGT12, S132 and S111
    hooks; no replacement of cache, normalization or final head.
    """
    check(layer==HOOK_LAYER and len(model.model.layers)==12
          and adapter.output_gate==0 and not adapter.candidate_accepted
          and delta.shape==(1,320) and delta.requires_grad,
          "research-only additive real original decoder onpath intervention")
    calls=[0]
    def hook(_mod,_args,output):
        calls[0]+=1
        check(calls[0]==1,
              "interlayer hook must run once per original forward")
        hidden=output[0] if isinstance(output,tuple) else output
        check(hidden.ndim==3 and hidden.shape[0]==1
              and hidden.shape[-1]==320 and
              bool(torch.isfinite(hidden).all()),
              "real original decoder layer5 tuple ABI changed")
        patch=torch.cat((torch.zeros_like(hidden[:,:-1,:]),
                         delta.unsqueeze(1)),dim=1)
        updated=hidden+patch
        return (updated,*output[1:]) if isinstance(output,tuple) else updated
    handle=model.model.layers[layer].register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()
        check(calls[0]==1,
              "middle original 12-layer path MUST actually execute")


def native_intervention_forward(model,sources,bridge,full,expansion,
                                adapter,record,tokenizer,*,contextual):
    """One TRUE ORIGINAL forward with grad flowing through layers6..11.

    The frozen source true prefix is used both for first-pass h12 and
    for next actual model.forward; target next BPE never input.
    """
    from torch import enable_grad
    prefix_ids=record["native_prefix_ids"].long()
    check(prefix_ids.shape==(1,record["prefix"]) and
          record["h12"].shape==(1,12,record["prefix"],320),
          "original physical same-prefix first and second passes only")
    txt=tokenizer.decode(prefix_ids[0].tolist(),skip_special_tokens=False)
    features=torch.as_tensor(s160.prompt_only_vector(tokenizer,txt),
                             dtype=torch.float32)
    check(features.shape==(512,) and bool(torch.isfinite(features).all()),
          "same original S95 features from current native prefix only")
    delta=adapter(record["h12"],contextual=contextual)
    # S115 frozen_cpu_phase_scope is INFERENCE-ONLY. For this deliberately
    # differentiable research forward, use the exact same S93 bridge
    # prompt_scope directly; S126-Q+S132 prefill hooks stay installed.
    with enable_grad(),expansion.prefill_only(model),\
         s160.prefill_only_tunnel(model,full),\
         bridge.prompt_scope("S93",features),\
         research_only_midlayer_injection(model,adapter,delta):
        clear(sources)
        out=model(input_ids=prefix_ids,past_key_values=None,
                  use_cache=True,output_hidden_states=False)
        check(out.past_key_values is not None
              and out.logits.ndim==3
              and out.logits.shape[:2]==prefix_ids.shape
              and out.logits.shape[-1]>1000,
              "must evaluate original full native LM head after REAL later layers")
        logits=out.logits[:,-1].float()
        clear(sources)
    check(logits.requires_grad and bool(torch.isfinite(logits).all()),
          "gradient must pass from original head to intervention")
    return logits


def prepare_source(model,sources,bridge,full,expansion,tokenizer,
                   ids,mask,row,prefix):
    """First full ORIGINAL CPU S142 prefix is immutable source context."""
    old=real_cpu_prefix(
        model,sources,bridge,full,expansion,tokenizer,ids,mask,row,prefix)
    old["native_prefix_ids"]=ids[row,:prefix].long().unsqueeze(0).clone()
    check(old["row"]==row and old["prefix"]==prefix and
          not bool((old["native_prefix_ids"]==
                    int(old["next_id"])).all()),
          "actual native next ID never provided as model input")
    return old


def full_native_CE(model,sources,bridge,full,expansion,tokenizer,
                   adapter,record,*,contextual):
    logits=native_intervention_forward(
        model,sources,bridge,full,expansion,adapter,
        record,tokenizer,contextual=contextual)
    target=torch.tensor([record["next_id"]],dtype=torch.long)
    check(0<=int(target[0])<logits.shape[-1],
          "full original native tokenizer index as one teacher-forced target")
    return F.cross_entropy(logits,target),logits


def train_variant(model,sources,bridge,full,expansion,tokenizer,
                  adapter,training,*,contextual):
    optimizer=torch.optim.Adam(adapter.trainables(),lr=0.00075)
    old={k:v.detach().half().clone()
         for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()}
    trace=[]
    for epoch in range(2):
        for index,item in enumerate(training):
            optimizer.zero_grad(set_to_none=True)
            loss,logits=full_native_CE(
                model,sources,bridge,full,expansion,tokenizer,
                adapter,item,contextual=contextual)
            check(bool(torch.isfinite(loss)),
                  "true full native loss via REAL original layer6..11")
            loss.backward()
            gate_grad=adapter.midlayer_gates.grad
            check(gate_grad is not None and
                  bool(torch.isfinite(gate_grad).all()) and
                  float(gate_grad.abs().sum())>1.e-12,
                  "REAL decoder layers6..11 must train interlayer gate")
            if epoch==1 and index==0:
                checks={
                    "learned_core_rank4":
                       adapter.shadow.unified.core_growth.up.grad,
                    "semantic_NRA3_graph":
                       adapter.shadow.unified.reasoning_role_gate.grad,
                    "structured_context":
                       adapter.shadow.unified.structured_interface.fuse.weight.grad,
                    "NPA1_planning":
                       adapter.shadow.unified.planning_action_area.pred_next_event.weight.grad,
                }
                for key,grad in checks.items():
                    check(grad is not None and bool(torch.isfinite(grad).all())
                          and float(grad.abs().sum())>1.e-15,
                          "real deep core shared gradient must reach "+key)
            torch.nn.utils.clip_grad_norm_(adapter.trainables(),1.)
            optimizer.step()
            trace.append({"epoch":epoch+1,"row":item["row"],
                "prefix":item["prefix"],"full_native_lm_CE":float(loss.detach()),
                "interlayer_gate_tanh":adapter.midlayer_gates.detach().tanh().tolist()})
    check(all(torch.equal(old[k],v.detach().half())
              for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()),
          "all 53751 prior old learned neural FP16 params immutable")
    return trace


def probe(model,sources,bridge,full,expansion,tokenizer,
          adapter,probe_rows,*,contextual):
    scores=[]
    with torch.no_grad():
        for item in probe_rows:
            loss,logits=full_native_CE(
                model,sources,bridge,full,expansion,tokenizer,
                adapter,item,contextual=contextual)
            original=item["base_logits"]
            orig_loss=F.cross_entropy(
                original,torch.tensor([item["next_id"]]))
            scores.append({
                "row":item["row"],"prefix":item["prefix"],
                "original_true_full_vocab_CE":float(orig_loss),
                "onpath_real_deep_decoder_native_CE":float(loss),
                "greedy_native_id_original":int(original.argmax(-1)),
                "greedy_native_id_true_onpath":int(logits.argmax(-1)),
                "gold_native_BPE_id_from_TRAIN_A":item["next_id"],
            })
    return {
        "count":len(scores),
        "mean_original_true_native_CE":sum(z["original_true_full_vocab_CE"]
                                           for z in scores)/len(scores),
        "mean_real_onpath_native_CE":sum(
            z["onpath_real_deep_decoder_native_CE"] for z in scores)/len(scores),
        "greedy_changed_native_output_count":sum(
            z["greedy_native_id_original"]!=z["greedy_native_id_true_onpath"]
            for z in scores),
        "source_ONLY_probe":scores,
    }


def run(s93,s93_side,s142,source,source_report,parent,parent_report,out):
    check(not out.exists() and source.name==SOURCE_FILE and
          source_report.name==SOURCE_REPORT and
          sha(source)==SOURCE_SHA,
          "only physical original12 TRAIN-A first pass source")
    witness=json.loads(source_report.read_text())
    check(witness["ALL12_CPU_SAFETENSORS_SHA256"]==SOURCE_SHA and
          witness["source_ONLY_no_S113_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU"]
          and witness["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"],
          "original source no GOLD plus historic 616 ancestry only")
    torch.manual_seed(SEED);torch.set_num_threads(2)
    inherited,old,loaded=parse_parent(
        s93,s93_side,s142,parent,parent_report)
    check(inherited["stage"]==PARENT_STAGE or
          inherited["stage"].startswith("S119K_"),
          "only S119k original 43M GT full-native child lineage")
    physical=load_file(str(source),device="cpu")
    valid=physical["native_BPE_source_valid_mask"]
    ids=physical["original_native_BPE_token_ids"]
    check(valid.shape==ids.shape and valid.shape[0]==256
          and valid.dtype==torch.bool and ids.dtype==torch.int32,
          "S114 original 256 native BPE source only")
    original,sources,bridge,full,expansion=load_actual_original(
        s93,s93_side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    digest=old_original_param_digest(original)
    onpath=LearnedOnPathInterlayer(loaded)
    local=copy.deepcopy(onpath)
    check(sum(p.numel() for p in onpath.parameters())==
          sum(p.numel() for p in local.parameters()) and
          all(torch.equal(a,b) for a,b in zip(
              onpath.state_dict().values(),local.state_dict().values())),
          "strict matched original pretrained architecture and initial values")
    original.model.add_module("nra3_s119m_onpath",onpath)
    original.eval()
    check(old_original_param_digest(original)==digest
          and all(not p.requires_grad for n,p in original.named_parameters()
                  if "nra3_s119m_onpath" not in n),
          "must not train any original 43M+GT/S142 weights")
    tr=[prepare_source(
            original,sources,bridge,full,expansion,
            tokenizer,ids,valid,row,prefix)
        for row,prefix in zip(TRAIN_ROWS,PREFIXES)]
    pr=[prepare_source(
            original,sources,bridge,full,expansion,
            tokenizer,ids,valid,row,prefix)
        for row,prefix in zip(PROBE_ROWS,PREFIXES)]
    first=probe(original,sources,bridge,full,expansion,tokenizer,
                onpath,pr,contextual=True)
    first_local=probe(original,sources,bridge,full,expansion,tokenizer,
                local,pr,contextual=False)
    check(first==first_local and
          abs(first["mean_original_true_native_CE"]-
              first["mean_real_onpath_native_CE"])<1.e-5,
          "ZERO on-path midlayer gate must preserve exact ORIGINAL decoder logits")
    timer=time.perf_counter()
    trace_context=train_variant(
        original,sources,bridge,full,expansion,tokenizer,
        onpath,tr,contextual=True)
    context_cpu_seconds=time.perf_counter()-timer
    timer=time.perf_counter()
    trace_local=train_variant(
        original,sources,bridge,full,expansion,tokenizer,
        local,tr,contextual=False)
    local_cpu_seconds=time.perf_counter()-timer
    result_context=probe(
        original,sources,bridge,full,expansion,tokenizer,
        onpath,pr,contextual=True)
    result_local=probe(
        original,sources,bridge,full,expansion,tokenizer,
        local,pr,contextual=False)
    check(old_original_param_digest(original)==digest
          and onpath.output_gate==0 and
          not onpath.candidate_accepted and
          all(torch.equal(
              onpath.shadow.state_dict()[k[len("model.nra3_calibri_joint.s119k_shadow."):]].detach().half(),v)
              for k,v in old.items() if ".nra3_neural_weights." in k),
          "exact all original 43M + 53751 learned NRA3/GT untouched")
    # Crucial: physical new child is stored as a complete learned tensor
    # alongside old 53,751. It must not be inserted in LIVE original model.
    tensors={"model.nra3_calibri_joint.s119m_onpath."+k:
             v.detach().half().cpu().contiguous()
             for k,v in onpath.state_dict().items()}
    out.mkdir(parents=True)
    checkpoint=out/CHECKPOINT
    save_file(tensors,str(checkpoint))
    reread=load_file(str(checkpoint),device="cpu")
    check(set(reread)==set(tensors)
          and all(torch.equal(reread[k],v) for k,v in tensors.items()),
          "physical research onpath learned FP16 child must roundtrip")
    report={
        "stage":STAGE,
        "original_43M_cb61_s93_sha256":sha(s93),
        "original_s142_43m_gt_sha256":sha(s142),
        "original_s114_real_native_source_sha256":SOURCE_SHA,
        "inherited_complete_s119k_shadow_sha256":sha(parent),
        "original_12layer_onpath_intervention_after_layer":HOOK_LAYER,
        "original_later_layers_6_through_11_execute_after_neural_intervention":True,
        "two_original_43m_full_prefix_passes_per_native_sample":True,
        "one_original_prefill_precomputed_full12_no_future_gold_input":True,
        "original_full_49152_native_token_output_used":True,
        "one_SHARED_loss_grad_reaches_original_later_layers_and_new_NRA3_NPA_core_RS":True,
        "original_pretrained_43M_and_GT_all_weights_sha_unchanged":True,
        "original_pretrained_S115_NRA3_53751_FP16_unchanged":True,
        "same_train_source_rows":list(zip(TRAIN_ROWS,PREFIXES)),
        "same_probe_SOURCE_TRAIN_A_not_independent_rows":
            list(zip(PROBE_ROWS,PREFIXES)),
        "untrained_zero_gate_exact_original_native_full_vocab":first,
        "contextual_two_epoch_full_native_train":trace_context,
        "local_two_epoch_full_native_train":trace_local,
        "contextual_real_onpath_same_TRAIN_A_probe":result_context,
        "matched_param_local_real_onpath_same_TRAIN_A_probe":result_local,
        "contextual_MINUS_local_native_full_vocab_CE":
            result_context["mean_real_onpath_native_CE"]-
            result_local["mean_real_onpath_native_CE"],
        "contextual_wins_two_train_corpus_positions_ONLY":
            result_context["mean_real_onpath_native_CE"]<
            result_local["mean_real_onpath_native_CE"],
        "contextual_research_train_CPU_seconds_NOT_throughput":context_cpu_seconds,
        "local_research_train_CPU_seconds_NOT_throughput":local_cpu_seconds,
        "physical_total_one_child_FP16_params":sum(x.numel() for x in tensors.values()),
        "physical_total_one_child_bytes":checkpoint.stat().st_size,
        "physical_total_one_child_sha256":sha(checkpoint),
        "researched_only_two_pass_deep_intervention_CPU_cost_NOT_MEASURED":True,
        "no_real_unconstrained_ANSWER_generation_post_training":True,
        "no_independent_holdout_user_questions":True,
        "not_normal_S101_J7_controller":True,
        "no_candidate_full616_J7_EOS_replay":True,
        "no_loader_all_in_bpp_or_paired_CPU_throughput":True,
        "not_onepass_preembedding_representation_area":True,
        "not_all_original_layers_joint_trainable":True,
        "protected_S118_B_GOLD_unopened":True,
        "license_CC_BY_NC_SA4_noncommercial":True,
        "output_gate":0,"candidate_accepted":False,
        "activation_allowed":False,
        "decision":"ACTUAL_ORIGINAL43M_GT_MID_LAYER_TWO_PASS_ONPATH_JOINT_NEURAL_GRAD_SOURCE_ONLY_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119M_REAL_43M_GT_ONPATH_INTERLAYER_NATIVE_FULL_VOCAB "+
          json.dumps({
            "after_true_original_decoder_layer":HOOK_LAYER,
            "genuine_complete_vocab":49152,
            "original_native_CE":first["mean_original_true_native_CE"],
            "contextual_onpath_CE":result_context["mean_real_onpath_native_CE"],
            "matched_local_onpath_CE":result_local["mean_real_onpath_native_CE"],
            "contextual_win_ONLY_SOURCE_TRAIN_A":report[
                "contextual_wins_two_train_corpus_positions_ONLY"],
            "trained_deep_gradient":True,
            "old_53751_FP16_exact":True,
            "original_43M_GT_exact":True,
            "candidate_accepted":False,
          },sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ("s93","s93_side","s142","source","source_report",
                 "parent","parent_report","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    run(a.s93,a.s93_side,a.s142,a.source,a.source_report,
        a.parent,a.parent_report,a.out)


if __name__=="__main__":main()
