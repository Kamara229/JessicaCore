"""CALIBRI S47: tiny ACTIVE hidden-state Green Tunnel, physical budget audited.

Real physical native-paid 36M S45 CB59 source is immutable. Learn an
architecture-adaptable, low-rank (320->8->320) causal correction to the
final hidden state BEFORE the tied decoder head predicts the next token.
It participates in the actual autoregressive computation, not an output
text postprocessor, hardcoded answer table, gold-token injection, external
FP32 teacher, or teacher-distribution imitation.

The controller is universally defined by the host hidden width and tied
head interface; ONLY the tiny 36M host is tested. Here the causal gate
turns on after the FIRST generated token for ALL questions; the first
token therefore stays bit-for-bit identical, including 34 previously
correct color responses. Train the controller on S30 64 TRAIN continuations
and seven old historical TRAIN continuations (never heldout QA). Independent
user-specified FP32 parity and transfer to another architecture are unproven.

Unlike experimental unpaid 10MB controllers, the ENTIRE physically stored
S47 controller FP32 file and manifest must FIT alongside the exact CB59
source in the same all-inclusive 0.700 bpp budget. Host vocab is shared,
not an extra exported tokenizer. Never touch Jessica, LIVE, Render,
Supabase, production, S29 sealed QA, inspected S26 QA, or GPU services.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v186.pairs import build_pairs
from v174.run import save
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS,SOURCE_BYTES,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s37 import paid_cb59_margin_protected_recovery as s37
from green_tunnel.contract import PhysicalSource,verify_physical_source

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
STEPS=200
RANK=8
SEED=4747
BATCH=16
PROBES=(40,80,120,160,200)
MAX_NEW=10
LR=.023
assert STEPS==PROBES[-1]

class ActiveGreenTunnel(nn.Module):
    def __init__(self,width,rank):
        super().__init__()
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        self.gate=nn.Linear(width,1,bias=True)
        with torch.no_grad():
            nn.init.normal_(self.down.weight,mean=0,std=.028)
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.gate.bias.fill_(.0)
    def forward(self,hidden):
        # No answer IDs or gold table: per-context neural hidden correction.
        v=self.up(torch.tanh(self.down(hidden)))
        return v*torch.sigmoid(self.gate(hidden))

def parse_training_cache(model,sources,tok,new,old):
    fresh={"noun":[],"eos":[]}
    held=[]
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            with torch.inference_mode():
                clear(sources)
                output=model(input_ids=row["ids"][None],use_cache=False,output_hidden_states=True)
                hidden=output.hidden_states[-1][0]
                log=output.logits[0]
                for position,kind in ((1,"noun"),(2,"eos")):
                    p=int(row["positions"][position])
                    fresh[kind].append((
                      hidden[p].detach().clone(),log[p].detach().clone(),int(row["targets"][position])))
                del output
                clear(sources)
    old_before=s37.old_guard(model,sources,tok,old)
    old_protected=s37.preserved_ids(old_before)
    if len(old_protected)!=7:
        raise AssertionError("S47 requires all seven prior correct old TRAIN identities")
    for pair in old[:4]:
        for side in ("left","right"):
            row=pair[side]
            ident=f'{pair["pair_id"]}_{side}'
            if ident not in old_protected:continue
            actual=old_before[ident]["actual"]
            if len(actual)<2:continue
            prefix=row["prompt_ids"]
            full=torch.cat((prefix,torch.tensor(actual[:-1],dtype=torch.long)))
            with torch.inference_mode():
                clear(sources)
                output=model(input_ids=full[None],use_cache=False,output_hidden_states=True)
                for step in range(1,len(actual)):
                    p=len(prefix)+step-1
                    held.append((output.hidden_states[-1][0,p].detach().clone(),
                      output.logits[0,p].detach().clone(),int(actual[step])))
                del output
                clear(sources)
    # Inference-mode tensors may not be saved in autograd for adapter
    # gradients: materialize ordinary detached CPU tensors outside mode.
    def ordinary(t):
        return torch.tensor(t.detach().cpu().numpy().copy(),dtype=torch.float32)
    result={}
    for kind,rows in (("noun",fresh["noun"]),("eos",fresh["eos"]),("old",held)):
        if not rows and kind!="old":raise AssertionError("missing train tokens "+kind)
        result[kind]={
            "hidden":torch.stack([ordinary(r[0]) for r in rows]) if rows else None,
            "base":torch.stack([ordinary(r[1]) for r in rows]) if rows else None,
            "target":torch.tensor([r[2] for r in rows],dtype=torch.long),
            "count":len(rows)}
    if result["noun"]["count"]!=64 or result["eos"]["count"]!=64 or result["old"]["count"]<7:
        raise AssertionError("S47 TRAIN new/old labels not eligible")
    return result,old_protected

def apply_delta(base,hidden,gate,head):
    return base+F.linear(gate(hidden),head)

@torch.inference_mode()
def greedy(model,sources,gate,head,tok,question):
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
    ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None]
    actual=[]
    for step in range(MAX_NEW):
        clear(sources)
        out=model(input_ids=ids,use_cache=False,output_hidden_states=gate is not None)
        logits=out.logits[0,-1]
        if gate is not None and step>=1:
            hidden=out.hidden_states[-1][0,-1]
            logits=apply_delta(logits,hidden,gate,head)
        winner=int(logits.argmax())
        del out
        actual.append(winner)
        clear(sources)
        if winner==tok.eos_token_id:break
        ids=torch.cat((ids,torch.tensor([[winner]],dtype=torch.long)),dim=1)
    return actual

@torch.inference_mode()
def probe(model,sources,gate,head,tok,pairs):
    rows={}
    for pair in pairs:
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            actual=greedy(model,sources,gate,head,tok,row["question"])
            expected=tok.encode(row["answer"],add_special_tokens=False)+[tok.eos_token_id]
            rows[key]={"text":tok.decode(actual,skip_special_tokens=True),
              "actual":actual,"expected":expected,
              "full_correct":actual==expected,
              "first_gold_correct":bool(actual) and actual[0]==expected[0]}
    return rows

def count(rows,k):
    return sum(bool(r[k]) for r in rows.values())

def clone_state(module):
    return {k:v.detach().clone() for k,v in module.state_dict().items()}

def batch_loss(tunnel,cache,head,rng):
    terms=[]
    for kind,coeff,count_items in (("noun",1.,BATCH),("eos",.35,BATCH),("old",.8,8)):
        batch=cache[kind]
        indices=torch.tensor([rng.randrange(batch["count"]) for _ in range(count_items)])
        hidden=batch["hidden"].index_select(0,indices)
        old_logit=batch["base"].index_select(0,indices)
        expected=batch["target"].index_select(0,indices)
        score=apply_delta(old_logit,hidden,tunnel,head)
        terms.append(coeff*F.cross_entropy(score,expected))
    # Prevent unconstrained massive changes unrelated to questions.
    return sum(terms),[float(x.detach()) for x in terms]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=a.source.read_bytes()
    source=PhysicalSource(codec="CB59_K32V8",original_unique_parameters=SOURCE_PARAMS,
           physical_whole_bytes=SOURCE_BYTES,physical_sha256=S45_SHA)
    original=verify_physical_source(source,raw)
    if len(raw)!=3141513 or original["actual_complete_bpp"]>=.7:
        raise AssertionError("S47 physical source exceeds exact immutable 0.7bpp")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new,old=s30.make_training_pairs(tok),build_pairs(tok,"TRAIN")
    if len(new)!=32 or len(old)!=24:raise AssertionError("TRAIN set drift")
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    # The small-first bootstrap normally exposes paid trainable centroids;
    # S47 freezes EVERY core parameter before optimizing ONLY the tunnel.
    model.requires_grad_(False)
    if sum(p.requires_grad for p in model.parameters()):
        raise AssertionError("S47 may not modify CB59 source weights")
    cache,old_ids=parse_training_cache(model,sources,tok,new,old)
    head=model.lm_head.source().detach().contiguous()
    clear(sources)
    width=int(model.config.hidden_size)
    if width!=320 or head.shape[1]!=width:
        raise AssertionError("Host physical head shape inconsistent")
    tunnel=ActiveGreenTunnel(width,RANK)
    # Only small controller trainable (no original FP32 weights).
    tunnel_nparams=sum(p.numel() for p in tunnel.parameters())
    if tunnel_nparams!=2*width*RANK+width+1:
        raise AssertionError("Unexpected Green Tunnel parameter growth")
    if (SOURCE_BYTES+4*tunnel_nparams+4096)*8/SOURCE_PARAMS>.7:
        raise AssertionError("Green Tunnel cannot fit complete bpp with budget overhead")

    new_base=probe(model,sources,None,head,tok,new)
    old_base=probe(model,sources,None,head,tok,old[:4])
    old_correct={k for k,v in old_base.items() if v["full_correct"]}
    if old_ids!=old_correct or len(old_ids)!=7:
        raise AssertionError("S45 baseline historical seven moved")
    base_color={k for k,v in new_base.items() if v["first_gold_correct"]}
    if len(base_color)!=34 or count(new_base,"full_correct")!=0:
        raise AssertionError("S45 physical 34 first-color and 0 full baseline not reproduced")

    opt=torch.optim.AdamW(tunnel.parameters(),lr=LR,weight_decay=.0004)
    rng=random.Random(SEED)
    trials=[]
    best_correct=-1
    best_state=None
    best_rows=None
    best_old=None
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        loss,parts=batch_loss(tunnel,cache,head,rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tunnel.parameters(),max_norm=10)
        opt.step()
        if step not in PROBES:continue
        gate=tunnel.eval()
        old_now=probe(model,sources,gate,head,tok,old[:4])
        lost_old=sorted(old_ids-{k for k,v in old_now.items() if v["full_correct"]})
        new_now=None; full_correct=None; unchanged_color=None
        if not lost_old:
            new_now=probe(model,sources,gate,head,tok,new)
            unchanged_color=(base_color <= {k for k,v in new_now.items() if v["first_gold_correct"]})
            full_correct=count(new_now,"full_correct")
            if not unchanged_color:
                raise AssertionError("Position-zero gate illegally changed color before activation")
        select=not lost_old and unchanged_color and full_correct is not None and full_correct>best_correct
        if select:
            best_correct=full_correct
            best_state=clone_state(gate)
            best_rows=new_now
            best_old=old_now
        event={"step":step,"train_objective":float(loss.detach()),"loss_components":parts,
               "lost_original_old_full_ids":lost_old,
               "original_34_first_color_signatures_unchanged":unchanged_color,
               "full_64_new_correct":full_correct,
               "accepted_checkpoint_for_train_only_comparison":select}
        trials.append(event)
        print(json.dumps({"event":"S47_ACTIVE_TUNNEL_PROGRESS",**event},ensure_ascii=False),flush=True)
        tunnel.train()

    before={"new_full":count(new_base,"full_correct"),
       "new_correct_first_color":count(new_base,"first_gold_correct"),
       "old_full":count(old_base,"full_correct")}
    accepted=best_state is not None and best_correct>0
    result_json=a.output/"S47_ACTIVE_GREEN_TUNNEL_TRAIN_ONLY.json"
    output=a.output/"S47_ACTIVE_GREEN_TUNNEL_SIDECAR.safetensors"
    manifest_path=a.output/"S47_PHYSICAL_ALL_INCLUSIVE_MANIFEST.json"
    manifest=None
    reload_verification=None
    after=None
    speed=None
    if accepted:
        tunnel.load_state_dict(best_state)
        tunnel.eval()
        if count(best_old,"full_correct")!=7:
            raise AssertionError("Historical seven changed in selected TRAIN-only candidate")
        if not base_color <= {k for k,v in best_rows.items() if v["first_gold_correct"]}:
            raise AssertionError("S47 candidate lost a first-color correct response")
        save_file({k:v.contiguous() for k,v in best_state.items()},str(output))
        extras=output.stat().st_size
        if extras>10_000_000:
            raise AssertionError("S47 Green Tunnel exceeded 10MB independent controller")
        reconstructed=ActiveGreenTunnel(width,RANK)
        reconstructed.load_state_dict(load_file(str(output)))
        reconstructed.eval()
        re_new=probe(model,sources,reconstructed,head,tok,new)
        re_old=probe(model,sources,reconstructed,head,tok,old[:4])
        if re_new!=best_rows or re_old!=best_old:
            raise AssertionError("S47 actual complete greedy responses changed after physical sidecar reload")
        reload_verification={"all_64_new_full_sequences_identical":True,
             "all_8_old_full_sequences_identical":True,"new_full_correct":best_correct,
             "old_full_correct":7}
        # Manifest *including its own exact real stored bytes* is charged.
        manifest={"stage":"S47_ACTIVE_GREEN_TUNNEL_PHYSICAL_BUDGET",
           "core_CB59_sha256":S45_SHA,"core_CB59_bytes":SOURCE_BYTES,
           "base_complete_bpp":original["actual_complete_bpp"],
           "sidecar_file":output.name,"sidecar_bytes":extras,
           "sidecar_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
           "sidecar_trainable_parameter_count":tunnel_nparams,
           "all_inclusive_budget_charges_manifest_and_sidecar":True,
           "reuses_pinned_host_tokenizer_no_additional_tokenizer_export":True,
           "model_frozen":True,"source_indices_and_centroids_frozen":True,
           "runtime_needs_CB59_and_sidecar":True}
        # Manifest is CHARGED too: the physical source+controller+final
        # serialized manifest, never a hidden omitted metadata sidecar.
        # Compute the self-referential byte count to an exact fixed point.
        byte_cap=(SOURCE_PARAMS*7)//80
        converged=False
        for _ in range(40):
            payload=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8")
            actual_manifest_bytes=len(payload)
            actual_total=SOURCE_BYTES+extras+actual_manifest_bytes
            actual_bpp=8*actual_total/SOURCE_PARAMS
            new_values={
                "manifest_bytes":actual_manifest_bytes,
                "all_inclusive_total_bytes":actual_total,
                "all_inclusive_actual_bpp":actual_bpp}
            if all(manifest.get(k)==v for k,v in new_values.items()):
                converged=True
                break
            manifest.update(new_values)
        payload=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8")
        actual_total=SOURCE_BYTES+extras+len(payload)
        print(json.dumps({"event":"S47_PHYSICAL_BUDGET_CHECK",
          "source_bytes":SOURCE_BYTES,"sidecar_bytes":extras,
          "manifest_bytes":len(payload),"physical_total_bytes":actual_total,
          "physical_byte_cap_0p7":byte_cap,
          "physical_bpp":8*actual_total/SOURCE_PARAMS,
          "self_consistent":converged}),flush=True)
        if not converged:
            raise AssertionError("Manifest byte accounting did not converge")
        if actual_total>byte_cap:
            raise AssertionError("Actual source+controller+manifest exceeds 0.7 bpp")
        if (manifest["all_inclusive_total_bytes"]!=actual_total or
            manifest["manifest_bytes"]!=len(payload) or
            manifest["all_inclusive_actual_bpp"]!=8*actual_total/SOURCE_PARAMS):
            raise AssertionError("Manifest physical bytes self-accounting mismatch")
        manifest_path.write_bytes(payload)
        after={"new_full":count(re_new,"full_correct"),
            "new_correct_first_color":count(re_new,"first_gold_correct"),
            "old_full":count(re_old,"full_correct"),
            "all_64_new":re_new,"all_8_old":re_old}
        # Paired wall clock on real end-to-end greedy generation, not a kernel
        # estimate. Results remain noisy and cannot alone certify CPU parity.
        benchmark=[new[i][side] for i,side in ((1,"left"),(4,"right"),(7,"left"),(10,"right"))]
        times={}
        for label,gate in (("core",None),("tunnel",reconstructed)):
            periods=[]
            for row in benchmark:
                start=time.perf_counter()
                greedy(model,sources,gate,head,tok,row["question"])
                periods.append(time.perf_counter()-start)
            times[label]=periods
        speed={"paired_4_TRAIN_question_CPU_greedy_seconds":times,
          "median_core":statistics.median(times["core"]),
          "median_green":statistics.median(times["tunnel"]),
          "speed_ratio_core_over_green":statistics.median(times["core"])/statistics.median(times["tunnel"]),
          ">=90pct_requirement_sample_only":statistics.median(times["core"])/statistics.median(times["tunnel"])>=.9,
          "not_independent_final_speed_certification":True}
    report={
      "stage":"S47_REAL_ACTIVE_TINY_LOW_RANK_HIDDEN_GREEN_TUNNEL",
      "source_CB59_sha256":S45_SHA,
      "source_paid_bytes":SOURCE_BYTES,
      "unmodified_core_bpp":original["actual_complete_bpp"],
      "all_inclusive_bpp_including_tunnel":manifest["all_inclusive_actual_bpp"] if manifest else None,
      "controller_hidden_width":width,
      "controller_rank":RANK,"controller_params":tunnel_nparams,
      "precommitted_training_steps":STEPS,"learning_rate":LR,
      "TRAIN_64_balanced_new_only":True,"TRAIN_old_7_rehearsed":True,
      "precommitted_checkpoint_steps":list(PROBES),
      "no_F32_teacher_no_dense_export":True,"no_host_model_weight_updates":True,
      "does_not_hardcode_answer_token_lookup":True,
      "no_output_text_postprocessor":True,
      "causal_last_hidden_intervention_before_tied_head":True,
      "no_QA_VALID_TEST_or_previous_S26_inspected_QA":True,
      "Jessica_chat_test_NOT_RUN":True,
      "LIVE_Render_Supabase_APK_unmodified":True,
      "unseen_generalization_and_FP32_parity_NOT_ESTABLISHED":True,
      "model_family_transfer_NOT_ESTABLISHED":True,
      "baseline":before,
      "checkpoint_probes":trials,
      "accepted_actual_new_full_improvement":accepted,
      "selected_train_full_answers":best_correct if best_state else None,
      "all_64_before":new_base,
      "all_8_old_before":old_base,
      "after":after,
      "physical_manifest":manifest,
      "physical_reload":reload_verification,
      "paired_CPU_sample":speed,
    }
    save(result_json,report)
    print(json.dumps({"event":"S47_ACTIVE_GREEN_TUNNEL_FINAL",
      "accepted":accepted,
      "new_full":after["new_full"] if after else 0,
      "new_first_color":after["new_correct_first_color"] if after else 34,
      "old_full":after["old_full"] if after else 7,
      "core_bpp":original["actual_complete_bpp"],
      "inclusive_bpp":manifest["all_inclusive_actual_bpp"] if manifest else None,
      "CPU_ratio_sample":speed["speed_ratio_core_over_green"] if speed else None,
      "saved_controller":output.exists()},
      ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
