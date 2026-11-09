"""CALIBRI S74 — real GPU MPS gradient training of protected first-token neural adapter.

S72 completed a sealed FIRST-AND-ONLY 96-prompt evaluation. The existing
S67 and S71 did not differ on that genuinely unseen set (both 16/96
strict full+EOS, 56/96 first token). The separately pretrained 135M
FP32 baseline is NOT same architecture and failed strict output format.
S72 results are historical evidence ONLY; NO S72 (or any other inspected
QA) prompts, scores, token traces, generated labels, or model-choice
signals are loaded or used here.

This is an isolated GPU migration of an existing S73 research TRAIN
algorithm. The original archived S73 result and S72 independent QA stay
immutable; this research branch verifies actual GPU gradient tensors and
the exact old skill set on CPU, not unseen QA.
This attempt is a new architecture hypothesis motivated by the
TRAIN-only inability of rank2 first-token tunnel to preserve AND learn
new first-token bindings. Freeze exact S71 sidecar; attach a small
rank16 320->16->320 residual gate ONLY to the first generated token.
Its gradients see ONLY authored S30/S54/S69/new S73 TRAIN prompts.
Use frozen pretrained-codebook 36M real hidden states and exact FULL
vocabulary gold-vs-top-rival first-token contrasts. Old S71 first rank2
and later rank128 context mixer and compressed 36M weights remain frozen.

Accept ONLY if every old correct full/first TRAIN identity remains
correct (including 7 historical), there is a genuine gain on the S69
and/or S73 TRAIN examples, and exact real autoregressive full-token
CPU replay passes. Real CPU speed >=90% and FULLY PAID controller+source+
manifest <=.7 bpp required. No generalization or FP32 parity claim.
CPU-only. GitHub draft research ONLY; no Jessica/LIVE/Render/Supabase.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s71 import neural_context_gate_preserve_s67 as s71
from v186.pairs import build_pairs, tokenize_one

S71_SHA = "0b4c03c639f2500fa98c97b4def58af4d4b55da3a0dd83267c2c4405d0925bbb"
RANK = 16
SEED = 730073
STEPS = 240
PROBES = (48, 96, 144, 192, 240)
LR = 0.0035
RIVALS = 8
TRAIN_NAMES = (
    ("Ravkin", "Yodela"),
    ("Minelt", "Dorav"),
    ("Palevi", "Zunor"),
    ("Ceiran", "Elnov"),
)
TRAIN_FRAMES = (
    "TRAIN S73 ticker: Clerk {a} has the two-part inventory key {left}. Clerk {b} has the distinct key {right}. Print only the two words of {a}'s key.",
    "TRAIN S73 keycard: In a departure schedule, the key beside {b} is {right}; the key beside {a} is {left}. Which exact color and noun belong to {a}?",
    "TRAIN S73 rack: Rack owner {a} is allocated the label {left}. Rack owner {b} is allocated {right}. Return the full two-word label for owner {a}.",
    "TRAIN S73 label: The assignment is {a} : {left}; also {b} : {right}. State exactly the two assigned words for {a} in order.",
)
assert len(TRAIN_NAMES) == len(TRAIN_FRAMES) == 4

def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

class FirstTokenResidual16(nn.Module):
    """No prompt-ID lookup or EVAL labels. Native hidden-state neural residual."""
    def __init__(self, frozen):
        super().__init__()
        self.frozen = frozen
        self.frozen.eval().requires_grad_(False)
        self.down = nn.Linear(320, RANK, bias=False)
        self.up = nn.Linear(RANK, 320, bias=False)
        self.gate = nn.Linear(320, 1, bias=True)
        with torch.no_grad():
            nn.init.normal_(self.down.weight, mean=0., std=.035)
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()

    def increment(self, hidden):
        normalized = F.layer_norm(hidden, (320,))
        return (self.up(torch.tanh(self.down(normalized))) *
                torch.sigmoid(self.gate(normalized)))

    def forward(self, hidden):
        return self.frozen(hidden) + self.increment(hidden)

def new_s73_train(tok):
    result = []
    for color, nouns in s30.TRAIN_ELIGIBLE.items():
        if len(nouns) != 2:
            raise AssertionError("S73 must use S30 TRAIN-only color/object pairs")
        first, second = (f"{color} {n}" for n in nouns)
        for i, (person, other) in enumerate(TRAIN_NAMES):
            template = TRAIN_FRAMES[i]
            left_q = template.format(a=person,b=other,left=first,right=second)
            right_q = template.format(a=person,b=other,left=second,right=first)
            if "TRAIN S73" not in left_q or left_q == right_q:
                raise AssertionError("missing S73 TRAIN provenance")
            left = tokenize_one(tok, left_q, first)
            right = tokenize_one(tok, right_q, second)
            if left["first_expected_id"] != right["first_expected_id"]:
                raise AssertionError("TRAIN opposition first expected color mismatch")
            result.append({"pair_id":73000+len(result), "left":left, "right":right})
    if len(result) != 16 or len({p[side]["question"] for p in result for side in ("left","right")}) != 32:
        raise AssertionError("S73 authored TRAIN matrix malformed")
    return result

def caches_for_first(model,sources,first,head,tok,pairs,base_rows):
    """One-time frozen native hidden/logit cache; no gradients through 36M source."""
    records=s56.first_cache(model,sources,tok,pairs,base_rows)
    built=[]
    for record in records:
        hidden=record["hidden"]
        base=record["base"]
        gold=int(record["gold"])
        with torch.no_grad():
            base_first=base+F.linear(first(hidden),head)
            rivals=base_first.topk(RIVALS+2).indices.tolist()
            rivals=[int(x) for x in rivals if int(x)!=gold][:RIVALS]
            if len(rivals)!=RIVALS:
                raise AssertionError("first token vocabulary rival shortage")
            ids=torch.tensor([gold]+rivals,dtype=torch.long)
            values=base_first[ids].detach().clone()
            projections=head.index_select(0,ids).detach().clone()
        built.append({
            "key":record["key"],
            "hidden":hidden.detach(),
            "previously_correct":bool(record["original_correct"]),
            "logits":values,
            "head":projections,
        })
    if not built:
        raise AssertionError("empty frozen real TRAIN first-token cache")
    return built

def contrastive_margin(first,records,rg,n_wrong,n_correct):
    wrong=[r for r in records if not r["previously_correct"]]
    right=[r for r in records if r["previously_correct"]]
    if not right:
        raise AssertionError("no protected prior correct first tokens")
    sampled=[]
    if n_wrong:
        sampled.extend(rg.choice(wrong) for _ in range(n_wrong)) if wrong else None
    if n_correct:
        sampled.extend(rg.choice(right) for _ in range(n_correct))
    if not sampled:
        raise AssertionError("no supervised first-token samples selected")
    hidden=torch.stack([r["hidden"] for r in sampled])
    frozen_values=torch.stack([r["logits"] for r in sampled])
    reduced_head=torch.stack([r["head"] for r in sampled])
    updated=frozen_values+torch.einsum("bh,bkh->bk",first.increment(hidden),reduced_head)
    gold=updated[:,0]
    competitor=torch.logsumexp(updated[:,1:],dim=1)
    margin=gold-competitor
    protected_mask=torch.tensor([r["previously_correct"] for r in sampled],dtype=torch.bool)
    # Old correctness is more important than new TRAIN wins.
    objective=torch.zeros((),dtype=torch.float32,device=first.up.weight.device)
    if protected_mask.any():
        objective=objective+3.0*F.softplus(2.0-margin[protected_mask]).mean()
    if (~protected_mask).any():
        objective=objective+F.softplus(1.0-margin[~protected_mask]).mean()
    return objective

def full_probe(model,sources,first,later,head,tok,old,aug,hist,fresh,new73):
    data=s69.take_probe(model,sources,first,later,head,tok,old,aug,hist,fresh)
    data["new_S73_TRAIN"]=s56.probe(model,sources,first,later,head,tok,new73)
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_immutable_CB59",type=Path)
    p.add_argument("S67_rank128_controller",type=Path)
    p.add_argument("S71_frozen_neural_controller",type=Path)
    p.add_argument("S71_paid_manifest",type=Path)
    p.add_argument("S61_lossless_compressed_source",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    rg=random.Random(SEED)

    raw=a.S45_immutable_CB59.read_bytes()
    controller=a.S71_frozen_neural_controller.read_bytes()
    envelope=a.S61_lossless_compressed_source.read_bytes()
    manifest=json.loads(a.S71_paid_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or digest(raw)!=s65.S45_SHA or
        digest(controller)!=S71_SHA or
        manifest["combined_controller_rank128_with_context_alpha_SHA256"]!=S71_SHA or
        manifest["physical_S61_CB61_SHA256"]!=digest(envelope) or
        manifest["physical_all_inclusive_bytes"]!=len(envelope)+len(controller)+a.S71_paid_manifest.stat().st_size or
        float(manifest["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S73 must start from one exact physically paid frozen S71")
    original67=load_file(str(a.S67_rank128_controller))
    original71=load_file(str(a.S71_frozen_neural_controller))
    if (digest(a.S67_rank128_controller.read_bytes())!=s69.CANDIDATE_SHA or
        any(not torch.equal(original71[k],v) for k,v in original67.items())):
        raise AssertionError("S73 must not modify old S67 neural controller")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    old=s30.make_training_pairs(tok)
    aug=s54.build_new_train(tok)
    hist=build_pairs(tok,"TRAIN")
    fresh=s69.build_new_train(tok)
    new73=new_s73_train(tok)
    if tuple(map(len,(old,aug,hist,fresh,new73)))!=(32,32,24,16,16):
        raise AssertionError("S73 TRAIN data provenance changed")
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    first.load_state_dict({k[6:]:v for k,v in original71.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in original71.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    mixer=s71.NeuralContextMixer(later,
        original71["later.up.weight"]+original71["context_delta.up.weight"])
    mixer.context_alpha.load_state_dict({
        "weight":original71["context_gate.weight"],"bias":original71["context_gate.bias"]})
    mixer.eval().requires_grad_(False)

    baseline=full_probe(model,sources,first,mixer,head,tok,old,aug,hist,fresh,new73)
    oldkeep=(
        s58.correct(baseline["original"]),s58.correct(baseline["augmented"]),
        s58.correct(baseline["historical"]),s58.first_correct(baseline["original"]),
        s58.first_correct(baseline["augmented"]))
    base=s69.quality(baseline,old,aug,fresh,oldkeep)
    if tuple(base[k] for k in (
        "S30_original_full","S54_augmented_full","historical_full",
        "S30_original_first","S54_augmented_first","old_original_pairs_both",
        "old_augmented_pairs_both","new_fresh_TRAIN_full","new_fresh_TRAIN_first",
        "new_fresh_TRAIN_both_pairs"))!=(42,28,7,58,46,13,7,11,20,2):
        raise AssertionError("S71 frozen full TRAIN reference no longer matches accepted CI")
    keep={
        name:(s58.correct(rows),s58.first_correct(rows))
        for name,rows in baseline.items()
    }
    before_scores={
        "fresh_S69_full":len(keep["fresh_TRAIN"][0]),
        "fresh_S69_first":len(keep["fresh_TRAIN"][1]),
        "new_S73_full":len(keep["new_S73_TRAIN"][0]),
        "new_S73_first":len(keep["new_S73_TRAIN"][1]),
    }
    new_train_sha=digest(json.dumps([
        {"id":p["pair_id"],"l":p["left"]["question"],"r":p["right"]["question"]}
        for p in new73],sort_keys=True,ensure_ascii=False).encode())

    group_cache={
        "original":caches_for_first(model,sources,first,head,tok,old,baseline["original"]),
        "augmented":caches_for_first(model,sources,first,head,tok,aug,baseline["augmented"]),
        "fresh_S69":caches_for_first(model,sources,first,head,tok,fresh,baseline["fresh_TRAIN"]),
        "new_S73":caches_for_first(model,sources,first,head,tok,new73,baseline["new_S73_TRAIN"]),
        "historical":[r for r in caches_for_first(
            model,sources,first,head,tok,hist[:4],baseline["historical"])
            if r["key"] in keep["historical"][0]],
    }
    if len(group_cache["historical"])!=7 or not all(r["previously_correct"] for r in group_cache["historical"]):
        raise AssertionError("failed to preserve seven historical answers")
    selected=None
    best=(before_scores["fresh_S69_full"]+before_scores["new_S73_full"],
          before_scores["fresh_S69_first"]+before_scores["new_S73_first"])
    initial_weights={k:v.clone() for k,v in first.state_dict().items()}
    # Only the new Green Tunnel module and TRAIN caches run on the real
    # Apple GPU; the frozen compressed core remains on CPU.
    if not torch.backends.mps.is_available():
        raise RuntimeError("S74 REAL GPU training requires Apple Silicon MPS; CPU fallback forbidden")
    gpu_device=torch.device("mps")
    smoke_a=torch.randn(128,128,device=gpu_device,requires_grad=True)
    smoke_b=(smoke_a@smoke_a.T).square().mean()
    smoke_b.backward()
    if smoke_a.grad is None or smoke_a.grad.device.type!="mps":
        raise AssertionError("MPS GPU backward proof failed")
    torch.mps.synchronize()
    del smoke_a,smoke_b
    for rows in group_cache.values():
        for row in rows:
            for key in ("hidden","logits","head"):
                row[key]=row[key].detach().to(gpu_device)
            if any(row[key].device.type!="mps" for key in ("hidden","logits","head")):
                raise AssertionError("TRAIN caches must reside on MPS GPU")
    first_new=FirstTokenResidual16(copy.deepcopy(first)).to(gpu_device)
    if any(p.device.type!="mps" for p in first_new.parameters() if p.requires_grad):
        raise AssertionError("S74 trainable adapter is not on GPU")
    first_new.train()
    gpu_training_start=time.perf_counter()
    gpu_gradient_proven=False
    optimizer=torch.optim.AdamW(
        [p for p in first_new.parameters() if p.requires_grad],
        lr=LR,weight_decay=0.00005)
    progress=[]
    print(json.dumps({
        "event":"S73_FIRST_TOKEN_NEURAL_TRAIN_ONLY_START",
        "S71_immutable_controller_SHA":S71_SHA,
        "fresh_new_S73_TRAIN_hash":new_train_sha,
        "original_S71_Green_Tunnel_preserved":True,
        "baseline":base,
        "fresh_TRAIN_baseline":before_scores,
        "new_first_stage_neural_trainable_parameters":
           sum(p.numel() for p in first_new.parameters() if p.requires_grad),
        "only_TRAIN_supervision_no_QA":True,
    }),flush=True)
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        loss=(
           .75*contrastive_margin(first_new,group_cache["original"],rg,6,10)+
           1.0*contrastive_margin(first_new,group_cache["augmented"],rg,8,10)+
           2.0*contrastive_margin(first_new,group_cache["fresh_S69"],rg,10,10)+
           2.0*contrastive_margin(first_new,group_cache["new_S73"],rg,10,10)+
           2.0*contrastive_margin(first_new,group_cache["historical"],rg,0,7)
        )
        if not torch.isfinite(loss):
            raise AssertionError("S73 native frozen feature TRAIN loss is nonfinite")
        loss.backward()
        if not gpu_gradient_proven:
            train_grads=[p.grad for p in first_new.parameters() if p.requires_grad and p.grad is not None]
            if not train_grads or any(g.device.type!="mps" for g in train_grads):
                raise AssertionError("S74 requires actual GPU-computed adapter gradients")
            gpu_gradient_proven=True
        torch.nn.utils.clip_grad_norm_(
           [p for p in first_new.parameters() if p.requires_grad],
           4.0,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.equal(v,first.state_dict()[k]) for k,v in initial_weights.items()):
            raise AssertionError("S73 attempted to mutate frozen S71 first rank2")
        if step not in PROBES:
            continue
        # Full sequence guard must run on EXACT normal CPU inference.
        torch.mps.synchronize()
        first_new=first_new.to("cpu").eval()
        rows=full_probe(model,sources,first_new,mixer,head,tok,old,aug,hist,fresh,new73)
        m=s69.quality(rows,old,aug,fresh,oldkeep)
        lost={name:{
          "full":sorted(keep[name][0]-s58.correct(r)),
          "first":sorted(keep[name][1]-s58.first_correct(r))}
          for name,r in rows.items()}
        preservation=(all(not x for v in lost.values() for x in v.values()) and
                      m["S67_original_20_opposition_pairs_preserved"])
        val={
           "fresh_S69_full":len(s58.correct(rows["fresh_TRAIN"])),
           "fresh_S69_first":len(s58.first_correct(rows["fresh_TRAIN"])),
           "new_S73_full":len(s58.correct(rows["new_S73_TRAIN"])),
           "new_S73_first":len(s58.first_correct(rows["new_S73_TRAIN"])),
        }
        score=(val["fresh_S69_full"]+val["new_S73_full"],
               val["fresh_S69_first"]+val["new_S73_first"])
        eligible=preservation and score>best
        row={"step":step,"TRAIN_approx_margin_loss_not_QA":float(loss.detach()),
             "all_original_S71_correct_identities_kept":preservation,
             "losses":lost,
             "TRAIN_new_correct":val,
             "selected_preliminary_TRAIN_candidate":eligible,**m}
        progress.append(row)
        print(json.dumps({"event":"S73_PROTECTED_TRUE_FULL_AUTOREGRESSIVE_TRAIN_PROBE",
                          **row}),flush=True)
        if eligible:
            best=score
            selected={"step":step,"score":score,"replayed":row,
                      "neural":{k:v.detach().cpu().clone() for k,v in first_new.state_dict().items()
                               if not k.startswith("frozen.")}}
        first_new=first_new.to(gpu_device).train()

    torch.mps.synchronize()
    gpu_train_plus_cpu_guard_seconds=time.perf_counter()-gpu_training_start
    first_new=first_new.to("cpu").eval()
    result={
      "stage":"CALIBRI_S74_REAL_APPLE_MPS_GPU_FIRST_TOKEN_TRAIN",
      "real_training_device":"mps",
      "real_GPU_gradients_proven":gpu_gradient_proven,
      "GPU_training_plus_full_CPU_safety_guards_seconds":gpu_train_plus_cpu_guard_seconds,
      "GPU_neural_trainable_parameter_count":sum(p.numel() for p in first_new.parameters() if p.requires_grad),
      "strict_no_CPU_training_fallback":True,
      "S72_one_time_QA_permanently_embargoed_from_gradients_and_selection":True,
      "same_S73_TRAIN_only_no_new_QA_or_test":True,
      "all_earlier_heldout_QA_embargoed":True,
      "TRAIN_before":before_scores,
      "TRAIN_probe_results":progress,
      "eligible_preliminary_TRAIN_candidate":selected is not None,
      "no_unseen_generalization_or_FP32_parity_claim":True,
      "GPU_used":True,"Apple_MPS_GPU_used":True,"Jessica_chat_test_run":False,
      "LIVE_modified":False,
    }
    if selected:
        first_new.load_state_dict(selected["neural"],strict=False)
        first_new.eval()
        cpu={}
        for label,fc in (("S71",first),("S73",first_new)):
            samples=make_samples(tok)
            elapsed=[]
            for _,_,q,_ in samples:
                start=time.perf_counter()
                s56.greedy_first_active(model,sources,fc,mixer,head,tok,q)
                elapsed.append(time.perf_counter()-start)
            cpu[label]=statistics.median(elapsed)
        ratio=cpu["S71"]/cpu["S73"]
        result["CPU_speed_ratio_S73_vs_S71"]=ratio
        result["CPU_speed_at_least_90percent_of_S71"]=ratio>=.90
        if ratio>=.90:
            paid=dict(original71)
            for key,weight in selected["neural"].items():
                paid["first_increment."+key]=weight.detach().cpu().contiguous()
            pth=a.output_dir/"S74_REAL_GPU_MPS_TRAIN_ONLY_PAID_FIRST_NEURAL.safetensors"
            save_file(paid,str(pth))
            saved=load_file(str(pth))
            if any(not torch.equal(v,saved[k]) for k,v in original71.items()):
                raise AssertionError("S73 physically saved original S71 neural skills changed")
            side=pth.read_bytes()
            m={
              "stage":"CALIBRI_S74_REAL_GPU_MPS_TRAIN_ONLY_PAID_FIRST_NEURAL_ADAPTER",
              "S71_frozen_controller_SHA256":S71_SHA,
              "S74_GPU_extra_first_neural_rank":RANK,
              "S74_replayed_exact_S73_TRAIN_sha256":new_train_sha,
              "S61_unchanged_exact_compressed_source_SHA256":digest(envelope),
              "S74_GPU_full_controller_bytes":len(side),
              "S74_GPU_full_controller_SHA256":digest(side),
              "all_prior_QA_permanently_embargoed":True,
              "selected_first_restoration_STEP_TRAIN_only":selected["step"],
              "Apple_MPS_GPU_real_trainable_module":True,
              "train_metrics":selected["replayed"],
              "no_unseen_inference_or_FP32_parity_test":True,
              "no_LIVE_or_Jessica_tests":True,
            }
            for _ in range(32):
                rawm=json.dumps(m,indent=2,sort_keys=True).encode()
                total=len(envelope)+len(side)+len(rawm)
                update={
                  "manifest_bytes":len(rawm),
                  "physical_all_inclusive_bytes":total,
                  "physical_all_inclusive_bpp":f"{total*8/SOURCE_PARAMS:.12f}"
                }
                if all(m.get(k)==v for k,v in update.items()):break
                m.update(update)
            manifest_bytes=json.dumps(m,indent=2,sort_keys=True).encode()
            if m["physical_all_inclusive_bytes"] != len(envelope)+len(side)+len(manifest_bytes):
                raise AssertionError("S73 physical size manifest did not converge")
            if m["physical_all_inclusive_bytes"]*8/SOURCE_PARAMS > .7:
                raise AssertionError("S73 exceeded fully paid 0.7 bpp")
            (a.output_dir/"S74_REAL_GPU_ACTUAL_ALL_BYTES_PAID_MANIFEST.json").write_bytes(manifest_bytes)
            result["candidate_SHA256"]=digest(side)
            result["physical_bpp"]=m["physical_all_inclusive_bpp"]
            result["selected_TRAIN_step"]=selected["step"]
        else:
            result["eligible_preliminary_TRAIN_candidate"]=False
    (a.output_dir/"S74_REAL_GPU_ACTUAL_TRAIN_ONLY_PROTECTED_FIRST_RESTORE.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S74_FINAL_REAL_MPS_GPU_TRAIN_ONLY",
                      "preliminary_training_gain":result["eligible_preliminary_TRAIN_candidate"],
                      "physical_bpp":result.get("physical_bpp"),
                      "CPU_speed_ratio":result.get("CPU_speed_ratio_S73_vs_S71"),
                      "GPU_training_plus_CPU_guard_seconds":gpu_train_plus_cpu_guard_seconds,
                      "real_MPS_gradients_confirmed":gpu_gradient_proven,
                      "no_QA_executed_no_FP32_parity_claim":True}),flush=True)

if __name__=="__main__":
    main()
