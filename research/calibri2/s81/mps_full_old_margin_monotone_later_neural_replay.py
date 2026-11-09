"""CALIBRI S81: REAL GPU late rank16 with all-old-token monotone margin replay and strict full-response acceptance.

Motivation (based ONLY on already known, frozen TRAIN outcomes):
 S74 trained first-rank16 residual with true MPS gradients; physical 0.65933 bpp,
 protected old full+first abilities, improved new TRAIN from 11/32 to 16/32.
 S76: 200/200 entire S30/S54/history/S69/S73 TRAIN answers IDENTICAL
 compressed CPU vs TEMP FP32-decoded MPS full-core (not quality/QA).
 S77: real incremental GPU KV-cache 200/200 parity, CPU/GPU ~1.45x.

This new experiment:
 - Freezes entire original CB59 36M model and S74 first-rank16,
   S71 later-rank128 and neural router.
 - Uses actual 36M full reconstructed model ON MPS for extracting
   *all* eligible TRAIN late-token hidden states and full vocab logits.
 - Trains extra later-rank16 (not a lookup) via MPS real-gradient
   native head projections and old-skill margin protection.
 - Verifies complete autoregressive token+EOS answers on the GPU
   at predeclared steps, then exact compressed-model CPU BEFORE
   physically accepting a chosen TRAIN-only candidate.
 - Evaluate a PRE-REGISTERED TRUST-REGION linear scale grid at every
   protected TRAIN checkpoint. An early-exit GPU verifier checks EVERY
   previously correct original answer identity and first token before
   expensive 232 full greedy. Never accept negative or forgetting steps.
 - Accept ONLY full new TRAIN improvement and zero losses of ANY
   old full/first correct identity (S30/S54/history/S69/S73),
   CPU >= 0.9x frozen S74 and actual all-in physical <= 0.7 bpp.

ALL S72, S68, S55/S53, S29 and other previously inspected independent
QA are EMBARGOED: no imports/words/prompt extraction, no evaluation on
them, no objective/score based on QA, no QA-retest. No generalization
or FP32 parity claims. No LIVE main/Render/Supabase/Jessica changes.
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
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s71 import neural_context_gate_preserve_s67 as s71
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s77.full_mps_incremental_kv_cache_200_train_parity import dense_mps_cached_greedy
from v186.pairs import build_pairs,tokenize_one

SEED=780078
ALPHAS=(.125,.25,.5,1.)
RANK=16
RIVALS=12
STEPS=120
CHECKS=(30,60,90,120)
LR=.002
S71_SHA=s73.S71_SHA
S73_TRAIN_SHA="c1de452dcd55746e9cb2c2f479a0764c90359c122859e2cf3f7c8d782db8cd14"
NEW_PAIRS=16
NEW_NAMES=(
    ("Cavinor","Tolade"),("Deslin","Arbinor"),
    ("Valmia","Orestan"),("Hovelin","Nerazi"),
)
NEW_TEMPLATES=(
    "TRAIN S78 atlas: The mapped tag assigned to {a} is {left}. The separate mapped tag assigned to {b} is {right}. State just the two tokens for {a}.",
    "TRAIN S78 clinic sheet: On the sheet, {b} is labeled {right}; {a} is labeled {left}. Reply only with the color/object combination for {a}.",
    "TRAIN S78 expedition pack: {a}'s packing symbol is {left}. Conversely {b}'s symbol is {right}. Transcribe the full two-word symbol for {a}.",
    "TRAIN S78 astronomy log: Register {a} records {left} whereas register {b} records {right}. Output the two-word value recorded for {a}.",
)
assert len(NEW_NAMES)==len(NEW_TEMPLATES)==4

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()

def new_train(tok):
    rows=[]
    for color,nouns in s30.TRAIN_ELIGIBLE.items():
        if len(nouns)!=2:raise AssertionError("S78 TRAIN color/noun provenance")
        left,right=[f"{color} {n}" for n in nouns]
        for i,(name,other) in enumerate(NEW_NAMES):
            s=NEW_TEMPLATES[i]
            a=tokenize_one(tok,s.format(a=name,b=other,left=left,right=right),left)
            b=tokenize_one(tok,s.format(a=name,b=other,left=right,right=left),right)
            if a["first_expected_id"]!=b["first_expected_id"]:
                raise AssertionError("S78 TRAIN opposing nouns must share first color")
            rows.append({"pair_id":78000+len(rows),"kind":"TRAIN_S78_only",
                         "left":a,"right":b})
    if len(rows)!=NEW_PAIRS or len({p[side]["question"] for p in rows for side in ("left","right")})!=32:
        raise AssertionError("S78 new authored TRAIN matrix mismatch")
    return rows

class LateResidual16(nn.Module):
    """Tiny paid neural late-token rank16. No prompt IDs/answer tables."""
    def __init__(self,frozen):
        super().__init__()
        self.probe_scale=1.0
        self.frozen=frozen
        self.frozen.eval().requires_grad_(False)
        self.down=nn.Linear(320,RANK,bias=False)
        self.up=nn.Linear(RANK,320,bias=False)
        self.gate=nn.Linear(320,1,bias=True)
        with torch.no_grad():
            nn.init.normal_(self.down.weight,std=.035)
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()
    def increment(self,hidden):
        norm=F.layer_norm(hidden,(320,))
        return self.up(torch.tanh(self.down(norm)))*torch.sigmoid(self.gate(norm))
    def forward(self,hidden):
        # The residual is TESTED at variable coefficients on TRAIN only.
        # Before any actual candidate is saved, alpha is baked into its
        # up matrix, so no runtime unaccounted coefficient is needed.
        return self.frozen(hidden)+self.probe_scale*self.increment(hidden)

def controllers(original):
    first0=s47.ActiveGreenTunnel(320,2)
    later0=s47.ActiveGreenTunnel(320,128)
    first0.load_state_dict({k[6:]:v for k,v in original.items() if k.startswith("first.")})
    later0.load_state_dict({k[6:]:v for k,v in original.items() if k.startswith("later.")})
    first0.eval().requires_grad_(False)
    later0.eval().requires_grad_(False)
    first=s73.FirstTokenResidual16(first0)
    inc={k.removeprefix("first_increment."):v for k,v in original.items()
         if k.startswith("first_increment.")}
    if set(inc)!={"down.weight","up.weight","gate.weight","gate.bias"}:
        raise AssertionError("S78 missing actual S74 trained first-token residual")
    for name,param in (
       ("down.weight",first.down.weight),("up.weight",first.up.weight),
       ("gate.weight",first.gate.weight),("gate.bias",first.gate.bias)):
        with torch.no_grad():param.copy_(inc[name])
    first.eval().requires_grad_(False)
    mixer=s71.NeuralContextMixer(later0,
        original["later.up.weight"]+original["context_delta.up.weight"])
    mixer.context_alpha.load_state_dict({
        "weight":original["context_gate.weight"],"bias":original["context_gate.bias"]})
    mixer.eval().requires_grad_(False)
    return first,mixer

def protected_probe(model,sources,first,later,head,tok,pairs_by_family):
    return {name:s56.probe(model,sources,first,later,head,tok,pairs)
            for name,pairs in pairs_by_family.items()}

def get_protected(rows):
    return {name:{"full":s58.correct(data),"first":s58.first_correct(data)}
            for name,data in rows.items()}

def losses_from(protected,actual):
    now=get_protected(actual)
    return {name:{"full":sorted(protected[name]["full"]-now[name]["full"]),
                  "first":sorted(protected[name]["first"]-now[name]["first"])}
            for name in protected}

def totals(rows,pairs):
    good=get_protected(rows)
    return {name:{"full":len(good[name]["full"]),
                  "first":len(good[name]["first"]),
                  "both_opposed_full":s58.both_correct(rows[name],p)}
            for name,p in pairs.items()}

@torch.no_grad()
def build_gpu_supervision_cache(dense,head,tok,frozen_later,pairs,baseline):
    """One GPU full-core forward per TRAIN item, noun/EOS at gold prefix.
    No gradients through source. Source is temporarily expanded to FP32
    in MPS memory but NEVER serialized.
    """
    out={}
    for family,case_pairs in pairs.items():
        records=[]
        for pair in case_pairs:
            for side in ("left","right"):
                item=pair[side];key=f'{pair["pair_id"]}_{side}'
                prior=baseline[family][key]
                gold=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
                if len(gold) not in (2,3):
                    if family=="history":continue
                    raise AssertionError("new/old two-token+EOS target must have three tokens")
                if family!="history" and len(gold)!=3:
                    raise AssertionError("non-historical TRAIN must have two content tokens plus EOS")
                # Only train second/EOS when FIRST token is actually
                # correct on the immutable S74 model.
                if not prior["first_gold_correct"]:continue
                ids=item["prompt_ids"].tolist()+gold[:-1]
                if len(ids)>320:raise AssertionError("malformed S78 TRAIN prompt too long")
                inp=torch.tensor([ids],device="mps",dtype=torch.long)
                result=dense(input_ids=inp,use_cache=False,output_hidden_states=True)
                hidden=result.hidden_states[-1][0]
                base=result.logits[0]
                start=len(ids)-(len(gold)-1)
                for local_step in range(len(gold)-1):
                    h=hidden[start+local_step].detach().clone()
                    b=base[start+local_step].detach()
                    gold_id=int(gold[local_step+1])
                    reference=b+F.linear(frozen_later(h),head)
                    strongest=reference.topk(RIVALS+2).indices.tolist()
                    rivals=[int(k) for k in strongest if int(k)!=gold_id][:RIVALS]
                    if len(rivals)!=RIVALS:raise AssertionError("insufficient native vocab rivals")
                    ids=torch.tensor([gold_id]+rivals,device="mps",dtype=torch.long)
                    rec={
                        "id":key,"kind":"noun" if local_step==0 else "EOS",
                        "hidden":h,
                        "base_scores":reference.index_select(0,ids).clone(),
                        "head_rows":head.index_select(0,ids).clone(),
                        "previously_full":bool(prior["full_correct"])}
                    records.append(rec)
                del result
        out[family]=records
        print(json.dumps({"event":"S78_ACTUAL_GPU_FULL36M_TRAIN_HIDDEN_CACHE",
            "family":family,"native_reconstructed_36M_GPU_features":len(records),
            "prompts":len(case_pairs)*2}),flush=True)
    return out

def pack_full_protected_native_margins(cache):
    """Record all formerly correct gold-vs-full-native top12 rivals on GPU.
    Includes Noun and EOS positions of every previously full-correct TRAIN
    question; all protected identities checked, no tiny mini-batch proxy.
    """
    all_rows=[record for rows in cache.values() for record in rows
              if record["previously_full"]]
    if len(all_rows)<200:raise AssertionError("S81 incomplete protected full-answer GPU cache")
    hidden=torch.stack([r["hidden"] for r in all_rows]).detach()
    logits=torch.stack([r["base_scores"] for r in all_rows]).detach()
    head=torch.stack([r["head_rows"] for r in all_rows]).detach()
    if any(t.device.type!="mps" for t in (hidden,logits,head)):
        raise AssertionError("S81 full old replay must run on physical MPS GPU")
    # Frozen S74 gold vs every known strongest full-vocabulary challenger.
    oldmargin=(logits[:,0]-logits[:,1:].max(dim=1).values).detach()
    return {"hidden":hidden,"base":logits,"head":head,
            "reference_gold_margin":oldmargin,
            "old_correct_count":len(all_rows)}

def old_monotone_loss(layer,all_old):
    h=all_old["hidden"]
    delta=layer.increment(h)
    scores=all_old["base"]+torch.einsum("bh,bkh->bk",delta,all_old["head"])
    actual_margin=scores[:,0]-scores[:,1:].max(dim=1).values
    original=all_old["reference_gold_margin"]
    # Enforce nondecreasing originally correct old gold-vs-rival margins,
    # not just a sampled cross-entropy term. Target +0.01 headroom.
    violation=F.relu(original+.01-actual_margin)
    reg=20.0*violation.square().mean()
    # Also penalize fragility when baseline old scores were barely correct.
    safe_gold=F.softplus(.30-actual_margin).mean()
    return reg+.35*safe_gold, {
        "strict_all_old_gpu_margin_replay":True,
        "old_margin_loss_decrease_max":float((original-actual_margin).detach().max()),
        "old_margin_count":len(original),
        "max_margin_guard_violation":float(violation.detach().max()),
    }

def new_unlearned_after_first_loss(layer,cache,rg,num):
    """Train only previously wrong complete answers under S74 baseline;
    previously correct TRAIN contexts participate in ALL-old replay.
    Original S74 first token is immutable throughout.
    """
    wrong=[r for r in cache if not r["previously_full"]]
    if not wrong:return torch.zeros((),device="mps")
    sampled=[wrong[rg.randrange(len(wrong))] for _ in range(num)]
    h=torch.stack([r["hidden"] for r in sampled])
    logits=torch.stack([r["base_scores"] for r in sampled])
    rows=torch.stack([r["head_rows"] for r in sampled])
    new_logits=logits+torch.einsum("bh,bkh->bk",layer.increment(h),rows)
    return F.cross_entropy(new_logits,torch.zeros(num,device="mps",dtype=torch.long))


@torch.inference_mode()
def fast_all_old_skill_guard(dense,first,later,head,tok,pairs,protected):
    # Never shortcut old-skill safety on a subset. Check *every* old
    # previously full or first correct TRAIN answer under true native
    # full greedy MPS, stopping as soon as any identity would be lost.
    for family,case_pairs in pairs.items():
        lookup={f'{p["pair_id"]}_{side}':p[side]
                for p in case_pairs for side in ("left","right")}
        for key in sorted(protected[family]["full"]|protected[family]["first"]):
            item=lookup[key]
            answer,_=dense_mps_cached_greedy(dense,tok,first,later,head,item["question"])
            expected=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
            if (key in protected[family]["full"] and answer!=expected):
                return {"family":family,"key":key,"type":"full","actual":answer}
            if (key in protected[family]["first"] and
                (not answer or answer[0]!=expected[0])):
                return {"family":family,"key":key,"type":"first","actual":answer}
    return None


def full_mps_probe(dense,first,later,head,tok,pairs):
    scores={}
    for family,all_pairs in pairs.items():
        data={}
        for pair in all_pairs:
            for side in ("left","right"):
                item=pair[side];key=f'{pair["pair_id"]}_{side}'
                tokens,_=dense_mps_cached_greedy(dense,tok,first,later,head,item["question"])
                gold=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
                data[key]={"actual":tokens,"expected":gold,
                    "full_correct":tokens==gold,
                    "first_gold_correct":bool(tokens) and tokens[0]==gold[0]}
        scores[family]=data
    return scores

def physically_paid(envelope,rawside,metrics,dirpath):
    manifest={
      "stage":"CALIBRI_S81_NEURAL_LATE_RANK16_ALL_OLD_MONOTONE_GPU_REPLAY_TRAIN_ONLY",
      "immutable_source_CB59_original_SHA256":s65.S45_SHA,
      "original_S74_full_controller_SHA256":metrics["original_S74_SHA256"],
      "new_controller_SHA256":sha(rawside),
      "physically_paid_compressed_source_sha256":sha(envelope),
      "paid_CB61_source_bytes":len(envelope),
      "paid_new_controller_bytes":len(rawside),
      "new_rank16_neural_parameters":RANK*320*2+321,
      "added_real_MPS_GPU_trained_module":True,
      "S72_QA_and_all_inspected_independent_QA_permanently_embargoed":True,
      "all_original_S74_old_correct_TRAIN_ids_protected":True,
      "candidate_selection_used_only_TRAIN":True,
      "training_protection":metrics["selected_probe"],
      "original_CPU_native_speed_ratio":metrics["cpu_speed_ratio"],
      "no_F16_or_approximate_accuracy_claim":True,
      "no_Jessica_CHAT_test_LIVE_changes":True,
    }
    for _ in range(40):
        encoded=json.dumps(manifest,sort_keys=True,indent=2).encode()
        physical=len(envelope)+len(rawside)+len(encoded)
        fields={"manifest_bytes":len(encoded),
                "full_physical_bytes":physical,
                "all_inclusive_physical_bpp":f"{physical*8/SOURCE_PARAMS:.12f}"}
        if all(manifest.get(k)==v for k,v in fields.items()):break
        manifest.update(fields)
    b=json.dumps(manifest,sort_keys=True,indent=2).encode()
    if manifest["full_physical_bytes"] != len(envelope)+len(rawside)+len(b):
        raise AssertionError("S78 physical manifest did not converge")
    if len(envelope)+len(rawside)+len(b) > .7*SOURCE_PARAMS/8:
        raise AssertionError("S78 exceeding original physically paid 0.7 bpp")
    (dirpath/"S81_ALL_INCLUSIVE_PAID_MONOTONE_GPU_TRAIN_MANIFEST.json").write_bytes(b)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_frozen_CB59",type=Path)
    p.add_argument("S74_frozen_controller",type=Path)
    p.add_argument("S74_paid_manifest",type=Path)
    p.add_argument("S61_exact_CB61",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S78 REAL Apple MPS GPU required, CPU fallback not permitted")
    torch.manual_seed(SEED);torch.set_num_threads(2)
    rng=random.Random(SEED)
    raw=a.S45_frozen_CB59.read_bytes()
    side=a.S74_frozen_controller.read_bytes()
    envelope=a.S61_exact_CB61.read_bytes()
    origin=json.loads(a.S74_paid_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        sha(side)!=origin["S74_GPU_full_controller_SHA256"] or
        origin["selected_first_restoration_STEP_TRAIN_only"]!=48 or
        origin["S71_frozen_controller_SHA256"]!=S71_SHA or
        origin["S61_unchanged_exact_compressed_source_SHA256"]!=sha(envelope) or
        origin["physical_all_inclusive_bytes"]!=len(envelope)+len(side)+a.S74_paid_manifest.stat().st_size or
        float(origin["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S78 only accepts immutable physically paid S74 rank16 GPU checkpoint")
    weights=load_file(str(a.S74_frozen_controller))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    pairs={
        "original":s30.make_training_pairs(tok),
        "augmented":s54.build_new_train(tok),
        "history":build_pairs(tok,"TRAIN")[:4],
        "S69":s69.build_new_train(tok),
        "S73":s73.new_s73_train(tok),
        "S78":new_train(tok),
    }
    if [len(x) for x in pairs.values()]!=[32,32,4,16,16,16]:
        raise AssertionError("S78 protected authored TRAIN population changed")
    control_sha=sha(json.dumps([
        {"family":name,"id":p["pair_id"],
         "left":p["left"]["question"],"right":p["right"]["question"]}
        for name,pp in pairs.items() for p in pp],sort_keys=True).encode())
    if sha(json.dumps([{"id":p["pair_id"],"l":p["left"]["question"],"r":p["right"]["question"]}
                      for p in pairs["S73"]],sort_keys=True,ensure_ascii=False).encode())!=S73_TRAIN_SHA:
        raise AssertionError("S73 previous TRAIN provenance mutated")
    # Native compressed CPU is the exact deployment-inference authority.
    cpu_model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    cpu_model.eval().requires_grad_(False)
    head_cpu=cpu_model.lm_head.source().detach().cpu().contiguous()
    cpu_first,cpu_mixer=controllers(weights)
    baseline_cpu=protected_probe(cpu_model,sources,cpu_first,cpu_mixer,head_cpu,tok,pairs)
    keep=get_protected(baseline_cpu)
    initial=totals(baseline_cpu,pairs)
    if (
        initial["original"]["full"],initial["augmented"]["full"],
        initial["history"]["full"],initial["S69"]["full"],
        initial["S73"]["full"]
    ) != (44,31,7,16,12):
        raise AssertionError("S74 trusted original native TRAIN full-skill reference drift")
    if (
        initial["original"]["first"],initial["augmented"]["first"],
        initial["S69"]["first"],initial["S73"]["first"]
    ) != (64,56,32,32):
        raise AssertionError("S74 expected real first-token train abilities changed")
    print(json.dumps({"event":"S78_REAL_NATIVE_COMPRESSED_CPU_FULL_TRAIN_BASELINE",
        "family_count":len(pairs),"all_authored_TRAIN":232,
        "old_neural_controller_SHA":sha(side),
        "prior_protected":initial,
        "TRAIN_population_sha256":control_sha}),flush=True)
    # Full expanded model and hidden/LM-head all actually on Apple GPU.
    dense=s30.dense_from_paid(CB59(raw))
    if sum(p.numel() for p in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S78 temporary GPU source wrong parameter count")
    dense=dense.to("mps").eval().requires_grad_(False)
    gpu_head=dense.lm_head.weight.detach()
    gpu_first,gpu_mixer=controllers(weights)
    gpu_first=gpu_first.to("mps").eval().requires_grad_(False)
    gpu_mixer=gpu_mixer.to("mps").eval().requires_grad_(False)
    before_gpu=full_mps_probe(dense,gpu_first,gpu_mixer,gpu_head,tok,pairs)
    mismatches={
        name:[key for key,record in baseline_cpu[name].items()
              if record["actual"]!=before_gpu[name][key]["actual"]]
        for name in pairs}
    if any(mismatches.values()):
        raise AssertionError("S78 GPU baseline not exact CPU parity on all TRAIN; unsafe to use MPS as candidate guard: "+repr(mismatches))
    print(json.dumps({"event":"S78_REAL_36M_MPS_GPU_232_FULL_NATIVE_CPU_PARITY",
        "TRAIN_answer_token_sequences_identical":232,
        "all_232_protected_native_compressed_CPU_outputs_match_real_GPU":True}),flush=True)
    cache=build_gpu_supervision_cache(dense,gpu_head,tok,gpu_mixer,pairs,baseline_cpu)
    for name in ("original","augmented","history","S69","S73","S78"):
        if not cache[name]:raise AssertionError("S78 lacks native GPU TRAIN hidden cache "+name)
    protected_all=pack_full_protected_native_margins(cache)
    layer=LateResidual16(gpu_mixer).to("mps")
    if any(p.device.type!="mps" for p in layer.parameters() if p.requires_grad):
        raise AssertionError("S78 new late-rank16 trainable parameters not on physical GPU")
    trainable=[p for p in layer.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable)!=RANK*320*2+321:
        raise AssertionError("S78 wrong number of physically accountable GPU trainable parameters")
    opt=torch.optim.AdamW(trainable,lr=LR,weight_decay=.00001)
    best=None
    bestkey=(initial["S69"]["full"]+initial["S73"]["full"]+initial["S78"]["full"],
             initial["S78"]["full"])
    print(json.dumps({"event":"S78_TRUE_GPU_NEURAL_LATE_RANK16_TRAIN_START",
        "physical_Apple_MPS_gpu":True,"true_GPU_trainable_parameter_count":sum(x.numel() for x in trainable),
        "GPU_cached_frozen_36M_native_features_by_family":{k:len(v) for k,v in cache.items()},
        "initial_native_TRAIN_scores":initial,
        "old_SKILL_protection":True,"any_independent_QA_used":False}),flush=True)
    progress=[]
    training_start=time.perf_counter()
    verified_mps_gradients=False
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        old_loss,guard_metrics=old_monotone_loss(layer,protected_all)
        loss=(
            1.2*new_unlearned_after_first_loss(layer,cache["original"],rng,12)+
            1.2*new_unlearned_after_first_loss(layer,cache["augmented"],rng,12)+
            1.8*new_unlearned_after_first_loss(layer,cache["S69"],rng,12)+
            1.8*new_unlearned_after_first_loss(layer,cache["S73"],rng,12)+
            2.4*new_unlearned_after_first_loss(layer,cache["S78"],rng,16)+
            old_loss)
        if not torch.isfinite(loss):
            raise AssertionError("S78 GPU gradient objective became NaN/Inf")
        loss.backward()
        if not verified_mps_gradients:
            grads=[p.grad for p in trainable if p.grad is not None]
            if not grads or any(g.device.type!="mps" for g in grads):
                raise AssertionError("S78 actual Apple GPU gradients were NOT computed")
            verified_mps_gradients=True
        torch.nn.utils.clip_grad_norm_(trainable,3.5,error_if_nonfinite=True)
        opt.step()
        if step not in CHECKS:continue
        layer.eval()
        # Without retuning to independent QA, verify a frozen TRAIN-only
        # trust-region line search in order of smallest intervention.
        # Backtracking requires EXACT prior correct identities (and old
        # both-opposed-pair wins) to persist under native GPU greedy.
        # Full 232-output probes are used only after cheap early-exit
        # *complete* old skill check passes.
        for alpha in ALPHAS:
            layer.probe_scale=float(alpha)
            early=fast_all_old_skill_guard(dense,gpu_first,layer,gpu_head,tok,pairs,keep)
            if early is not None:
                entry={"step":step,"alpha":alpha,
                       "old_skill_guard_passed":False,
                       "first_lost_old_TRAIN_id":early,
                       "full_232_GREEDY_probe_skipped_due_to_real_forgetting":True,
                       "eligible":False}
                progress.append(entry)
                print(json.dumps({"event":"S80_GPU_TRAIN_GUARDED_ALPHA_PROBE",**entry}),flush=True)
                continue
            actual=full_mps_probe(dense,gpu_first,layer,gpu_head,tok,pairs)
            lost=losses_from(keep,actual)
            metrics=totals(actual,pairs)
            no_old_losses=all(not loss for group in lost.values() for loss in group.values())
            old_pair_guard=(metrics["original"]["both_opposed_full"]>=initial["original"]["both_opposed_full"]
              and metrics["augmented"]["both_opposed_full"]>=initial["augmented"]["both_opposed_full"])
            key=(metrics["S69"]["full"]+metrics["S73"]["full"]+metrics["S78"]["full"],
                 metrics["S78"]["full"])
            eligible=no_old_losses and old_pair_guard and key>bestkey and key[0]>bestkey[0]
            entry={"step":step,"alpha":alpha,
                "native_MPS_232_full_greedy":True,
                "all_prior_full_and_first_correct_identities_kept":no_old_losses,
                "all_old_margin_replay_MPS_gpu":guard_metrics,
                "old_both_opposed_pairs_kept":old_pair_guard,
                "old_lost_TRAIN_identities":lost,
                "actual_correct_counts":metrics,
                "all_training_only_no_QA":True,
                "eligible":eligible}
            progress.append(entry)
            print(json.dumps({"event":"S80_GPU_TRAIN_GUARDED_ALPHA_PROBE",**entry}),flush=True)
            if eligible:
                bestkey=key
                w={k:v.detach().cpu().clone() for k,v in layer.state_dict().items()
                   if not k.startswith("frozen.")}
                # Bake the accepted real alpha into the only 320x16
                # trainable output to pay its bytes exactly once.
                w["up.weight"].mul_(float(alpha))
                best={"step":step,"alpha":alpha,"metrics":entry,"state":w}
        layer.probe_scale=1.0
    torch.mps.synchronize()
    gpu_elapsed=time.perf_counter()-training_start
    report={
        "stage":"CALIBRI_S81_REAL_MPS_GPU_ALL_OLD_MONOTONE_REPLAY_LATE_RANK16_TRAIN_ONLY",
        "original_S74_frozen_controller_sha256":sha(side),
        "original_all_inclusive_physical_bpp":origin["physical_all_inclusive_bpp"],
        "source_model_unique_parameters":SOURCE_PARAMS,
        "GPU_parity_against_real_CPU_native_232_TRAIN_prompts":True,
        "GPU_gradient_verified":verified_mps_gradients,
        "full_36M_GPU_frozen_source_hidden_caches":True,
        "real_GPU_training_plus_4_step_TRUST_REGION_MPS_guards_seconds":gpu_elapsed,
        "predeclared_safe_alpha_grid":ALPHAS,
        "old_native_margin_replay_all_correct_gold_tokens":protected_all["old_correct_count"],
        "all_old_margin_replay_every_GPU_optimizer_step":True,
        "original_TRAIN_scores":initial,
        "TRAIN_population_sha256":control_sha,
        "TRAIN_safe_alpha_line_search_full_232_native_greedy_and_early_rejects":progress,
        "selected_candidate_only_if_exact_native_GPU_TRAIN_gain_and_no_previous_loss":best is not None,
        "all_previously_inspected_independent_QA_embargoed_no_QA_run":True,
        "no_unseen_generalization_or_FP32_parity_claim":True,
        "no_LIVE_Jessica_changes":True,
    }
    if best:
        cpu_later=LateResidual16(cpu_mixer)
        for k,v in best["state"].items():
            name,param=k.split(".",1)
            with torch.no_grad():
                getattr(cpu_later,name).state_dict()[param].copy_(v)
        cpu_later.eval().requires_grad_(False)
        final_cpu=protected_probe(cpu_model,sources,cpu_first,cpu_later,head_cpu,tok,pairs)
        lost_native=losses_from(keep,final_cpu)
        actual_native=totals(final_cpu,pairs)
        changed=any(actual_native[name]!=best["metrics"]["actual_correct_counts"][name]
                    for name in pairs)
        acceptable=(not changed and
            all(not l for group in lost_native.values() for l in group.values()) and
            actual_native["original"]["both_opposed_full"]>=initial["original"]["both_opposed_full"] and
            actual_native["augmented"]["both_opposed_full"]>=initial["augmented"]["both_opposed_full"])
        report["final_real_compressed_CPU_TRAIN_scores"]=actual_native
        report["actual_CPU_vs_GPU_candidate_result_mismatch"]=changed
        report["lost_correct_S74_full_CPU_native_ids"]=lost_native
        report["final_real_CPU_native_quality_guard_passed"]=acceptable
        if acceptable:
            samples=make_samples(tok)
            measured={}
            for name,late in (("S74",cpu_mixer),("S78",cpu_later)):
                durations=[]
                for _,_,question,_ in samples:
                    t=time.perf_counter()
                    s56.greedy_first_active(cpu_model,sources,cpu_first,late,head_cpu,tok,question)
                    durations.append(time.perf_counter()-t)
                measured[name]=statistics.median(durations)
            ratio=measured["S74"]/measured["S78"]
            report["actual_native_CPU_speed_ratio_S78_vs_S74"]=ratio
            if ratio>=.9:
                updated=dict(weights)
                for k,v in best["state"].items():
                    updated["later_increment."+k]=v.detach().cpu().contiguous()
                output_path=a.output_dir/"S81_REAL_GPU_MONOTONE_REPLAY_PROTECTED_CANDIDATE.safetensors"
                save_file(updated,str(output_path))
                if any(not torch.equal(v,load_file(str(output_path))[key])
                       for key,v in weights.items()):
                    raise AssertionError("S78 physical candidate altered frozen S74 weights")
                paid=physically_paid(envelope,output_path.read_bytes(),{
                    "original_S74_SHA256":sha(side),
                    "selected_probe":best["metrics"],
                    "cpu_speed_ratio":ratio},a.output_dir)
                report["accepted_physically_paid_candidate_SHA256"]=sha(output_path.read_bytes())
                report["fully_paid_bpp"]=paid["all_inclusive_physical_bpp"]
                report["selected_new_later_rank16_TRAIN_step"]=best["step"]
                report["selected_real_neural_alpha"]=best["alpha"]
                report["accepted_as_TRAIN_only_research_candidate"]=True
            else:
                report["accepted_as_TRAIN_only_research_candidate"]=False
        else:report["accepted_as_TRAIN_only_research_candidate"]=False
    else:report["accepted_as_TRAIN_only_research_candidate"]=False
    (a.output_dir/"S81_RESEARCH_ONLY_REAL_GPU_MONOTONE_REPLAY_RESULTS.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S81_FINAL_FULL_OLD_MARGIN_GPU_AND_CPU_GUARDS",
        "TRAIN_candidate_accepted":report["accepted_as_TRAIN_only_research_candidate"],
        "physical_bpp":report.get("fully_paid_bpp"),
        "gpu_training_and_probes_seconds":gpu_elapsed,
        "real_GPU_gradient":verified_mps_gradients,
        "independent_QA_tested_or_reused":False}),flush=True)

if __name__=="__main__":main()
