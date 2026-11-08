"""CALIBRI S49: actual paired-noun recovery with a physically paid Green Tunnel.

S48 diagnosed zero of 32 paired TRAIN noun contrasts solved, although
17/64 individual TRAIN outputs happened to match. S49 optimizes BOTH
opposite nouns as ONE paired objective using actual autoregressive first
token context (not teacher-forced hallucinated contexts). This is an
isolated research TRAIN experiment, not QA or FP32 parity evidence.

The 36M CB59 original source stays exact and frozen, S47 controller's
paid 8-rank weights initialize a 12-rank head to increase bounded
conditional capacity. Only 12-rank active hidden controller is updated.
No hardcoded answers, decoder's output is still causal neural logits.
All 34 already-correct first colors remain unchanged by gating only
AFTER the first token. Exact 7/8 historical old full+EOS and 17/64
baseline correct new full+EOS TRAIN identities are checkpoint guards.
Additional controller bytes + entire physical manifest are charged
within the all-inclusive 0.700bpp cap, not a free sidecar. Independent
S29 QA, previously inspected S26 QA, Jessica chat, LIVE, Render, Supabase,
APK, and production branches are NEVER accessed or changed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v186.pairs import build_pairs
from v174.run import save
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from green_tunnel.contract import PhysicalSource,verify_physical_source

SEED=4949
STEPS=240
PROBES=(40,80,120,160,200,240)
LR=.012
RANK=12
BASE_RANK=8
S45_SOURCE_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S47_SIDE_SHA="02227f62d961b73ed1e7cf89e370d5a4852bf53839ce8fa8f608c44e286cdb43"
MAX_PROBE_ROWS=64

def frozen_hidden(model,sources,ids):
    clear(sources)
    with torch.inference_mode():
        out=model(input_ids=ids[None],use_cache=False,output_hidden_states=True)
        h=out.hidden_states[-1][0,-1].detach().cpu().clone()
        b=out.logits[0,-1].detach().cpu().clone()
    del out
    clear(sources)
    # Recover ordinary (non-inference) tensors for autograd through tunnel.
    return torch.tensor(h.numpy().copy(),dtype=torch.float32),torch.tensor(b.numpy().copy(),dtype=torch.float32)

def read_and_check_manifest(manifest_path,sidecar_path):
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    actual=sidecar_path.read_bytes()
    if (m["core_CB59_sha256"]!=S45_SOURCE_SHA
        or m["sidecar_sha256"]!=S47_SIDE_SHA
        or hashlib.sha256(actual).hexdigest()!=S47_SIDE_SHA
        or len(actual)!=22068 or m["sidecar_bytes"]!=22068
        or m["all_inclusive_total_bytes"]!=3164382
        or float(m["all_inclusive_actual_bpp"])>=.7):
        raise AssertionError("S47 actual trained paid Green Tunnel manifest or bytes changed")
    return m

def expand_rank_no_initial_response_change(old,width):
    enlarged=s47.ActiveGreenTunnel(width,RANK)
    with torch.no_grad():
        enlarged.down.weight.zero_()
        enlarged.up.weight.zero_()
        enlarged.down.weight[:BASE_RANK].copy_(old.down.weight)
        enlarged.up.weight[:,:BASE_RANK].copy_(old.up.weight)
        # New rank features are random but invisible until trained.
        nn.init.normal_(enlarged.down.weight[BASE_RANK:],mean=0,std=.03)
        enlarged.gate.weight.copy_(old.gate.weight)
        enlarged.gate.bias.copy_(old.gate.bias)
    return enlarged

def eligible_pair_cache(model,sources,tok,new,baseline,old_cache):
    cases=[]
    skipped=[]
    for pair in new:
        both={}
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            output=baseline[key]
            first=int(output["actual"][0])
            expected=int(row["targets"][0])
            if first!=expected:
                continue
            if len(row["targets"])!=3 or int(row["targets"][1])==tok.eos_token_id:
                raise AssertionError("TRAIN gold noun/EOS differs")
            ids=torch.cat((row["prompt_ids"],torch.tensor([first],dtype=torch.long)))
            hidden,base=frozen_hidden(model,sources,ids)
            both[side]={"hidden":hidden,"base":base,"noun":int(row["targets"][1]),
                        "key":key,"gold_first":expected}
        if set(both)=={"left","right"}:
            if both["left"]["noun"]==both["right"]["noun"]:
                raise AssertionError("not a contrasting noun pair")
            cases.append({"pair":pair["pair_id"],**both})
        else:
            skipped.append(pair["pair_id"])
    if len(cases)!=17 or len(skipped)!=15:
        raise AssertionError(f"expected S47 17 eligible first-color pairs, actual {len(cases)}")
    return cases,skipped

def pair_cache_stats(cases):
    contrasts=[]
    for row in cases:
        l=row["left"]["hidden"];r=row["right"]["hidden"]
        delta=torch.linalg.vector_norm(l-r)
        denom=(torch.linalg.vector_norm(l)*torch.linalg.vector_norm(r)).clamp(min=1e-12)
        cosine=float(torch.dot(l,r)/denom)
        contrasts.append({"id":row["pair"],"euclidean_diff":float(delta),
            "cosine_similarity":cosine})
    return {"pair_hidden_differences":contrasts,
        "min_difference":min(x["euclidean_diff"] for x in contrasts),
        "max_cosine":max(x["cosine_similarity"] for x in contrasts),
        "number_of_exactly_identical_context_hidden_states":sum(x["euclidean_diff"]==0 for x in contrasts)}

def train_objective(tunnel,head,pairs,old_cache):
    total=[]
    ce=[]
    margins=[]
    for row in pairs:
        left,right=row["left"],row["right"]
        x=s47.apply_delta(left["base"],left["hidden"],tunnel,head)
        y=s47.apply_delta(right["base"],right["hidden"],tunnel,head)
        gl,gr=left["noun"],right["noun"]
        loss_l=F.cross_entropy(x[None],torch.tensor([gl]))
        loss_r=F.cross_entropy(y[None],torch.tensor([gr]))
        # Both sides must win AGAINST the competing contrasting noun:
        # optimizing a single averaged noun would permit global-mode collapse.
        margin_l=x[gl]-x[gr]
        margin_r=y[gr]-y[gl]
        contrast=F.softplus(1.8-margin_l)+F.softplus(1.8-margin_r)
        ce.append(.5*(loss_l+loss_r))
        margins.append(contrast)
    same_task=torch.stack(ce).mean()+3.5*torch.stack(margins).mean()
    o=old_cache
    old_logits=s47.apply_delta(o["base"],o["hidden"],tunnel,head)
    old_CE=F.cross_entropy(old_logits,o["target"])
    # Historical old exact full+EOS are also hard-guarded at each probe.
    loss=same_task+.8*old_CE
    return loss,{"paired_ce":float(torch.stack(ce).mean().detach()),
       "paired_margin":float(torch.stack(margins).mean().detach()),
       "old_replay":float(old_CE.detach())}

def pair_counts(rows,ids):
    both=0
    for id in ids:
        if rows[f"{id}_left"]["full_correct"] and rows[f"{id}_right"]["full_correct"]:
            both+=1
    return both

def full_correct_ids(rows):
    return {k for k,v in rows.items() if v["full_correct"]}

def physical_manifest(base_source,sidecar_path,report_path,out):
    packed=sidecar_path.read_bytes()
    size=len(packed)
    manifest={
      "stage":"S49_REAL_PHYSICALLY_PAID_GREEN_TUNNEL",
      "source_CB59_bytes":len(base_source),
      "source_CB59_sha256":hashlib.sha256(base_source).hexdigest(),
      "S47_controller_parent_sha256":S47_SIDE_SHA,
      "new_controller_rank":RANK,"new_controller_bytes":size,
      "new_controller_sha256":hashlib.sha256(packed).hexdigest(),
      "new_controller_file":sidecar_path.name,
      "all_inclusive_source_controller_manifest":True,
      "source_5bit_indices_and_centroids_frozen":True,
      "no_unpaid_runtime_weights_or_original_FP32":True,
      "one_external_unchanged_host_tokenizer_not_reexported":True,
      "report_is_training_only_not_inference_dependency":True
    }
    converged=False
    for _ in range(50):
        b=json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True).encode("utf-8")
        total=len(base_source)+size+len(b)
        update={"manifest_bytes":len(b),"all_inclusive_total_bytes":total,
          "all_inclusive_bpp":f"{8*total/SOURCE_PARAMS:.12f}"}
        if all(manifest.get(k)==v for k,v in update.items()):
            converged=True;break
        manifest.update(update)
    b=json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True).encode("utf-8")
    total=len(base_source)+size+len(b)
    if not converged or total!=manifest["all_inclusive_total_bytes"]:
        raise AssertionError("all-inclusive physical manifest does not self-account")
    if total>(SOURCE_PARAMS*7)//80:
        raise AssertionError("S49 physical active Green Tunnel exceeds 0.7 bpp")
    out.write_bytes(b)
    return manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source",type=Path)
    ap.add_argument("S47_controller",type=Path)
    ap.add_argument("S47_manifest",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    random.seed(SEED)
    raw=args.source.read_bytes()
    physical=verify_physical_source(PhysicalSource("CB59_K32V8",SOURCE_PARAMS,
        SOURCE_BYTES,S45_SOURCE_SHA),raw)
    read_and_check_manifest(args.S47_manifest,args.S47_controller)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    if len(new)!=32 or len(old)!=24:
        raise AssertionError("Not identical S47 TRAIN composition")
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("S49 core weight gradient is disallowed")
    head=model.lm_head.source().detach().cpu().contiguous()
    clear(sources)
    oldTunnel=s47.ActiveGreenTunnel(320,BASE_RANK)
    oldTunnel.load_state_dict(load_file(str(args.S47_controller)))
    oldTunnel.eval()
    baseline_new=s47.probe(model,sources,oldTunnel,head,tok,new)
    baseline_old=s47.probe(model,sources,oldTunnel,head,tok,old[:4])
    old_keep=full_correct_ids(baseline_old)
    new_keep=full_correct_ids(baseline_new)
    first_correct={k for k,v in baseline_new.items() if v["first_gold_correct"]}
    if len(old_keep)!=7 or len(new_keep)!=17 or len(first_correct)!=34:
        raise AssertionError("Baseline S47 actual full/first-color answers drift")
    if pair_counts(baseline_new,[p["pair_id"] for p in new])!=0:
        raise AssertionError("S48 no-two-sides baseline changed")
    expanded=expand_rank_no_initial_response_change(oldTunnel,320)
    expanded.eval()
    enlarged_new=s47.probe(model,sources,expanded,head,tok,new)
    if enlarged_new!=baseline_new:
        raise AssertionError("New 12-rank zero-padded controller moved native initial greedy predictions")
    cache,skipped=eligible_pair_cache(model,sources,tok,new,baseline_new,None)
    pair_stats=pair_cache_stats(cache)
    # Preserve historical old replay labels using the original S47 cache.
    cache_old,old_identity=s47.parse_training_cache(model,sources,tok,new,old)
    if old_identity!=old_keep:
        raise AssertionError("S47 old TRAIN replay identities drift")
    trainable=sum(p.numel() for p in expanded.parameters())
    if trainable!=2*320*RANK+320+1:
        raise AssertionError("extra unaccounted model parameters")
    # Actual safetensors file header and manifest MUST still fit. Preflight
    # assumes conservative 4096-byte overhead, hard actual later.
    if (SOURCE_BYTES+4*trainable+4096)>(SOURCE_PARAMS*7)//80:
        raise AssertionError("rank-12 controller cannot physically fit complete 0.7bpp")
    optimizer=torch.optim.AdamW(expanded.parameters(),lr=LR,weight_decay=.0001)
    baseline_paired=0
    accepted_state=None
    accepted_new=None
    accepted_old=None
    best_pairs=0
    best_total=17
    logs=[]
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        objective,parts=train_objective(expanded,head,cache,cache_old["old"])
        objective.backward()
        torch.nn.utils.clip_grad_norm_(expanded.parameters(),10.)
        optimizer.step()
        if step not in PROBES:continue
        expanded.eval()
        old_now=s47.probe(model,sources,expanded,head,tok,old[:4])
        lost_old=sorted(old_keep-full_correct_ids(old_now))
        new_now=s47.probe(model,sources,expanded,head,tok,new)
        lost_new=sorted(new_keep-full_correct_ids(new_now))
        lost_color=sorted(first_correct-{k for k,v in new_now.items() if v["first_gold_correct"]})
        full=len(full_correct_ids(new_now))
        pairs_correct=pair_counts(new_now,[p["pair_id"] for p in new])
        improved=(pairs_correct>best_pairs and full>best_total and
                  not lost_old and not lost_new and not lost_color)
        if improved:
            best_pairs=pairs_correct
            best_total=full
            accepted_state={k:v.detach().clone() for k,v in expanded.state_dict().items()}
            accepted_new=new_now
            accepted_old=old_now
        item={"step":step,"objective":float(objective.detach()),"components":parts,
          "new_full_correct":full,"both_sides_pairs_correct":pairs_correct,
          "lost_old7":lost_old,"lost_prior17":lost_new,"lost_prior34_colors":lost_color,
          "accept_train_candidate":improved}
        logs.append(item)
        print(json.dumps({"event":"S49_PAIRED_NOUN_TRAIN_PROGRESS",**item},ensure_ascii=False),flush=True)
        expanded.train()

    accepted=accepted_state is not None
    side=args.output/"S49_ACTIVE_GREEN_TUNNEL_RANK12.safetensors"
    manifest_path=args.output/"S49_ACTIVE_GREEN_TUNNEL_MANIFEST.json"
    manifest=None
    after=None
    reload=None
    cpu=None
    if accepted:
        expanded.load_state_dict(accepted_state)
        expanded.eval()
        save_file({k:v.detach().cpu().contiguous() for k,v in accepted_state.items()},str(side))
        manifest=physical_manifest(raw,side,None,manifest_path)
        reloaded=s47.ActiveGreenTunnel(320,RANK)
        reloaded.load_state_dict(load_file(str(side)))
        reloaded.eval()
        re_new=s47.probe(model,sources,reloaded,head,tok,new)
        re_old=s47.probe(model,sources,reloaded,head,tok,old[:4])
        if re_new!=accepted_new or re_old!=accepted_old:
            raise AssertionError("Real controller physical reload changed full response tokens/EOS")
        if (not old_keep<=full_correct_ids(re_old)
            or not new_keep<=full_correct_ids(re_new)
            or not first_correct<={k for k,v in re_new.items() if v["first_gold_correct"]}):
            raise AssertionError("Physical reload lost original protected TRAIN ids")
        reload={"new64_exact_free_sequences_identical":True,
                "old8_exact_free_sequences_identical":True,
                "new_full":len(full_correct_ids(re_new)),
                "both_contrast_sides_full_correct":pair_counts(re_new,[p["pair_id"] for p in new]),
                "old_full":len(full_correct_ids(re_old))}
        after={"all64":re_new,"all8old":re_old}
        selected=[new[i][side] for i,side in ((6,"left"),(16,"right"),(26,"left"),(31,"right"))]
        times={}
        for label,gate in (("s47",oldTunnel),("s49",reloaded)):
            ds=[]
            for row in selected:
                start=time.perf_counter()
                s47.greedy(model,sources,gate,head,tok,row["question"])
                ds.append(time.perf_counter()-start)
            times[label]=ds
        speed=statistics.median(times["s47"])/statistics.median(times["s49"])
        cpu={"four_TRAIN_cases_paired_wall_seconds":times,
             "median_ratio_vs_S47":speed,
             "sample_above_90pct":speed>=.90,
             "not_independent_CPU_certification":True}
    report={
      "stage":"S49_REAL_PAIRED_NOUN_ACTIVE_GREEN_TUNNEL_TRAIN",
      "source_S45_sha256":S45_SOURCE_SHA,
      "source_S47_controller_sha256":S47_SIDE_SHA,
      "whole_source_physical_bpp":physical["actual_complete_bpp"],
      "TRAIN_ONLY":True,
      "S29_independent_and_S26_previously_inspected_QA_NEVER_USED":True,
      "Jesscia_ordinary_chat_TEST_NOT_RUN":True,
      "LIVE_Render_Supabase_APK_UNCHANGED":True,
      "no_original_FP32_teacher_or_full_dense_export":True,
      "base_controller_rank":BASE_RANK,
      "trainable_new_rank":RANK,
      "controller_trainable_parameter_count":trainable,
      "precommitted_training_steps":STEPS,
      "precommitted_probe_steps":list(PROBES),
      "new_64_old8_TRAIN_only_probes":True,
      "baseline_new_full_correct":17,
      "baseline_pairs_both_correct":baseline_paired,
      "baseline_old7_ids":sorted(old_keep),
      "baseline_new17_ids":sorted(new_keep),
      "baseline_first_gold34_ids":sorted(first_correct),
      "eligible_both_color_correct_pairs":[r["pair"] for r in cache],
      "skipped_pairs_first_color_not_correct":skipped,
      "hidden_contrast_evidence":pair_stats,
      "all_checkpoint_probes":logs,
      "one_or_more_new_exact_paired_noun_skills_acquired":accepted,
      "accepted_pairs_both_correct":best_pairs if accepted else 0,
      "accepted_new_full_correct":best_total if accepted else 17,
      "physical_all_inclusive_manifest":manifest,
      "physical_reload":reload,
      "paired_CPU_sample":cpu,
      "after":after,
      "not_FP32_quality_parity_or_unseen_generalization":True
    }
    save(args.output/"S49_REAL_PAIRED_NOUN_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S49_REAL_PAIRED_NOUN_FINAL",
      "accepted":accepted,
      "both_sides_pairs":report["accepted_pairs_both_correct"],
      "all64_full":report["accepted_new_full_correct"],
      "original_old7_preserved":True,
      "first34_preserved":True,
      "physical_bpp":manifest["all_inclusive_bpp"] if manifest else None,
      "CPU_speed_ratio_sample":cpu["median_ratio_vs_S47"] if cpu else None,
      "sidecar_saved":side.exists()},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
