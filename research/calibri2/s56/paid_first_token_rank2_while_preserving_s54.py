"""CALIBRI S56: add a *physically paid* first-token active Green Tunnel.

Motivation, based ONLY on aggregate frozen evaluations:
S55 independent read-only #42517582295 confirmed S54 20/96 strict full
answers and 2/48 both-correct paired contrasting answers, vs S47 19/96
and 0/48. This is not FP32 parity or proven natural-language
generalization. S55, S53, sealed S29 and previously inspected S26 QA
MUST NOT be imported, copied, trained on or used as checkpoint selection.

S54 preserved first-token answers by enabling its rank12 controller only
after the first generated token. On its ORIGINAL S30 and separate NEW
S54 TRAIN sets this caps first-color correctness at 34/64 and 17/64.
Here we test one targeted architecture change: an additional tiny rank2
320->2->320 neural first-token controller, initially EXACTLY ZERO, while
holding S54 rank12 post-first controller and 36.38M paid compressed CB59
entirely frozen. Additional first controller bytes AND actual manifest
must fit inside the SAME inclusive <=0.700 bpp. At runtime, the new first
gate actually updates causal logits; no hardcoded answer mapping.

Training uses ONLY S30 original 64 explicit TRAIN and S54 novel 64 explicit
TRAIN, replay 7 prior historical TRAIN. All previous S54 original 29 full
answers, S54-new 15 full answers, 12 original paired, 7 new paired,
34 original first colors, 17 augmentation first colors and historic old7
are strict checkpoint guards. Accept only additional whole full+EOS
TRAIN answers, added first colors, no lost protected skills and actual
physical sidecar reload with sample CPU >=90% of S54.
No QA scores used for training or hyperparameter selection.
Jessica ordinary chat tests, LIVE, Render, Supabase, main and APK untouched.
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
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s52 import sequence_eos_protected_green_tunnel as s52
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from v186.pairs import build_pairs
from v176.operators import clear
from v174.run import save

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S54_SHA="88fa9aab9966d5c757013c36ab80bc713e15650dcd8659561b73d5863c022c80"
S54_SIZE=32308
S54_BPP="0.698067535118"
S54_TOTAL=3174504
S54_MANIFEST_SIZE=683
FIRST_RANK=2
LATER_RANK=12
SEED=5656
LR=.012
STEPS=240
PROBES=(40,80,120,160,200,240)
MAX_NEW=10
assert PROBES[-1]==STEPS

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.inference_mode()
def greedy_first_active(model,sources,first,later,head,tok,question):
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
    ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None]
    answers=[]
    for step in range(MAX_NEW):
        clear(sources)
        out=model(input_ids=ids,use_cache=False,output_hidden_states=True)
        hidden=out.hidden_states[-1][0,-1]
        base=out.logits[0,-1]
        tunnel=first if step==0 else later
        logits=s47.apply_delta(base,hidden,tunnel,head)
        token=int(logits.argmax())
        del out
        answers.append(token)
        clear(sources)
        if token==tok.eos_token_id:break
        ids=torch.cat((ids,torch.tensor([[token]],dtype=torch.long)),dim=1)
    return answers

@torch.inference_mode()
def probe(model,sources,first,later,head,tok,pairs):
    rows={}
    for p in pairs:
        for side in ("left","right"):
            item=p[side]
            key=f'{p["pair_id"]}_{side}'
            actual=greedy_first_active(model,sources,first,later,head,tok,item["question"])
            expected=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
            rows[key]={"text":tok.decode(actual,skip_special_tokens=True),
                       "actual":actual,"expected":expected,
                       "full_correct":actual==expected,
                       "first_gold_correct":bool(actual) and actual[0]==expected[0]}
    return rows

def fullcorrect(rows):
    return {key for key,row in rows.items() if row["full_correct"]}

def firstcorrect(rows):
    return {key for key,row in rows.items() if row["first_gold_correct"]}

def pairboth(rows,pairs):
    return sum(rows[f'{p["pair_id"]}_left']["full_correct"] and
               rows[f'{p["pair_id"]}_right']["full_correct"] for p in pairs)

def first_cache(model,sources,tok,pairs,baseline):
    records=[]
    for p in pairs:
        for side in ("left","right"):
            row=p[side]
            key=f'{p["pair_id"]}_{side}'
            correct=baseline[key]["first_gold_correct"]
            hidden,base=s52.frozen_hidden(model,sources,row["prompt_ids"])
            first_id=int(row["targets"][0])
            if correct and int(baseline[key]["actual"][0])!=first_id:
                raise AssertionError("first-color TRAIN guard mismatch")
            records.append({"key":key,"hidden":hidden,"base":base,
                            "gold":first_id,"original_correct":correct})
    return records

def stack_first_records(records):
    return {"hidden":torch.stack([r["hidden"] for r in records]),
            "base":torch.stack([r["base"] for r in records]),
            "target":torch.tensor([r["gold"] for r in records],dtype=torch.long),
            "correct":torch.tensor([r["original_correct"] for r in records],dtype=torch.bool),
            "keys":[r["key"] for r in records]}

def first_train_objective(first,head,orig,added,old7,randomizer):
    # No QA. Real precomputed prompt-end hidden state, native full vocabulary.
    # Fixed TRAIN-only mini-batches, prioritize previously wrong first colors.
    terms=[]
    details={}
    for kind,cache,coeff,n in (("original",orig,1.0,16),("new",added,1.0,16)):
        count=cache["target"].shape[0]
        wrong_idx=torch.where(~cache["correct"])[0].tolist()
        correct_idx=torch.where(cache["correct"])[0].tolist()
        if not wrong_idx or not correct_idx:
            raise AssertionError("missing a prior-correct or first-wrong TRAIN source")
        indices=torch.tensor(
            [wrong_idx[randomizer.randrange(len(wrong_idx))] for _ in range(n//2)]+
            [correct_idx[randomizer.randrange(len(correct_idx))] for _ in range(n//2)],
            dtype=torch.long)
        scores=s47.apply_delta(cache["base"].index_select(0,indices),
                              cache["hidden"].index_select(0,indices),first,head)
        gold=cache["target"].index_select(0,indices)
        ce=F.cross_entropy(scores,gold)
        terms.append(coeff*ce)
        details[kind+"_sampled_first_CE"]=float(ce.detach())
    # Prevent any of the 34 original+17 added correct first tokens
    # and 7 historical old correct first tokens from losing to the
    # strongest WRONG native vocabulary output. No table at inference.
    protections=[]
    for cache in (orig,added,old7):
        idx=torch.where(cache["correct"])[0]
        predictions=s47.apply_delta(cache["base"].index_select(0,idx),
                         cache["hidden"].index_select(0,idx),first,head)
        correct=cache["target"].index_select(0,idx)
        competitor=predictions.detach().clone()
        competitor[torch.arange(len(correct)),correct]=-torch.inf
        strongest_wrong=competitor.argmax(dim=-1)
        margin=predictions.gather(1,correct[:,None]).flatten() - predictions.gather(
            1,strongest_wrong[:,None]).flatten()
        protections.append((F.softplus(1.1-margin).mean()+.15*F.cross_entropy(predictions,correct)))
        details["guard_min_"+str(len(protections))]=float(margin.detach().min())
    loss=sum(terms)+2.5*sum(protections)
    details["loss"]=float(loss.detach())
    return loss,details

def merged_physical_manifest(source,sidecar,manifest_path):
    side_data=sidecar.read_bytes()
    m={"stage":"S56_NATIVE_ALL_INCLUSIVE_PAID_DUAL_CAUSAL_GREEN_TUNNEL",
       "source_CB59_bytes":len(source),
       "source_CB59_sha256":hashlib.sha256(source).hexdigest(),
       "S54_parent_sha256":S54_SHA,
       "dual_rank_later":LATER_RANK,
       "dual_rank_first":FIRST_RANK,
       "dual_controller_bytes":len(side_data),
       "dual_controller_sha256":hashlib.sha256(side_data).hexdigest(),
       "all_physical_weights_and_manifest_paid":True,
       "no_external_teacher_or_extra_weights":True,
       "QA_not_imported_or_used":True}
    converged=False
    for _ in range(30):
        b=json.dumps(m,indent=2,sort_keys=True,ensure_ascii=False).encode()
        total=len(source)+len(side_data)+len(b)
        revision={"manifest_bytes":len(b),"all_inclusive_total_bytes":total,
                  "all_inclusive_bpp":f"{8*total/SOURCE_PARAMS:.12f}"}
        if all(m.get(key)==value for key,value in revision.items()):
            converged=True
            break
        m.update(revision)
    final=json.dumps(m,indent=2,sort_keys=True,ensure_ascii=False).encode()
    if (not converged or len(source)+len(side_data)+len(final)!=m["all_inclusive_total_bytes"]):
        raise AssertionError("S56 sidecar or physical manifest did not converge")
    if 8*len(source+side_data+final)/SOURCE_PARAMS>.7:
        raise AssertionError("S56 total physical budget >0.7bpp")
    manifest_path.write_bytes(final)
    return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_source",type=Path)
    p.add_argument("S54_controller",type=Path)
    p.add_argument("S54_manifest",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    rng=random.Random(SEED)
    source=a.S45_source.read_bytes()
    if len(source)!=SOURCE_BYTES or sha(a.S45_source)!=S45_SHA:
        raise AssertionError("S45 frozen compressed weights don't match SHA")
    if a.S54_controller.stat().st_size!=S54_SIZE or sha(a.S54_controller)!=S54_SHA:
        raise AssertionError("S54 frozen neural controller doesn't match SHA")
    old_manifest=json.loads(a.S54_manifest.read_text(encoding="utf-8"))
    if (a.S54_manifest.stat().st_size!=S54_MANIFEST_SIZE
        or old_manifest["all_inclusive_total_bytes"]!=S54_TOTAL
        or old_manifest["controller_sha256"]!=S54_SHA
        or old_manifest["all_inclusive_bpp"]!=S54_BPP):
        raise AssertionError("S54 physically accepted parent manifest changed")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    added=s54.build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    if len(original)!=32 or len(added)!=32 or len(old)!=24:
        raise AssertionError("TRAIN provenance drift")
    model,sources=create_small_paid_model(CB59(source),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    later=s47.ActiveGreenTunnel(320,LATER_RANK)
    later.load_state_dict(load_file(str(a.S54_controller)))
    later.eval().requires_grad_(False)
    first=s47.ActiveGreenTunnel(320,FIRST_RANK)
    # Its up-projection is initialized to ZERO, so original S54
    # first-token native logits are EXACTLY unchanged initially.
    if not torch.equal(first.up.weight,torch.zeros_like(first.up.weight)):
        raise AssertionError("New first-token adapter must be zero initially")
    no_first_original=s47.probe(model,sources,later,head,tok,original)
    no_first_added=s47.probe(model,sources,later,head,tok,added)
    no_first_old=s47.probe(model,sources,later,head,tok,old[:4])
    start_orig=probe(model,sources,first,later,head,tok,original)
    start_add=probe(model,sources,first,later,head,tok,added)
    start_old=probe(model,sources,first,later,head,tok,old[:4])
    if (start_orig!=no_first_original or start_add!=no_first_added
        or start_old!=no_first_old):
        raise AssertionError("zero-effect first gate must reproduce S54 EVERY native response")
    orig_keep=fullcorrect(start_orig)
    add_keep=fullcorrect(start_add)
    old_keep=fullcorrect(start_old)
    orig_color_keep=firstcorrect(start_orig)
    add_color_keep=firstcorrect(start_add)
    if (len(orig_keep)!=29 or pairboth(start_orig,original)!=12
        or len(add_keep)!=15 or pairboth(start_add,added)!=7
        or len(old_keep)!=7 or len(orig_color_keep)!=34 or len(add_color_keep)!=17):
        raise AssertionError("S54 actual original TRAIN exact baseline changed")
    original_first=stack_first_records(first_cache(model,sources,tok,original,start_orig))
    added_first=stack_first_records(first_cache(model,sources,tok,added,start_add))
    old7_records=first_cache(model,sources,tok,old[:4],start_old)
    old7_first=stack_first_records([r for r in old7_records if r["key"] in old_keep])
    if len(old7_first["keys"])!=7 or not bool(old7_first["correct"].all()):
        raise AssertionError("S56 original seven old first tokens not all guardable")
    # Strong independent bpp preflight including FP32 rank2 plus
    # safetensors header and manifest reserve.
    future_weights=(sum(t.numel() for t in later.state_dict().values())+
                   sum(t.numel() for t in first.state_dict().values()))
    if future_weights!=8001+1601:
        raise AssertionError("S56 neural parameter accounting wrong")
    if SOURCE_BYTES+future_weights*4+2048>(SOURCE_PARAMS*7)//80:
        raise AssertionError("S56 rank2 physically cannot fit 0.7bpp")
    optimizer=torch.optim.AdamW(first.parameters(),lr=LR,weight_decay=.0001)
    best=None
    best_rows=None
    best_key=(len(orig_keep)+len(add_keep),len(orig_color_keep)+len(add_color_keep))
    attempts=[]
    print(json.dumps({"event":"S56_UNCHANGED_S54_FROZEN_INITIAL",
        "old7":7,"original29":29,"added15":15,"original12_pairs":12,
        "added7_pairs":7,"original_first34":34,"added_first17":17,
        "first_gate_rank":FIRST_RANK,"paid_total_params":future_weights},ensure_ascii=False),flush=True)
    for step in range(1,STEPS+1):
        first.train()
        optimizer.zero_grad(set_to_none=True)
        loss,details=first_train_objective(first,head,original_first,added_first,old7_first,rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(first.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        if step not in PROBES:continue
        first.eval()
        cur_old=probe(model,sources,first,later,head,tok,old[:4])
        cur_orig=probe(model,sources,first,later,head,tok,original)
        cur_add=probe(model,sources,first,later,head,tok,added)
        oc=fullcorrect(cur_orig);ac=fullcorrect(cur_add);kc=fullcorrect(cur_old)
        fc=firstcorrect(cur_orig);gc=firstcorrect(cur_add)
        op=pairboth(cur_orig,original);np=pairboth(cur_add,added)
        protected=(orig_keep<=oc and add_keep<=ac and old_keep<=kc
                   and orig_color_keep<=fc and add_color_keep<=gc
                   and op>=12 and np>=7)
        score=(len(oc)+len(ac),len(fc)+len(gc))
        improve=protected and score[0]>44 and score>best_key
        if improve:
            best_key=score
            best={key:t.detach().clone() for key,t in first.state_dict().items()}
            best_rows=(cur_orig,cur_add,cur_old)
        record={"step":step,"original_full64":len(oc),
           "augmented_full64":len(ac),"old8_full":len(kc),
           "original_both_pairs":op,"augmented_both_pairs":np,
           "original_color_first":len(fc),"augmented_color_first":len(gc),
           "lost_original29":sorted(orig_keep-oc),
           "lost_augmented15":sorted(add_keep-ac),
           "lost_old7":sorted(old_keep-kc),
           "lost_original_correct34_first":sorted(orig_color_keep-fc),
           "lost_augmented_correct17_first":sorted(add_color_keep-gc),
           "first_training_loss":details,
           "accept_TRAIN_candidate":improve}
        attempts.append(record)
        print(json.dumps({"event":"S56_FIRST_TOKEN_PAID_TRAIN_PROGRESS",**record},
              ensure_ascii=False),flush=True)
    side=a.out/"S56_DUAL_ACTIVE_PAID_GREEN_TUNNEL.safetensors"
    manifest_path=a.out/"S56_DUAL_ACTIVE_PAID_GREEN_TUNNEL_MANIFEST.json"
    saved=False
    physical=None
    reload_checks=None
    cpu=None
    if best is not None:
        first.load_state_dict(best)
        first.eval()
        tensors={("later."+k):v.detach().cpu().contiguous() for k,v in later.state_dict().items()}
        tensors.update({("first."+k):v.detach().cpu().contiguous() for k,v in first.state_dict().items()})
        save_file(tensors,str(side))
        physical=merged_physical_manifest(source,side,manifest_path)
        reload_tensors=load_file(str(side))
        nfirst=s47.ActiveGreenTunnel(320,FIRST_RANK)
        nlater=s47.ActiveGreenTunnel(320,LATER_RANK)
        nfirst.load_state_dict({k.split(".",1)[1]:v for k,v in reload_tensors.items() if k.startswith("first.")})
        nlater.load_state_dict({k.split(".",1)[1]:v for k,v in reload_tensors.items() if k.startswith("later.")})
        nfirst.eval()
        nlater.eval()
        verify_old=probe(model,sources,nfirst,nlater,head,tok,old[:4])
        verify_orig=probe(model,sources,nfirst,nlater,head,tok,original)
        verify_add=probe(model,sources,nfirst,nlater,head,tok,added)
        if (verify_orig!=best_rows[0] or verify_add!=best_rows[1] or
            verify_old!=best_rows[2] or
            not orig_keep<=fullcorrect(verify_orig) or
            not add_keep<=fullcorrect(verify_add) or
            not old_keep<=fullcorrect(verify_old) or
            not orig_color_keep<=firstcorrect(verify_orig) or
            not add_color_keep<=firstcorrect(verify_add)):
            raise AssertionError("S56 reloaded physical dual first+later model failed guard")
        checked=[original[i][side]["question"] for i,side in
                 ((6,"left"),(16,"right"),(26,"left"),(31,"right"))]
        times={}
        for label,fg,lg in (("S54",None,later),("S56",nfirst,nlater)):
            ds=[]
            for question in checked:
                now=time.perf_counter()
                if label=="S54":
                    s47.greedy(model,sources,lg,head,tok,question)
                else:
                    greedy_first_active(model,sources,fg,lg,head,tok,question)
                ds.append(time.perf_counter()-now)
            times[label]=ds
        ratio=statistics.median(times["S54"])/statistics.median(times["S56"])
        cpu={"four_TRAIN_cases_seconds":times,"speed_ratio_vs_S54":ratio,
             "meets_90pct_sample_guard":ratio>=.9,
             "not_independent_CPU_certification":True}
        if ratio>=.9:
            saved=True
            reload_checks={
              "all_original64_identical_to_accepted_candidate":True,
              "all_augmented64_identical_to_accepted_candidate":True,
              "all_historical_old8_identical_to_accepted_candidate":True,
              "original_full":len(fullcorrect(verify_orig)),
              "augmented_full":len(fullcorrect(verify_add)),
              "original_first":len(firstcorrect(verify_orig)),
              "augmented_first":len(firstcorrect(verify_add)),
              "original_both_pairs":pairboth(verify_orig,original),
              "augmented_both_pairs":pairboth(verify_add,added),
              "old_full":len(fullcorrect(verify_old))}
        else:
            side.unlink()
            manifest_path.unlink()
            physical=None
    report={
      "stage":"S56_FIRST_TOKEN_TRAIN_ONLY_PAID_NEURAL_GREEN_TUNNEL",
      "S54_parent_core_SHA":S45_SHA,
      "S54_parent_green_tunnel_SHA":S54_SHA,
      "source_S55_frozen_QA_CI_aggregate_only_no_QA_access":42517582295,
      "S29_S26_S53_S55_QA_NEVER_IMPORTED_OR_SCORED":True,
      "all_sources_TRAIN_only":True,
      "paid_frozen_CB59_36M_unchanged":True,
      "paid_frozen_post_first_S54_rank12_unchanged":True,
      "new_first_token_rank2_paid_inclusive":True,
      "original29_and_augmented15_strict_EOS_preservation":True,
      "old7_unchanged_guard":True,
      "original_both12_and_augmented_both7_strict_guard":True,
      "first_original34_and_augmented17_strict_guard":True,
      "S54_baseline_original29_augmented15_old7":True,
      "predeclared_steps":STEPS,
      "predeclared_probe_steps":list(PROBES),
      "all_train_probe_results":attempts,
      "new_controller_physically_saved_only_if_accepted":saved,
      "accepted_all_inclusive_physical_manifest":physical,
      "accepted_physical_reload_full_sequences":reload_checks,
      "four_TRAIN_example_CPU_ratio":cpu,
      "no_generalization_or_original_FP32_parity_claim":True,
      "no_Jessica_chat_testing":True,
      "no_LIVE_main_Render_Supabase_APK_modifications":True}
    save(a.out/"S56_FIRST_TOKEN_PAID_GREEN_TUNNEL_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S56_PAID_FIRST_TOKEN_FINAL",
      "accepted":saved,
      "original_full":reload_checks["original_full"] if reload_checks else 29,
      "augmented_full":reload_checks["augmented_full"] if reload_checks else 15,
      "new_orig_first":reload_checks["original_first"] if reload_checks else 34,
      "new_aug_first":reload_checks["augmented_first"] if reload_checks else 17,
      "original_both_pairs":reload_checks["original_both_pairs"] if reload_checks else 12,
      "augmented_both_pairs":reload_checks["augmented_both_pairs"] if reload_checks else 7,
      "old7_retained":True,
      "physical_bpp":physical["all_inclusive_bpp"] if physical else None,
      "CPU_sample_ratio":cpu["speed_ratio_vs_S54"] if cpu else None,
      "saved_dual_neural_weight_file":saved},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
