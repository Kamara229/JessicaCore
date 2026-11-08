"""CALIBRI S54 — independently written TRAIN augmentation, frozen S52 protection.

S53 #42515879784 is already a ONE-TIME DIAGNOSTIC, permanently embargoed
from all future gradients, checkpoint decisions, prompts and benchmarks:
S52 scored 12/96 exact full+EOS unseen novel-language answers vs S47 18/96,
zero BOTH-correct opposing pairs for either. We do NOT load any S53 module,
asset, questions, responses or scored errors, nor S29 final sealed QA or
previously inspected S26 QA. The S54 branch starts from S52, NOT from S53.

S54 asks only whether entirely new *explicitly TRAIN-labelled* contexts
can be learned by the SAME already-paid rank12 active Green Tunnel while
preserving all original S52 29/64 full+EOS, 12 two-sided paired identities,
34 first-colors and old7. No QA-driven selection and no generalization
claim. All input weights physically reloaded with frozen SHA256 checks.
The neural controller is trained causally, no answer table at inference.
A candidate must be a strict TRAIN improvement and pass same 0.7 bpp
physical-reload and CPU >=0.9x S52 four-TRAIN-case sample guard.
No Jessica ordinary-chat tests, LIVE, main, Render, Supabase or APK edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s52 import sequence_eos_protected_green_tunnel as s52
from v186.pairs import tokenize_one,build_pairs
from v176.operators import clear
from v174.run import save

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S52_SHA="80e52005a2c1ed97937a302c12b4d0b620264e13fe91246880544a9cb41eb4cd"
S52_SIZE=32308
S52_MANIFEST_SIZE=896
S52_PHYSICAL_TOTAL=3174717
S52_BPP="0.698114373422"
SEED=5454
LR=.003
STEPS=240
PROBES=(40,80,120,160,200,240)
RANK=12
BATCH_NEW=12

# Newly authored TRAIN strings with a literal TRAIN tag and unique names
# deliberately absent from S53's independent one-time diagnostic, S29 QA,
# and the original S30 64 TRAIN examples. No test questions are imported.
COLOR_PAIRS=(
 ("red","bowl","drum"),
 ("green","bowl","rope"),
 ("blue","ring","rope"),
 ("yellow","ring","drum")
)
OWNERS=(
 ("Cora","Bram"),("Tessa","Kian"),("Luma","Rafi"),("Soren","Veda"),
 ("Amal","Quinn"),("Yara","Niko"),("Uma","Dax"),("Eli","Pia")
)
NEW_TRAIN_FRAMES=(
 "TRAIN expansion {i}: Codebook {a} => {left}; codebook {b} => {right}. Emit the two-word code at {a}.",
 "TRAIN exercise {i}: Reorder by labels. {b}={right}, and {a}={left}. Only write the two words paired with {a}.",
 "TRAIN contrast {i}: {a} points to {left} || {b} points to {right}. Transcribe {a}'s color then noun.",
 "TRAIN diagram {i}: {b} marks {right}; by comparison {a} marks {left}. Output the two-word marking for {a}.",
 "TRAIN record {i}: Under header {a}: {left}. Under {b}: {right}. Return the two words for header {a}.",
 "TRAIN routing {i}: A={a}=>{left}; B={b}=>{right}. What two-word code should {a} emit?",
 "TRAIN cross-check {i}: Read {b} ({right}), read {a} ({left}). Which exact two words label {a}?",
 "TRAIN classification {i}: {a} maps onto {left}; {b} maps onto {right}. State {a}'s color-and-noun result."
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_new_train(tok):
    pairs=[]
    for color,left,right in COLOR_PAIRS:
        for i,(a,b) in enumerate(OWNERS):
            template=NEW_TRAIN_FRAMES[i]
            value_left=f"{color} {left}"
            value_right=f"{color} {right}"
            question_l=template.format(i=9000+len(pairs),a=a,b=b,left=value_left,right=value_right)
            question_r=template.format(i=9000+len(pairs),a=a,b=b,left=value_right,right=value_left)
            pl=tokenize_one(tok,question_l,value_left)
            pr=tokenize_one(tok,question_r,value_right)
            if pl["first_expected_id"]!=pr["first_expected_id"]:
                raise AssertionError("S54 opposing noun answers must share their first color")
            pairs.append({"pair_id":9000+len(pairs),"left":pl,"right":pr,"color":color})
    if len(pairs)!=32:
        raise AssertionError("S54 new TRAIN contrast pairs drifted")
    allqs=[p[s]["question"] for p in pairs for s in ("left","right")]
    if len(set(allqs))!=64 or any("TRAIN" not in q or "FPTrain" in q for q in allqs):
        raise AssertionError("S54 new TRAIN independent questions invalid")
    return pairs

def cache_eligibles(model,sources,tok,pairs,baseline):
    eligible=[]
    excluded=[]
    for p in pairs:
        sample={}
        for side in ("left","right"):
            row=p[side]
            key=f'{p["pair_id"]}_{side}'
            previous=baseline[key]
            if not previous["first_gold_correct"]:
                continue
            first=int(previous["actual"][0])
            if first!=int(row["targets"][0]):
                raise AssertionError("inconsistent S52 new TRAIN free first")
            prefix=torch.cat((row["prompt_ids"],torch.tensor([first],dtype=torch.long)))
            h,b=s52.frozen_hidden(model,sources,prefix)
            gold_noun=int(row["targets"][1])
            after_noun=torch.cat((prefix,torch.tensor([gold_noun],dtype=torch.long)))
            eos_h,eos_base=s52.frozen_hidden(model,sources,after_noun)
            sample[side]={"hidden":h,"base":b,"noun":gold_noun,
                          "eos_hidden":eos_h,"eos_base":eos_base,
                          "eos":int(tok.eos_token_id)}
        if len(sample)==2:
            eligible.append({"pair":p["pair_id"],**sample})
        else:
            excluded.append(p["pair_id"])
    return eligible,excluded

def protect_complete_baseline(model,sources,tok,new,baseline,gate,head):
    noun=[]
    eos=[]
    correct={key for key,entry in baseline.items() if entry["full_correct"]}
    if len(correct)!=29:
        raise AssertionError("S52 accepted correct 29 original full answers required")
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            if key not in correct:
                continue
            actual=baseline[key]["actual"]
            expected=baseline[key]["expected"]
            if actual!=expected or len(actual)!=3 or actual[-1]!=tok.eos_token_id:
                raise AssertionError("S52 originally correct answer must have color noun EOS")
            first,word=int(actual[0]),int(actual[1])
            prefix=torch.cat((row["prompt_ids"],torch.tensor([first],dtype=torch.long)))
            hn,bn=s52.frozen_hidden(model,sources,prefix)
            prefix2=torch.cat((prefix,torch.tensor([word],dtype=torch.long)))
            he,be=s52.frozen_hidden(model,sources,prefix2)
            for inp_h,inp_b,gold,dest in ((hn,bn,word,noun),(he,be,int(tok.eos_token_id),eos)):
                with torch.no_grad():
                    lg=s47.apply_delta(inp_b,inp_h,gate,head)
                    if int(lg.argmax())!=gold:
                        raise AssertionError("S52 saved full original noun/EOS baseline mismatched")
                dest.append((key,inp_h,inp_b,gold))
    def pack(records):
        if len(records)!=29 or len({r[0] for r in records})!=29:
            raise AssertionError("not all 29 original accepted answers protected")
        return {"hidden":torch.stack([r[1] for r in records]),
                "base":torch.stack([r[2] for r in records]),
                "target":torch.tensor([r[3] for r in records],dtype=torch.long),
                "keys":[r[0] for r in records]}
    return pack(noun),pack(eos),correct

def new_pair_objective(gate,head,eligible,rng):
    if not eligible:
        # Can't update from pairs with wrong first token while S52 gate
        # is deliberately inactive on first-token generation.
        return None
    rows=[eligible[rng.randrange(len(eligible))] for _ in range(BATCH_NEW)]
    terms=[]
    for p in rows:
        x,y=p["left"],p["right"]
        xl=s47.apply_delta(x["base"],x["hidden"],gate,head)
        yl=s47.apply_delta(y["base"],y["hidden"],gate,head)
        gl,gr=x["noun"],y["noun"]
        noun_ce=.5*(F.cross_entropy(xl[None],torch.tensor([gl]))
                      +F.cross_entropy(yl[None],torch.tensor([gr])))
        margin=F.softplus(1.6-(xl[gl]-xl[gr]))+F.softplus(1.6-(yl[gr]-yl[gl]))
        le=s47.apply_delta(x["eos_base"],x["eos_hidden"],gate,head)
        re=s47.apply_delta(y["eos_base"],y["eos_hidden"],gate,head)
        eos=.5*(F.cross_entropy(le[None],torch.tensor([x["eos"]]))
                +F.cross_entropy(re[None],torch.tensor([y["eos"]])))
        terms.append(noun_ce+2.5*margin+.4*eos)
    return torch.stack(terms).mean()

def correct_set(rows):
    return {k for k,x in rows.items() if x["full_correct"]}

def pair_both(rows,pairs):
    return sum(rows[f'{p["pair_id"]}_left']["full_correct"] and
               rows[f'{p["pair_id"]}_right']["full_correct"] for p in pairs)

def write_manifest(source,controller_file,manifest_file):
    payload=controller_file.read_bytes()
    m={"stage":"S54_REAL_PHYSICALLY_PAID_DIVERSE_TRAIN_CONTROLLER",
       "source_CB59_bytes":len(source),
       "source_CB59_sha256":hashlib.sha256(source).hexdigest(),
       "controller_parent_S52_sha256":S52_SHA,
       "controller_sha256":hashlib.sha256(payload).hexdigest(),
       "controller_bytes":len(payload),
       "controller_rank":12,
       "no_external_model_weights_or_unpaid_auxiliary":True,
       "all_physical_files_inclusive":True,
       "not_S53_S29_QA_trained_or_tuned":True}
    for _ in range(20):
        b=json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2).encode()
        total=len(source)+len(payload)+len(b)
        nxt={"manifest_bytes":len(b),"all_inclusive_total_bytes":total,
             "all_inclusive_bpp":f"{8*total/SOURCE_PARAMS:.12f}"}
        if all(m.get(k)==v for k,v in nxt.items()):
            break
        m.update(nxt)
    b=json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2).encode()
    if len(source)+len(payload)+len(b)!=m["all_inclusive_total_bytes"]:
        raise AssertionError("S54 all-inclusive paid manifest did not converge")
    if 8*m["all_inclusive_total_bytes"]/SOURCE_PARAMS>.7:
        raise AssertionError("S54 full physical weight budget violated")
    manifest_file.write_bytes(b)
    return m

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("S45_source",type=Path)
    parser.add_argument("S52_controller",type=Path)
    parser.add_argument("S52_manifest",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng=random.Random(SEED)
    source=args.S45_source.read_bytes()
    if len(source)!=SOURCE_BYTES or sha(args.S45_source)!=S45_SHA:
        raise AssertionError("S54 S45 source SHA changed")
    if args.S52_controller.stat().st_size!=S52_SIZE or sha(args.S52_controller)!=S52_SHA:
        raise AssertionError("S54 S52 physically paid neural adapter SHA changed")
    prior=json.loads(args.S52_manifest.read_text(encoding="utf-8"))
    if (args.S52_manifest.stat().st_size!=S52_MANIFEST_SIZE or
        prior["all_inclusive_total_bytes"]!=S52_PHYSICAL_TOTAL or
        prior["all_inclusive_bpp"]!=S52_BPP or
        prior["source_CB59_sha256"]!=S45_SHA or
        prior["new_controller_sha256"]!=S52_SHA):
        raise AssertionError("S54 prior accepted S52 physical payment drift")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    augmented=build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    if len(original)!=32 or len(old)!=24:
        raise AssertionError("TRAIN original composition changed")
    if {p[side]["question"] for p in original for side in ("left","right")} & {
        p[side]["question"] for p in augmented for side in ("left","right")}:
        raise AssertionError("S54 new TRAIN duplicates original")
    model,sources=create_small_paid_model(CB59(source),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    gate=s47.ActiveGreenTunnel(320,RANK)
    gate.load_state_dict(load_file(str(args.S52_controller)))
    gate.eval()
    base_new=s47.probe(model,sources,gate,head,tok,original)
    base_add=s47.probe(model,sources,gate,head,tok,augmented)
    base_old=s47.probe(model,sources,gate,head,tok,old[:4])
    old_keep=correct_set(base_old)
    original_keep=correct_set(base_new)
    first_keep={k for k,x in base_new.items() if x["first_gold_correct"]}
    if (len(old_keep)!=7 or len(original_keep)!=29 or len(first_keep)!=34 or
        pair_both(base_new,original)!=12):
        raise AssertionError("S54 S52 physically saved baseline does not reproduce accepted counts")
    initial_new_full=len(correct_set(base_add))
    initial_new_pairs=pair_both(base_add,augmented)
    anchors_noun,anchors_eos,all29=protect_complete_baseline(
        model,sources,tok,original,base_new,gate,head)
    original_cache,_=s52.eligible_pair_cache(model,sources,tok,original,base_new,None)
    if not original_cache:
        raise AssertionError("No protected original paired TRAIN cache")
    aug_cache,aug_skipped=cache_eligibles(model,sources,tok,augmented,base_add)
    cache_old,historic_guard=s47.parse_training_cache(model,sources,tok,original,old)
    if historic_guard!=old_keep:
        raise AssertionError("S54 historical skill guard changed")
    initial={"original_full64":29,"original_both_pairs32":12,"original_first64":34,
             "original_old8":7,"augmented_full64":initial_new_full,
             "augmented_both_pairs32":initial_new_pairs,
             "augmented_first64":sum(r["first_gold_correct"] for r in base_add.values()),
             "augmented_eligible_pairs":len(aug_cache),
             "augmented_excluded_wrong_first_token":len(aug_skipped)}
    print(json.dumps({"event":"S54_FROZEN_S52_BASELINE_AND_DISTINCT_TRAIN",**initial},ensure_ascii=False),flush=True)
    # Trainable SAME already-paid rank12 module; no new neural parameters.
    if sum(p.numel() for p in gate.parameters())!=8001:
        raise AssertionError("S54 controller parameter count changed")
    optimizer=torch.optim.AdamW(gate.parameters(),lr=LR,weight_decay=.0001)
    selected=None
    selected_probes=None
    selected_new_score=(initial_new_pairs,initial_new_full)
    records=[]
    if aug_cache:
        for step in range(1,STEPS+1):
            optimizer.zero_grad(set_to_none=True)
            orig_loss,orig_parts=s52.train_objective(
                gate,head,original_cache,cache_old["old"],anchors_noun,anchors_eos)
            aug_loss=new_pair_objective(gate,head,aug_cache,rng)
            objective=orig_loss+.8*aug_loss
            if not torch.isfinite(objective):
                raise AssertionError("S54 real TRAIN objective not finite")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            if step not in PROBES:continue
            gate.eval()
            old_now=s47.probe(model,sources,gate,head,tok,old[:4])
            orig_now=s47.probe(model,sources,gate,head,tok,original)
            aug_now=s47.probe(model,sources,gate,head,tok,augmented)
            original_correct=correct_set(orig_now)
            old_correct=correct_set(old_now)
            lost_orig=sorted(original_keep-original_correct)
            lost_old=sorted(old_keep-old_correct)
            lost_color=sorted(first_keep-{k for k,x in orig_now.items() if x["first_gold_correct"]})
            orig_pairs=pair_both(orig_now,original)
            new_full=len(correct_set(aug_now))
            new_pairs=pair_both(aug_now,augmented)
            # Declared in advance: do not sacrifice previously learned
            # original TRAIN answers and both-correct pairs to add a new
            # augmentation. Zero QA selection or held-out feedback.
            valid=(not lost_orig and not lost_old and not lost_color and
                   orig_pairs>=12 and new_full>initial_new_full and
                   new_pairs>initial_new_pairs)
            better=valid and (new_pairs,new_full)>selected_new_score
            if better:
                selected={k:v.detach().clone() for k,v in gate.state_dict().items()}
                selected_probes=(orig_now,aug_now,old_now)
                selected_new_score=(new_pairs,new_full)
            checkpoint={"step":step,"original_new_full64":len(original_correct),
                        "original_both_sides_pairs32":orig_pairs,
                        "new_augmentation_full64":new_full,
                        "new_augmentation_pairs_both32":new_pairs,
                        "lost_S52_original29":lost_orig,
                        "lost_S52_original_first34":lost_color,
                        "lost_old7":lost_old,
                        "strict_TRAIN_only_acceptance":better,
                        "objective":float(objective.detach()),
                        "new_diverse_pair_objective":float(aug_loss.detach()),
                        "preserved_noun_margin_29":orig_parts["protected_new17_min_global_winner_margin"],
                        "preserved_eos_margin_29":orig_parts["protected_new17_EOS_min_global_winner_margin"]}
            records.append(checkpoint)
            print(json.dumps({"event":"S54_DIVERSE_TRAIN_PROGRESS",**checkpoint},ensure_ascii=False),flush=True)
            gate.train()
    saved=False
    manifest=None
    verified=None
    cpu=None
    side_file=args.output/"S54_ACTIVE_GREEN_TUNNEL_RANK12.safetensors"
    manifest_file=args.output/"S54_ALL_INCLUSIVE_PHYSICAL_MANIFEST.json"
    if selected is not None:
        gate.load_state_dict(selected)
        gate.eval()
        save_file({k:v.detach().cpu().contiguous() for k,v in selected.items()},str(side_file))
        manifest=write_manifest(source,side_file,manifest_file)
        reloaded=s47.ActiveGreenTunnel(320,RANK)
        reloaded.load_state_dict(load_file(str(side_file)))
        reloaded.eval()
        check_new=s47.probe(model,sources,reloaded,head,tok,original)
        check_aug=s47.probe(model,sources,reloaded,head,tok,augmented)
        check_old=s47.probe(model,sources,reloaded,head,tok,old[:4])
        if (check_new!=selected_probes[0] or check_aug!=selected_probes[1] or
            check_old!=selected_probes[2]):
            raise AssertionError("S54 actual physical controller reload changed full greedy token/EOS")
        if (not original_keep<=correct_set(check_new) or
            not old_keep<=correct_set(check_old) or
            not first_keep<={k for k,v in check_new.items() if v["first_gold_correct"]} or
            pair_both(check_new,original)<12):
            raise AssertionError("S54 physical reload lost accepted S52 skill")
        # Same strictly limited four independent-of-optimizer TRAIN CPU
        # contexts for both controllers, never claim certified throughput.
        ref=s47.ActiveGreenTunnel(320,RANK)
        ref.load_state_dict(load_file(str(args.S52_controller)))
        ref.eval()
        selected_cases=[original[i][side]["question"] for i,side in
                        ((6,"left"),(16,"right"),(26,"left"),(31,"right"))]
        times={}
        for label,tunnel in (("S52",ref),("S54",reloaded)):
            measured=[]
            for prompt in selected_cases:
                st=time.perf_counter()
                s47.greedy(model,sources,tunnel,head,tok,prompt)
                measured.append(time.perf_counter()-st)
            times[label]=measured
        speed=statistics.median(times["S52"])/statistics.median(times["S54"])
        cpu={"four_TRAIN_cases_paired_wall_seconds":times,
             "median_ratio_vs_S52":speed,
             "CPU_sample_over_90_percent":speed>=.9,
             "not_independent_CPU_certification":True}
        if speed>=.9:
            saved=True
            verified={"S52_original64_exact_sequences_reloaded":True,
                      "S54_new64_exact_sequences_reloaded":True,
                      "old8_exact_sequences_reloaded":True,
                      "original_new_full":len(correct_set(check_new)),
                      "original_both_pairs":pair_both(check_new,original),
                      "new_augmented_full":len(correct_set(check_aug)),
                      "new_augmented_both_pairs":pair_both(check_aug,augmented),
                      "old_full":len(correct_set(check_old))}
        else:
            side_file.unlink()
            manifest_file.unlink()
            manifest=None
            verified=None
    output={"stage":"S54_DIVERSE_EXPLICIT_TRAIN_WITH_FULL_S52_SKILL_PROTECTION",
        "S45_source_sha256":S45_SHA,
        "S52_parent_controller_sha256":S52_SHA,
        "S53_diagnostic_CI_observed_before_new_TRAIN":42515879784,
        "S53_S29_S26_QA_never_opened_or_tuned":True,
        "S53_previous_QA_12_versus_S47_18_never_used_as_selection":True,
        "all_experiment_questions_explicit_TRAIN_only":True,
        "new_TRAIN_pairs":32,
        "new_TRAIN_frames":list(NEW_TRAIN_FRAMES),
        "new_TRAIN_owner_names":list(OWNERS),
        "original_S52_full_29_protected_both_noun_and_EOS":True,
        "original_S52_12_both_correct_pairs_hard_guard":True,
        "historical_old7_exact_hard_guard":True,
        "first_color_correct34_gated_unchanged":True,
        "unchanged_base_CB59_frozen":True,
        "same_previously_paid_rank12_controller":True,
        "initial_S52_training_answers":initial,
        "all_predeclared_training_probe_results":records,
        "candidate_physically_accepted":saved,
        "accepted_manifest":manifest,
        "accepted_physical_reload":verified,
        "paired_CPU_sample":cpu,
        "no_unseen_QA_or_FP32_parity_claim":True,
        "no_original_FP32_reference":True,
        "Jessica_chat_TEST_not_run":True,
        "LIVE_main_Render_Supabase_APK_UNCHANGED":True}
    save(args.output/"S54_DIVERSE_NEW_TRAIN_PROTECTED_S52_REPORT.json",output)
    print(json.dumps({"event":"S54_DIVERSE_NEW_TRAIN_FINAL",
        "accepted":saved,
        "new_augmented_pairs_both":verified["new_augmented_both_pairs"] if verified else initial_new_pairs,
        "new_augmented_full64":verified["new_augmented_full"] if verified else initial_new_full,
        "S52_original29_and12_pairs_retained":verified is not None,
        "old7_retained":verified is not None,
        "physical_bpp":manifest["all_inclusive_bpp"] if manifest else None,
        "CPU_sample_ratio":cpu["median_ratio_vs_S52"] if cpu else None,
        "new_controller_exported":saved},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
