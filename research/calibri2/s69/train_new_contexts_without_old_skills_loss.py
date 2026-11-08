"""CALIBRI S69: NEW TRAIN-only context curriculum with immutable S67 guard.

S68 one-time independent QA was inspected and is now permanently embargoed.
NO S68, S29, S26 QA, S53, S55 or S57 modules, prompts, records, scores,
model checkpoints, correction signals or gradients are loaded. New TRAIN
prompts below are fresh, all start TRAIN, use only S30 TRAIN-eligible
color/object combinations, and cannot reuse exact embargoed EVAL data.

This is *not* a scientific claim of generalization; the exact free-answer
performance on a separate future independent QA and FP32 parity remain
unknown. We refuse any S67 regression, even if new TRAIN improves.

Source: frozen true 8D/CB59 S45, physical CB61 S61. Paid neural tunnel:
S67 rank2 first/color + rank128 later; train only the 116 added later
up-projection columns, no original 12/gate/down/core/first changes.
CPU-only, bounded, resumable; no GPU requirement or Jessica chat test.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES, MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s52 import sequence_eos_protected_green_tunnel as s52
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from v186.pairs import tokenize_one,build_pairs

CANDIDATE_SHA="6698b8664b8e4391a669defe0316d076f2c296a2bb8c502f4dd53e8b028ac572"
SEED=690069
STEPS=12
PROBES=(4,8,12)
LR=.00035
# Only TRAIN-eligible pairs. Deliberately do not train S30 withheld QA.
COLORS=("red","green","blue","yellow")
TRAIN_NAMES=(
    ("Huno","Velton"),("Isami","Marek"),("Farla","Tinor"),
    ("Jostin","Beleni"))
# New training frames, with fixed provenance and no S68 vocab or scripts.
TRAIN_FRAMES=(
    "TRAIN route 6901: The archive shelf for {a} is coded {left}. For {b} the shelf code is {right}. Write the TWO tokens assigned to {a}.",
    "TRAIN routing 6902: In a shipping manifest, customer {a} selects {left}; customer {b} selects {right}. Output only {a}'s two-word selection.",
    "TRAIN mapping 6903: The notation table binds {a} to {left} and {b} to {right}. Report the ordered color/object pair for {a}.",
    "TRAIN encoding 6904: Row {a} => {left}; row {b} => {right}. Reconstruct {a}'s color plus noun without extra words."
)
assert len(TRAIN_NAMES)==len(TRAIN_FRAMES)==4
assert set(COLORS)==set(s30.TRAIN_ELIGIBLE)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def build_new_train(tok):
    pairs=[]
    for ci,color in enumerate(COLORS):
        nouns=s30.TRAIN_ELIGIBLE[color]
        if len(nouns)!=2:raise AssertionError("new TRAIN attempted withheld object pair")
        x,y=[f"{color} {n}" for n in nouns]
        for j,(a,b) in enumerate(TRAIN_NAMES):
            prompt=TRAIN_FRAMES[j]
            q0=prompt.format(a=a,b=b,left=x,right=y)
            q1=prompt.format(a=a,b=b,left=y,right=x)
            if "TRAIN" not in q0 or "EVAL-ONLY" in q0 or q0==q1:
                raise AssertionError("new TRAIN prompt provenance invalid")
            l=tokenize_one(tok,q0,x)
            r=tokenize_one(tok,q1,y)
            if l["first_expected_id"]!=r["first_expected_id"]:
                raise AssertionError("TRAIN opposition must share same first color token")
            pairs.append({"pair_id":69000+len(pairs),"kind":"fresh_TRAIN_only",
                          "left":l,"right":r,"color":color})
    if len(pairs)!=16 or len({p[s]["question"] for p in pairs for s in ("left","right")})!=32:
        raise AssertionError("fresh TRAIN matrix malformed or duplicated")
    return pairs

def take_probe(model,sources,first,later,head,tok,orig,aug,history,fresh):
    return {
        "original":s56.probe(model,sources,first,later,head,tok,orig),
        "augmented":s56.probe(model,sources,first,later,head,tok,aug),
        "historical":s56.probe(model,sources,first,later,head,tok,history[:4]),
        "fresh_TRAIN":s56.probe(model,sources,first,later,head,tok,fresh),
    }

def quality(rows,orig,aug,fresh,protected):
    kept=(
        s58.correct(rows["original"]),s58.correct(rows["augmented"]),
        s58.correct(rows["historical"]),s58.first_correct(rows["original"]),
        s58.first_correct(rows["augmented"]),
    )
    lost=[sorted(prev-new) for prev,new in zip(protected,kept)]
    both_orig=s58.both_correct(rows["original"],orig)
    both_aug=s58.both_correct(rows["augmented"],aug)
    return {
        "S30_original_full":len(kept[0]),"S54_augmented_full":len(kept[1]),
        "historical_full":len(kept[2]),
        "S30_original_first":len(kept[3]),"S54_augmented_first":len(kept[4]),
        "old_original_pairs_both":both_orig,"old_augmented_pairs_both":both_aug,
        "new_fresh_TRAIN_full":len(s58.correct(rows["fresh_TRAIN"])),
        "new_fresh_TRAIN_first":len(s58.first_correct(rows["fresh_TRAIN"])),
        "new_fresh_TRAIN_both_pairs":s58.both_correct(rows["fresh_TRAIN"],fresh),
        "lost_previous_correct_identities": {
           "S30":lost[0],"S54":lost[1],"history":lost[2],
           "S30_first":lost[3],"S54_first":lost[4]
        },
        "all_previously_correct_S67_identities_preserved":all(not x for x in lost),
        "S67_original_20_opposition_pairs_preserved":both_orig>=13 and both_aug>=7,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("immutable_S45_CB59",type=Path)
    p.add_argument("S67_frozen_rank128_controller",type=Path)
    p.add_argument("S67_physical_manifest",type=Path)
    p.add_argument("S61_exact_lossless_CB61_source",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    rng=random.Random(SEED)
    raw=a.immutable_S45_CB59.read_bytes()
    controller=a.S67_frozen_rank128_controller.read_bytes()
    cb61=a.S61_exact_lossless_CB61_source.read_bytes()
    m=json.loads(a.S67_physical_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        sha(controller)!=CANDIDATE_SHA or
        m["candidate_S67_rank128_controller_sha256"]!=CANDIDATE_SHA or
        m["physical_CB61_source_sha256"]!=sha(cb61) or
        float(m["all_inclusive_bpp"])>.7):
        raise AssertionError("S69 fails to identify sealed S67 source/controller")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    orig=s30.make_training_pairs(tok)
    aug=s54.build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    fresh=build_new_train(tok)
    if len(orig)!=32 or len(aug)!=32 or len(old)!=24 or len(fresh)!=16:
        raise AssertionError("old or fresh TRAIN sizes changed")
    # Store exact TRAIN task provenance before any fitting. This is not QA.
    train_hash=sha(json.dumps([
        {"pair_id":p["pair_id"],"left":p["left"]["question"],
         "right":p["right"]["question"]}
        for p in fresh],sort_keys=True).encode())
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    weights=load_file(str(a.S67_frozen_rank128_controller))
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    first.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    first_frozen={k:v.detach().clone() for k,v in first.state_dict().items()}
    before=take_probe(model,sources,first,later,head,tok,orig,aug,old,fresh)
    protected=(s58.correct(before["original"]),s58.correct(before["augmented"]),
               s58.correct(before["historical"]),s58.first_correct(before["original"]),
               s58.first_correct(before["augmented"]))
    initial=quality(before,orig,aug,fresh,protected)
    if (
        initial["S30_original_full"],initial["S54_augmented_full"],initial["historical_full"],
        initial["S30_original_first"],initial["S54_augmented_first"],
        initial["old_original_pairs_both"],initial["old_augmented_pairs_both"]
    ) != (42,28,7,58,46,13,7):
        raise AssertionError("S67 TRAIN protection baseline drift")
    # Before fitting, compute the native FIRST-correct contexts for real
    # teacher-forced second token/EOS; do not alter first-token gate.
    original_cache,_=s58.eligible_real_first_contexts(model,sources,tok,orig,before["original"])
    aug_cache,_=s58.eligible_real_first_contexts(model,sources,tok,aug,before["augmented"])
    fresh_cache,excluded=s58.eligible_real_first_contexts(model,sources,tok,fresh,before["fresh_TRAIN"])
    if not fresh_cache:
        report={"stage":"S69_FIRST_TOKEN_BLOCKS_NEW_CONTEXT_CURRICULUM",
                "initial":initial,"eligible_fresh_both_first_correct_pairs":0,
                "scientific_result":"NO_ELIGIBLE_FIRST_CORRECT_NEW_TRAIN_PAIRS",
                "quality_claim":False,
                "no_test_in_Jessica_or_LIVE":True}
        (a.output_dir/"S69_REAL_TRAIN_RESULT.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return
    noun_orig,eos_orig,_=s58.full_answer_replay(model,sources,tok,orig,before["original"],42)
    noun_aug,eos_aug,_=s58.full_answer_replay(model,sources,tok,aug,before["augmented"],28)
    historical_cache,old7=s47.parse_training_cache(model,sources,tok,orig,old)
    if old7!=protected[2]:
        raise AssertionError("historical seven correct identities do not match")
    frozen_up=later.up.weight[:,:12].detach().clone()
    frozen_down=later.down.weight.detach().clone()
    frozen_gate=later.gate.weight.detach().clone()
    frozen_bias=later.gate.bias.detach().clone()
    s65.restore_new_column_only(later,frozen_up)
    # All grad flows through ONLY 116 new after-first output columns.
    optimizer=torch.optim.AdamW([later.up.weight],lr=LR,weight_decay=0.0)
    steps=[]
    best=None
    best_key=(initial["new_fresh_TRAIN_full"],initial["new_fresh_TRAIN_both_pairs"])
    print(json.dumps({"event":"S69_INITIAL_NEW_CONTEXT_TRAIN_ONLY",
        "fresh_TRAIN_sha256":train_hash,
        "new_TRAIN_pair_count":16,
        "fresh_eligible_first_correct_contrast_pairs":len(fresh_cache),
        "fresh_excluded_first_wrong_pairs":len(excluded),
        "initial":initial,
        "trainable_only_new_later_up_FP32_parameters":320*116,
        "full_original_S67_37M_CB59_weight_bytes_untouched":True
        }),flush=True)
    for step in range(1,STEPS+1):
        later.eval()
        optimizer.zero_grad(set_to_none=True)
        # New contexts + supported old questions jointly supervise actual
        # TRAIN gold nouns/EOS; each old skill is protected by full-greedy QA.
        new_loss=s58.new_paired_loss(later,head,fresh_cache,rng)
        orig_loss=s58.new_paired_loss(later,head,original_cache,rng)
        aug_loss=s58.new_paired_loss(later,head,aug_cache,rng)
        original_noun,_,_=s52.protected_new_global_winner_replay(later,head,noun_orig)
        original_eos,_,_=s52.protected_new_global_winner_replay(later,head,eos_orig)
        augment_noun,_,_=s52.protected_new_global_winner_replay(later,head,noun_aug)
        augment_eos,_,_=s52.protected_new_global_winner_replay(later,head,eos_aug)
        historical_loss=s58.old_historical_replay(later,head,historical_cache)
        loss=3*new_loss + orig_loss + aug_loss + \
            10*(original_noun+augment_noun) + \
            12*(original_eos+augment_eos) + 2*historical_loss
        if not torch.isfinite(loss):
            raise AssertionError("S69 backpropagated loss not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([later.up.weight],max_norm=5.,error_if_nonfinite=True)
        optimizer.step()
        s65.check_frozen(later,frozen_up,frozen_down,frozen_gate,frozen_bias)
        if step not in PROBES:continue
        current=take_probe(model,sources,first,later,head,tok,orig,aug,old,fresh)
        metrics=quality(current,orig,aug,fresh,protected)
        preserved=(metrics["all_previously_correct_S67_identities_preserved"] and
                   metrics["S67_original_20_opposition_pairs_preserved"] and
                   metrics["new_fresh_TRAIN_first"] >= initial["new_fresh_TRAIN_first"])
        better=(preserved and
                metrics["new_fresh_TRAIN_full"]>initial["new_fresh_TRAIN_full"] and
                (metrics["new_fresh_TRAIN_full"],metrics["new_fresh_TRAIN_both_pairs"])>best_key)
        if better:
            best_key=(metrics["new_fresh_TRAIN_full"],metrics["new_fresh_TRAIN_both_pairs"])
            best={k:v.detach().clone() for k,v in later.state_dict().items()}
        record={"step":step,"train_gradient_proxy_loss_not_quality":float(loss.detach()),
                "previous_skill_preservation_pass":preserved,
                "new_TRAIN_improvement_candidate":better,**metrics}
        steps.append(record)
        print(json.dumps({"event":"S69_PROBE_OLD_PROTECTED_NEW_TRAIN_GREEDY",**record}),flush=True)
        torch.save({"step":step,"original_frozen_S67_controller_sha256":CANDIDATE_SHA,
             "fresh_TRAIN_sha256":train_hash,
             "later_state":{k:v.detach().cpu() for k,v in later.state_dict().items()},
             "optimizer":optimizer.state_dict(),
             "randomizer_state":rng.getstate(),
             "only_for_restarting_TRAIN_not_inference_weights":True},
             a.output_dir/f"S69_CPU_TRAIN_RESUME_STEP_{step:03d}.pt")
    result={
        "stage":"CALIBRI_S69_FRESH_CONTEXT_TRAIN_ONLY_116_NEW_UP_COLUMNS",
        "initial":initial,"actual_native_greedy_checkpoints":steps,
        "fresh_TRAIN_pairs":16,"fresh_TRAIN_prompt_SHA256":train_hash,
        "selected_train_candidate":best is not None,
        "old_S67_CB59_and_first_rank2_and_later_rank12_frozen":True,
        "no_independent_QA_loaded_or_used":True,
        "S68_QA_permanently_embargoed_not_imported":True,
        "future_independent_heldout_QA_and_CPU_speed_required":True,
        "no_FP32_parity_claim":True,
        "no_GPU_Jessica_chat_tests_or_LIVE_changes":True,
    }
    if best is not None:
        later.load_state_dict(best)
        side=a.output_dir/"S69_NEW_CONTEXT_TRAIN_CANDIDATE_ONLY.safetensors"
        s65.sidecar(first,later,side)
        physically_paid=s65.saved_physical_manifest(cb61,side.read_bytes(),
                               a.output_dir/"S69_PHYSICALLY_PAID_TRAIN_MANIFEST.json")
        result["physically_paid_all_inclusive_bpp"]=physically_paid["all_inclusive_bpp"]
        result["candidate_controller_SHA256"]=sha(side.read_bytes())
        result["needs_real_CPU_speed_and_brand_new_blind_QA_before_promotion"]=True
    if any(not torch.equal(first.state_dict()[k],v) for k,v in first_frozen.items()):
        raise AssertionError("S69 changed frozen first token gate")
    (a.output_dir/"S69_REAL_TRAIN_RESULT.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"event":"S69_FINAL_TRAIN_ONLY",
         "accepted_OLD_SKILL_SAFE_new_TRAIN_candidate":best is not None,
         "initial_fresh_full":initial["new_fresh_TRAIN_full"],
         "last_fresh_full":steps[-1]["new_fresh_TRAIN_full"],
         "physically_paid_bpp":result.get("physically_paid_all_inclusive_bpp"),
         "no_generalization_or_FP32_parity_claim":True}),flush=True)

if __name__=="__main__":
    main()
