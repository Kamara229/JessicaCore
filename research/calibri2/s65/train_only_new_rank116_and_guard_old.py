"""CALIBRI S65: train only newly paid rank-128 Green Tunnel columns.

Research-only S56 baseline remains immutable; first-rank2, original
after-first rank12, every 8D CB59 weight, and real S61 lossless source remain
frozen. Train added 116 after-first columns on TRAIN only with old-skill,
opposition and EOS replay. Evaluate native greedy answers, NOT mere NLL.

Accepted on TRAIN only means eligible for *future blind independent QA*,
never promotion/quality parity. CPU-only; GPU is not a prerequisite.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
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
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from v186.pairs import build_pairs

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S56_SHA="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
S64_CONTROLLER_SHA="978d6b1962870cafe45fcda5a3ce64911e651f3e3daf2385cbbb1552cdb114c8"
RANK_FIRST=2
RANK_LATER=128
PAID_FIXED_ORIGINAL_RANK=12
SEED=65116
STEPS=16
PROBES=(8,16)
LR=.0007

def sha(raw):return hashlib.sha256(raw).hexdigest()

def sidecar(first,later,output):
    tensors={"first."+k:v.detach().cpu().contiguous()
             for k,v in first.state_dict().items()}
    tensors.update({"later."+k:v.detach().cpu().contiguous()
                   for k,v in later.state_dict().items()})
    save_file(tensors,str(output))

def saved_physical_manifest(compressed,controller,output):
    """Pay physical CB61 source + expanded rank128 controller + manifest."""
    info={
        "stage":"CALIBRI_S65_RANK128_NEW_COLUMNS_TRAIN_ONLY_RESEARCH",
        "unchanged_original_S45_sha256":S45_SHA,
        "unchanged_original_S56_controller_sha256":S56_SHA,
        "unchanged_initial_S64_controller_sha256":S64_CONTROLLER_SHA,
        "physical_lossless_CB61_source_bytes":len(compressed),
        "physical_lossless_CB61_source_sha256":sha(compressed),
        "physically_paid_rank128_controller_bytes":len(controller),
        "physically_paid_rank128_controller_sha256":sha(controller),
        "original_FP32_after_first_rank":12,
        "new_after_first_rank":128,
        "only_new_116_up_columns_trained":True,
        "original_after_first_gate_and_first_token_tunnel_frozen":True,
        "CPU_only_inference_compatible":True,
        "requires_new_sealed_heldout_QA_and_real_CPU_speed_before_promotion":True,
        "not_a_generalization_or_FP32_parity_claim":True,
        "no_QA_TEST_DEV_S26_S29_S53_S55_S57_used":True,
        "no_LIVE_Render_Supabase_APK_modified":True,
        "unique_neural_source_parameters":SOURCE_PARAMS,
    }
    for _ in range(40):
        b=json.dumps(info,indent=2,sort_keys=True).encode()
        revision={
            "manifest_bytes":len(b),
            "all_inclusive_bytes":len(compressed)+len(controller)+len(b),
            "all_inclusive_bpp":f"{8*(len(compressed)+len(controller)+len(b))/SOURCE_PARAMS:.12f}",
        }
        if all(info.get(k)==v for k,v in revision.items()):break
        info.update(revision)
    b=json.dumps(info,indent=2,sort_keys=True).encode()
    if info["all_inclusive_bytes"] != len(compressed)+len(controller)+len(b):
        raise ValueError("nonconvergent complete physical sidecar accounting")
    if info["all_inclusive_bytes"]*8/SOURCE_PARAMS>.7:
        raise ValueError("TRAIN candidate cannot fit physical <=0.7 bpp")
    output.write_bytes(b)
    return info

def restore_new_column_only(later,original_up):
    """AdamW weight decay=0 + gradient hook: old physical bytes stay exact."""
    later.eval().requires_grad_(False)
    old=original_up.detach().clone()
    later.up.weight.requires_grad_(True)
    def protect(grad):
        g=grad.clone()
        g[:,:PAID_FIXED_ORIGINAL_RANK]=0
        return g
    later.up.weight.register_hook(protect)
    return old

def check_frozen(later,original_up,original_down,original_gate,original_bias):
    if not torch.equal(later.up.weight[:,:PAID_FIXED_ORIGINAL_RANK].detach(),original_up):
        raise AssertionError("protected original rank12 up bytes changed")
    if not torch.equal(later.down.weight.detach(),original_down):
        raise AssertionError("protected original/new 128 down weights changed")
    if not torch.equal(later.gate.weight.detach(),original_gate):
        raise AssertionError("protected gate weight changed")
    if not torch.equal(later.gate.bias.detach(),original_bias):
        raise AssertionError("protected gate bias changed")

def eligibility(orig,aug,old,original,augmented,kept):
    now=(s58.correct(orig),s58.correct(aug),s58.correct(old),
         s58.first_correct(orig),s58.first_correct(aug))
    lost=[sorted(a-b) for a,b in zip(kept,now)]
    pa=s58.both_correct(orig,original)
    pb=s58.both_correct(aug,augmented)
    preserve=(all(not arr for arr in lost) and pa>=13 and pb>=7)
    improved=preserve and pa+pb>20 and len(now[0])+len(now[1])>=69
    return {"original_full":len(now[0]),"augmented_full":len(now[1]),
            "historical_full":len(now[2]),"original_first":len(now[3]),
            "augmented_first":len(now[4]),
            "opposed_original_pairs_both":pa,"opposed_augmented_pairs_both":pb,
            "lost_identity_counts":[len(arr) for arr in lost],
            "baseline_all_correct_answers_preserved":preserve,
            "TRAIN_candidate_eligible_for_future_independent_QA":improved}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("immutable_S45_CB59",type=Path)
    p.add_argument("immutable_S56_dual",type=Path)
    p.add_argument("S64_rank128_initial",type=Path)
    p.add_argument("S61_lossless_source_CB61",type=Path)
    p.add_argument("S64_manifest",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED);torch.set_num_threads(2)
    rng=random.Random(SEED)
    source=a.immutable_S45_CB59.read_bytes()
    oldcontroller=a.immutable_S56_dual.read_bytes()
    newcontroller=a.S64_rank128_initial.read_bytes()
    packed=a.S61_lossless_source_CB61.read_bytes()
    physical=json.loads(a.S64_manifest.read_text())
    if (len(source)!=SOURCE_BYTES or sha(source)!=S45_SHA or
        sha(oldcontroller)!=S56_SHA or sha(newcontroller)!=S64_CONTROLLER_SHA or
        physical["controller_sha256"]!=S64_CONTROLLER_SHA or
        physical["compressed_source_sha256"]!=sha(packed) or
        float(physical["all_inclusive_bpp"])>.7):
        raise ValueError("immutable physical model and paid S64 bpp provenance drift")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    added=s54.build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(original)==32 and len(added)==32 and len(old)==24
    model,sources=create_small_paid_model(CB59(source),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    state=load_file(str(a.S64_rank128_initial))
    earlier=load_file(str(a.immutable_S56_dual))
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,RANK_LATER)
    first.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    for k,v in earlier.items():
        if k=="later.up.weight":
            if not torch.equal(v,later.up.weight[:,:PAID_FIXED_ORIGINAL_RANK]):raise AssertionError("original later up differs")
        elif k=="later.down.weight":
            if not torch.equal(v,later.down.weight[:PAID_FIXED_ORIGINAL_RANK]):raise AssertionError("original later down differs")
        elif k.startswith("later."):
            if not torch.equal(v,state[k]):raise AssertionError("original after first gate differs")
        elif not torch.equal(v,state[k]):
            raise AssertionError("original first rank2 differs")
    first_state={k:v.detach().clone() for k,v in first.state_dict().items()}
    baseline_orig=s56.probe(model,sources,first,later,head,tok,original)
    baseline_add=s56.probe(model,sources,first,later,head,tok,added)
    baseline_old=s56.probe(model,sources,first,later,head,tok,old[:4])
    start={
        "original_full":len(s58.correct(baseline_orig)),
        "augmented_full":len(s58.correct(baseline_add)),
        "historical_full":len(s58.correct(baseline_old)),
        "original_first":len(s58.first_correct(baseline_orig)),
        "augmented_first":len(s58.first_correct(baseline_add)),
        "opposed_original_pairs_both":s58.both_correct(baseline_orig,original),
        "opposed_augmented_pairs_both":s58.both_correct(baseline_add,added)}
    if start!={"original_full":42,"augmented_full":27,"historical_full":7,
               "original_first":58,"augmented_first":46,
               "opposed_original_pairs_both":13,"opposed_augmented_pairs_both":7}:
        raise AssertionError("S64 rank128 changed S56 real full TRAIN response baseline")
    keep=(s58.correct(baseline_orig),s58.correct(baseline_add),
          s58.correct(baseline_old),s58.first_correct(baseline_orig),
          s58.first_correct(baseline_add))
    ocache,_=s58.eligible_real_first_contexts(model,sources,tok,original,baseline_orig)
    acache,_=s58.eligible_real_first_contexts(model,sources,tok,added,baseline_add)
    nouno,eoso,_=s58.full_answer_replay(model,sources,tok,original,baseline_orig,42)
    nouna,eosa,_=s58.full_answer_replay(model,sources,tok,added,baseline_add,27)
    oldcache,protected=s47.parse_training_cache(model,sources,tok,original,old)
    if protected!=keep[2]:raise AssertionError("old seven skill guard drift")
    if not ocache or not acache:raise AssertionError("no eligible opposing pairs in TRAIN")
    oldup=later.up.weight[:,:PAID_FIXED_ORIGINAL_RANK].detach().clone()
    olddown=later.down.weight.detach().clone()
    oldgate=later.gate.weight.detach().clone()
    oldbias=later.gate.bias.detach().clone()
    restore_new_column_only(later,oldup)
    optimizer=torch.optim.AdamW([later.up.weight],lr=LR,weight_decay=0.0)
    print(json.dumps({"event":"S65_BASELINE_NATIVE_GREEDY_TRAIN_ONLY",
       "quality":start,"original_rank12_unchanged":True,
       "trained_FP32_parameters_only":320*(128-12),
       "CPU_GPU_policy":"CPU-only runnable, no GPU allocation needed"}),flush=True)
    attempts=[];best_key=(20,69);best=None
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        contrast=s58.new_paired_loss(later,head,ocache,rng)
        addcontrast=s58.new_paired_loss(later,head,acache,rng)
        ol,_,_=s52.protected_new_global_winner_replay(later,head,nouno)
        oe,_,_=s52.protected_new_global_winner_replay(later,head,eoso)
        al,_,_=s52.protected_new_global_winner_replay(later,head,nouna)
        ae,_,_=s52.protected_new_global_winner_replay(later,head,eosa)
        historical=s58.old_historical_replay(later,head,oldcache)
        loss=contrast+addcontrast+7*(ol+al)+8*(oe+ae)+2*historical
        if not torch.isfinite(loss):raise ValueError("non-finite S65 loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([later.up.weight],max_norm=5.,error_if_nonfinite=True)
        optimizer.step()
        check_frozen(later,oldup,olddown,oldgate,oldbias)
        if step not in PROBES:continue
        actual_orig=s56.probe(model,sources,first,later,head,tok,original)
        actual_add=s56.probe(model,sources,first,later,head,tok,added)
        actual_old=s56.probe(model,sources,first,later,head,tok,old[:4])
        candidate=eligibility(actual_orig,actual_add,actual_old,original,added,keep)
        candidate.update({"step":step,"TRAIN_loss_proxy_not_a_quality_score":float(loss.detach())})
        attempts.append(candidate)
        if candidate["TRAIN_candidate_eligible_for_future_independent_QA"]:
            key=(candidate["opposed_original_pairs_both"]+candidate["opposed_augmented_pairs_both"],
                 candidate["original_full"]+candidate["augmented_full"])
            if key>best_key:
                best_key=key
                best={k:v.detach().clone() for k,v in later.state_dict().items()}
        torch.save({"step":step,"immutable_original_sha256":S45_SHA,
             "immutable_S64_sidecar_sha256":S64_CONTROLLER_SHA,
             "later":{k:v.detach().cpu() for k,v in later.state_dict().items()},
             "optimizer":optimizer.state_dict(),
             "randomizer_state":rng.getstate(),
             "for_CPU_resume_only_not_paid_inference_weights":True},
             a.out/f"S65_STEP_{step:03d}_TRAIN_RESUME.pt")
        print(json.dumps({"event":"S65_ACTUAL_NATIVE_TRAIN_PROBE",**candidate}),flush=True)
    eligible=best is not None
    trial=a.out/"S65_EXPERIMENT_TRAIN_ONLY_RESULT.json"
    report={
        "stage":"S65_TRAIN_ONLY_NEW_116_COLUMNS_OF_PAID_128_GREEN_TUNNEL",
        "baseline":start,"probes":attempts,
        "accepted_solely_as_train_candidate_for_future_blind_QA":eligible,
        "FP32_or_unseen_quality_proved":False,
        "all_8D_compressed_CB59_neural_weights_unchanged":True,
        "rank2_first_and_original_12_after_first_parameters_frozen":True,
        "S61_CB61_physical_source_frozen":True,
        "no_GPU_Jessica_CHAT_Render_Supabase_APK_changes":True}
    if eligible:
        later.load_state_dict(best)
        side=a.out/"S65_RANK128_TRAIN_CANDIDATE_ONLY.safetensors"
        sidecar(first,later,side)
        paid=saved_physical_manifest(packed,side.read_bytes(),
                        a.out/"S65_PHYSICALLY_PAID_TRAIN_CANDIDATE_MANIFEST.json")
        report["candidate_controller_sha256"]=sha(side.read_bytes())
        report["all_inclusive_bpp"]=paid["all_inclusive_bpp"]
        report["still_not_promoted_until_fresh_heldout_QA_and_CPU_speed"]=True
    trial.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S65_TRAIN_ONLY_FINAL_RESULT",
       "eligible_for_future_blind_QA":eligible,
       "result":attempts[-1],"result_report":trial.name}),flush=True)
    if any(not torch.equal(first.state_dict()[k],v) for k,v in first_state.items()):
        raise AssertionError("frozen S56 first gate changed")

if __name__=="__main__":main()
