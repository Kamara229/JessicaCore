"""CALIBRI S67: physically pay and verify S66's best safe half-step.

Pre-registered S67 exploratory rule, DIFFERENT from S66's strict >20
two-sided pair target (which S66 did not meet): preserve ALL 42 original,
27 augmented, seven historical full answers; gain >=1 new augmented full
answer; preserve all first-token skills and >=20 originally solved opposed
pairs. No claim of a quality leap, independent generalization, or success
under S66's original acceptance rule. S67's checkpoint is a candidate for
FRESH blind QA only, never promoted to LIVE.

S45 CB59, first-rank2, old later rank12, tokenizer, Green Tunnel gate,
and source CB61 exact physical compressed source remain immutable.
GPU optional, CPU-only path demonstrably works; Jessica chat not used.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS, SOURCE_BYTES, MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s66 import train_only_line_search_s65_delta as s66
from v186.pairs import build_pairs

ALPHA=0.5
MIN_FULL_GAIN=1
MIN_OPPOSING_PAIR_PRESERVED=20

def sha(x): return hashlib.sha256(x).hexdigest()

def paid_manifest(compressed,controller,out):
    metadata={
        "stage":"CALIBRI_S67_EXPLORE_PROTECTED_HALF_DELTA_CANDIDATE_NOT_S66_SUCCESS",
        "source_codec":"CB61ZL01 physically paid lossless exact CB59",
        "original_S45_CB59_sha256":s65.S45_SHA,
        "physical_CB61_source_sha256":sha(compressed),
        "physical_CB61_source_bytes":len(compressed),
        "original_S56_rank12_controller_sha256":s65.S56_SHA,
        "initial_S64_rank128_controller_sha256":s65.S64_CONTROLLER_SHA,
        "candidate_S67_rank128_controller_sha256":sha(controller),
        "candidate_S67_rank128_controller_bytes":len(controller),
        "after_first_rank":128,
        "new_up_columns":116,
        "selected_rejected_S65_step16_interpolation_coefficient":ALPHA,
        "frozen_all_original_rank12_and_first_rank2_FP32_weights":True,
        "S66_original_strict_opposing_pair_improvement_target_failed":True,
        "S67_exploratory_acceptance_is_not_prior_S66_success":True,
        "no_sealed_QA_or_FP32_quality_or_GPU_used":True,
        "unseen_quality_and_real_CPU_speed_must_be_verified_before_promotion":True,
        "no_LIVE_main_Render_Supabase_APK_Jessica_changes":True,
        "original_neural_source_parameters":SOURCE_PARAMS,
    }
    for _ in range(32):
        b=json.dumps(metadata,sort_keys=True,indent=2).encode()
        total=len(compressed)+len(controller)+len(b)
        revision={"physical_manifest_bytes":len(b),
                  "all_inclusive_bytes":total,
                  "all_inclusive_bpp":f"{total*8/SOURCE_PARAMS:.12f}"}
        if all(metadata.get(k)==v for k,v in revision.items()):break
        metadata.update(revision)
    b=json.dumps(metadata,sort_keys=True,indent=2).encode()
    if len(b)!=metadata["physical_manifest_bytes"] or \
       len(compressed)+len(controller)+len(b)!=metadata["all_inclusive_bytes"]:
        raise ValueError("non-convergent physical manifest")
    if metadata["all_inclusive_bytes"]*8/SOURCE_PARAMS>.7:
        raise ValueError("candidate exceeds physical .7bpp ceiling")
    out.write_bytes(b)
    return metadata

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_original_CB59",type=Path)
    p.add_argument("S56_immutable_controller",type=Path)
    p.add_argument("S64_initial_rank128",type=Path)
    p.add_argument("S61_exact_lossless_source",type=Path)
    p.add_argument("S64_physical_manifest",type=Path)
    p.add_argument("S65_step16_rejected_resume",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(670067)

    src=a.S45_original_CB59.read_bytes()
    immutable=a.S56_immutable_controller.read_bytes()
    initial=a.S64_initial_rank128.read_bytes()
    paid_source=a.S61_exact_lossless_source.read_bytes()
    m64=json.loads(a.S64_physical_manifest.read_text())
    if (len(src)!=SOURCE_BYTES or sha(src)!=s65.S45_SHA or
        sha(immutable)!=s65.S56_SHA or sha(initial)!=s65.S64_CONTROLLER_SHA or
        m64["compressed_source_sha256"]!=sha(paid_source) or
        m64["controller_sha256"]!=sha(initial)):
        raise AssertionError("immutable source/controller and physical budget drift")
    ck=torch.load(str(a.S65_step16_rejected_resume),map_location="cpu",weights_only=True)
    if ck.get("step")!=16 or ck.get("immutable_original_sha256")!=s65.S45_SHA or \
       ck.get("immutable_S64_sidecar_sha256")!=s65.S64_CONTROLLER_SHA:
        raise AssertionError("S65 rejected delta checkpoint provenance drift")

    initial_state=load_file(str(a.S64_initial_rank128))
    original_state=load_file(str(a.S56_immutable_controller))
    new_state=ck["later"]
    for k in ("down.weight","gate.weight","gate.bias"):
        if not torch.equal(new_state[k],initial_state["later."+k]):
            raise AssertionError("training unexpectedly modified protected later "+k)
    old_up=initial_state["later.up.weight"]
    train_up=new_state["up.weight"]
    if not torch.equal(train_up[:,:12],old_up[:,:12]):
        raise AssertionError("new training violated immutable 12 original columns")
    if not torch.equal(original_state["later.up.weight"],old_up[:,:12]):
        raise AssertionError("initial rank128 does not reproduce immutable S56")
    if not torch.equal(original_state["later.down.weight"],initial_state["later.down.weight"][:12]):
        raise AssertionError("old down rows drift")
    for k in ("first.down.weight","first.up.weight","first.gate.weight","first.gate.bias",
              "later.gate.weight","later.gate.bias"):
        if not torch.equal(initial_state[k],original_state[k]):
            raise AssertionError("previous S56 first/gate drift")
    interpolated_up=old_up+ALPHA*(train_up-old_up)
    if torch.count_nonzero(interpolated_up-old_up).item()==0:
        raise AssertionError("candidate has no new FP32 weights")
    if not torch.equal(interpolated_up[:,:12],old_up[:,:12]):
        raise AssertionError("protected original old rank12 changed")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    orig=s30.make_training_pairs(tok)
    aug=s54.build_new_train(tok)
    history=build_pairs(tok,"TRAIN")
    if (len(orig),len(aug),len(history))!=(32,32,24):
        raise AssertionError("TRAIN provenance drift")
    model,sources=create_small_paid_model(CB59(src),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    first.load_state_dict({k[6:]:v for k,v in initial_state.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in initial_state.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)

    baseline=s66.snapshot(model,sources,first,later,head,tok,orig,aug,history)
    old_ids=(s58.correct(baseline[0]),s58.correct(baseline[1]),
             s58.correct(baseline[2]),s58.first_correct(baseline[0]),
             s58.first_correct(baseline[1]))
    base=s66.stats(baseline,orig,aug,old_ids)
    if not (base["original_full"]==42 and base["augmented_full"]==27 and
            base["historical_full"]==7 and base["original_first"]==58 and
            base["augmented_first"]==46 and
            base["original_opposed_pairs_both"]==13 and
            base["augmented_opposed_pairs_both"]==7):
        raise AssertionError("immutable S56 full native baseline mismatch")

    # Real 16-prompt native CPU timing, both using the same rank128 topology.
    samples=make_samples(tok)
    speed={"baseline":[],"candidate":[]}
    saved_unmodified=old_up.detach().clone()
    for i,(_,_,question,_) in enumerate(samples):
        order=("baseline","candidate") if i%2==0 else ("candidate","baseline")
        for stage in order:
            with torch.no_grad():
                later.up.weight.copy_(saved_unmodified if stage=="baseline" else interpolated_up)
            before=time.perf_counter()
            s56.greedy_first_active(model,sources,first,later,head,tok,question)
            speed[stage].append(time.perf_counter()-before)

    with torch.no_grad(): later.up.weight.copy_(interpolated_up)
    candidate=s66.snapshot(model,sources,first,later,head,tok,orig,aug,history)
    result=s66.stats(candidate,orig,aug,old_ids)
    ratio=statistics.median(speed["baseline"])/statistics.median(speed["candidate"])
    eligible=(
        result["all_previous_full_answer_and_first_token_skills_preserved"] and
        result["original_full"]+result["augmented_full"]>=42+27+MIN_FULL_GAIN and
        result["historical_full"]>=7 and
        result["original_opposed_pairs_both"]>=13 and
        result["augmented_opposed_pairs_both"]>=7 and
        result["original_opposed_pairs_both"]+result["augmented_opposed_pairs_both"]>=MIN_OPPOSING_PAIR_PRESERVED and
        ratio>=.9
    )
    report={
        "stage":"CALIBRI_S67_EXPLORATORY_SAFE_HALF_DELTA_TRAIN_ONLY",
        "alpha":ALPHA,
        "S66_full_acceptance_failed_at_every_alpha":True,
        "S67_goal_pre_registered_separately_not_an_S66_retroactive_success":True,
        "S56_original_native_full_answer_counts":base,
        "candidate_native_full_answer_counts":result,
        "native_16_TR​AIN_prompt_CPU_speed_ratio":ratio,
        "native_CPU_speed_90_percent_floor_pass":ratio>=.9,
        "candidate_is_only_exploratory_eligible_for_new_blind_QA":eligible,
        "NO_independent_QA_or_FP32_comparison_done":True,
        "NO_GPU_NO_Jessica_chat_test_NO_LIVE_modifications":True,
    }
    if eligible:
        output=a.output_dir/"S67_HALF_DELTA_RANK128_CANDIDATE_ONLY.safetensors"
        s65.sidecar(first,later,output)
        m=paid_manifest(paid_source,output.read_bytes(),
                        a.output_dir/"S67_FULLY_PAID_EXPLORATORY_MANIFEST.json")
        report["candidate_sidecar_sha256"]=sha(output.read_bytes())
        report["all_inclusive_bpp"]=m["all_inclusive_bpp"]
        report["all_inclusive_bytes"]=m["all_inclusive_bytes"]
    (a.output_dir/"S67_TRAIN_ONLY_ACTUAL_NATIVE_REPLAY.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S67_ACTUAL_CPU_SAFE_INCREMENTAL_NATIVE_TRAIN",
      "eligible":eligible,"replayed_S56_original":base["original_full"],
      "replayed_candidate_original":result["original_full"],
      "replayed_candidate_augmented":result["augmented_full"],
      "preserved_old7":result["historical_full"],
      "new_plus_old_opposing_both":result["original_opposed_pairs_both"]+result["augmented_opposed_pairs_both"],
      "median_cpu_ratio":ratio,
      "physical_bpp":report.get("all_inclusive_bpp")}),flush=True)

if __name__=="__main__":
    main()
