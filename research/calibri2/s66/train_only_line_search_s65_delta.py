"""S66: Interpolate S65's rejected TRAIN delta in actual Green Tunnel outputs.

S65 step16 lost an original S56 full correct answer (42->41), even though
augmented improved (27->28), so it was rejected. Do not misreport success.
On the *same TRAIN only* measure frozen S56 rank128 base and scaled versions
of the S65 added116-column update; protect every original full correct answer,
old skills, first-token colors and 20 originally solved opposed pairs.

No held-out QA, FP32 claims, GPU or Jessica tests, no LIVE changes.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from v186.pairs import build_pairs

ALPHAS=(.0625,.125,.25,.375,.5,.75,1.0)
SRC_SHA=s65.S45_SHA
CTRL_SHA=s65.S64_CONTROLLER_SHA

def sha(blob):return hashlib.sha256(blob).hexdigest()
def frozen_ids(rows):return s58.correct(rows)
def first_ids(rows):return s58.first_correct(rows)

def snapshot(model,sources,first,later,head,tok,orig,added,old):
    return (s56.probe(model,sources,first,later,head,tok,orig),
            s56.probe(model,sources,first,later,head,tok,added),
            s56.probe(model,sources,first,later,head,tok,old[:4]))

def stats(rows,original,added,keep):
    a,b,o=rows
    now=(frozen_ids(a),frozen_ids(b),frozen_ids(o),first_ids(a),first_ids(b))
    lost=[sorted(previous-current) for previous,current in zip(keep,now)]
    pa=s58.both_correct(a,original)
    pb=s58.both_correct(b,added)
    result={
      "original_full":len(now[0]),"augmented_full":len(now[1]),
      "historical_full":len(now[2]),
      "original_first":len(now[3]),"augmented_first":len(now[4]),
      "original_opposed_pairs_both":pa,"augmented_opposed_pairs_both":pb,
      "lost_correct_train_identifiers":{
          "original":lost[0],"augmented":lost[1],"historical":lost[2],
          "original_first":lost[3],"augmented_first":lost[4],
      },
      "all_previous_full_answer_and_first_token_skills_preserved":all(not ids for ids in lost),
    }
    result["eligible_train_candidate"]=(
      result["all_previous_full_answer_and_first_token_skills_preserved"] and
      pa>=13 and pb>=7 and pa+pb>20 and
      len(now[0])+len(now[1])>=69)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_CB59",type=Path)
    p.add_argument("S56_immutable_controller",type=Path)
    p.add_argument("S64_rank128_controller",type=Path)
    p.add_argument("S61_lossless_CB61",type=Path)
    p.add_argument("S64_paid_manifest",type=Path)
    p.add_argument("S65_step16_cpu_resume",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(660066)
    source=a.S45_CB59.read_bytes()
    old=a.S56_immutable_controller.read_bytes()
    new=a.S64_rank128_controller.read_bytes()
    packed=a.S61_lossless_CB61.read_bytes()
    if len(source)!=SOURCE_BYTES or sha(source)!=SRC_SHA or sha(old)!=s65.S56_SHA or sha(new)!=CTRL_SHA:
        raise ValueError("S66 immutable original/S64 physical source not pinned")
    paid=json.loads(a.S64_paid_manifest.read_text())
    if paid["controller_sha256"]!=CTRL_SHA or paid["compressed_source_sha256"]!=sha(packed) or float(paid["all_inclusive_bpp"])>.7:
        raise ValueError("S64 controller/source not fully physically paid")
    ck=torch.load(str(a.S65_step16_cpu_resume),map_location="cpu",weights_only=True)
    if ck.get("step")!=16 or ck.get("immutable_original_sha256")!=SRC_SHA or ck.get("immutable_S64_sidecar_sha256")!=CTRL_SHA:
        raise ValueError("S65 frozen train resume provenance mismatch")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    orig=s30.make_training_pairs(tok)
    added=s54.build_new_train(tok)
    oldpairs=build_pairs(tok,"TRAIN")
    assert len(orig)==32 and len(added)==32 and len(oldpairs)==24
    model,sources=create_small_paid_model(CB59(source),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    state=load_file(str(a.S64_rank128_controller))
    first.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    baseline_new=state["later.up.weight"].detach().clone()
    delta=ck["later"]["up.weight"].detach().clone()-baseline_new
    if tuple(delta.shape)!=(320,128) or torch.count_nonzero(delta[:,:12]).item()!=0:
        raise AssertionError("S65 changed protected original rank12 weights")
    for k in ("down.weight","gate.weight","gate.bias"):
        if not torch.equal(ck["later"][k],state["later."+k]):
            raise AssertionError("S65 changed protected "+k)
    base=snapshot(model,sources,first,later,head,tok,orig,added,oldpairs)
    frozen=(frozen_ids(base[0]),frozen_ids(base[1]),frozen_ids(base[2]),
            first_ids(base[0]),first_ids(base[1]))
    baseline_result=stats(base,orig,added,frozen)
    assert (
        baseline_result["original_full"],baseline_result["augmented_full"],
        baseline_result["historical_full"],baseline_result["original_first"],
        baseline_result["augmented_first"],baseline_result["original_opposed_pairs_both"],
        baseline_result["augmented_opposed_pairs_both"]
    )==(42,27,7,58,46,13,7),baseline_result
    candidates=[]
    best_key=(20,69)
    best=None
    for alpha in ALPHAS:
        with torch.no_grad():
            later.up.weight.copy_(baseline_new + float(alpha)*delta)
        start=time.perf_counter()
        actual=snapshot(model,sources,first,later,head,tok,orig,added,oldpairs)
        report=stats(actual,orig,added,frozen)
        report["S65_added_column_delta_fraction"]=alpha
        report["cpu_full_train_probe_seconds"]=time.perf_counter()-start
        candidates.append(report)
        if report["eligible_train_candidate"]:
            key=(report["original_opposed_pairs_both"]+report["augmented_opposed_pairs_both"],
                 report["original_full"]+report["augmented_full"])
            if key>best_key:
                best_key=key
                best=(alpha,later.up.weight.detach().clone())
        print(json.dumps({"event":"S66_TRUE_NATIVE_FULL_ANSWER_INTERPOLATION_TRAIN_ONLY",**report}),flush=True)
    eligible=best is not None
    result={
      "stage":"CALIBRI_S66_S65_NEGATIVE_DELTA_LINE_SEARCH_TRAIN_ONLY",
      "S65_step16_rejected_loss_of_original_correct":True,
      "unaltered_S56_reference_quality":baseline_result,
      "probes":candidates,
      "eligible_train_candidate_for_future_blind_QA":eligible,
      "not_new_independent_quality_or_FP32_parity_evidence":True,
      "old_rank12_and_first_rank2_and_all_CB59_8D_weights_unchanged":True,
      "no_GPU_or_Jessica_CHAT_test_or_LIVE_changes":True,
    }
    if eligible:
        alpha,learned=best
        with torch.no_grad():later.up.weight.copy_(learned)
        out=a.output_dir/"S66_ACCEPTED_TRAIN_ONLY_GREEN_TUNNEL.safetensors"
        s65.sidecar(first,later,out)
        manifest=s65.saved_physical_manifest(packed,out.read_bytes(),
                       a.output_dir/"S66_PHYSICALLY_PAID_TRAIN_ONLY_MANIFEST.json")
        result.update({"selected_alpha":alpha,
                       "all_inclusive_bpp":manifest["all_inclusive_bpp"],
                       "all_inclusive_source_plus_sidecar_plus_manifest_bytes":manifest["all_inclusive_bytes"],
                       "promotion_requires_fresh_independent_QA_and_CPU_speed":True})
    (a.output_dir/"S66_TRAIN_INTERPOLATION_RESULT.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"event":"S66_FINAL", "eligible_train_candidate_for_future_blind_QA":eligible,
       "tested_alphas":len(candidates), "accepted_alpha":best[0] if eligible else None}),flush=True)

if __name__=="__main__": main()
