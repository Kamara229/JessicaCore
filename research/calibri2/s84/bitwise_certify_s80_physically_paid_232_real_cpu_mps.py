"""CALIBRI S84: strict bitwise identity certificate for S80 paid checkpoint.

S80 CLAIMED a positive TRAIN-only step but verified final compressed CPU
and GPU *aggregate counts*, not necessarily exact per-prompt token
sequences. This independent implementation audit checks ALL 232
pre-existing TRAIN prompts and fails closed if even one S80 CPU/GPU token
list differs, if S74 originals were changed or forgotten, if physical
all-in <=0.7 bpp cannot be proven, or if the claimed TRAIN improvement
was not reproduced. No model training and NO unseen QA. S79, S72, S68,
S55, S53, S29 and all inspected heldout questions are forbidden to
import, regenerate, check, use or read.

Original physically paid S74 and S80 SHA256 artifacts are immutable.
S80 GPU rank16 correction is loaded only from its already accepted
physical paid artifact, not reconstructed or picked by checkpoint hunt.
CPU exact compressed CB59 source is the authoritative inference engine.
GPU uses temporary SAME 36M FP32 expansion from compressed CB59,
strictly for computing exact token parity and CPU speed.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s80 import real_mps_late_rank16_preserve_all_alpha_search as s80
from calibri2.s77.full_mps_incremental_kv_cache_200_train_parity import dense_mps_cached_greedy
from v186.pairs import build_pairs

def sha(data):return hashlib.sha256(data).hexdigest()

def frozen_late_s80(old,weights):
    later=s80.LateResidual16(old)
    extra={k.removeprefix("later_increment."):v for k,v in weights.items()
           if k.startswith("later_increment.")}
    if set(extra)!={"down.weight","up.weight","gate.weight","gate.bias"}:
        raise AssertionError("S84 S80 additional late rank16 weights absent")
    later.load_state_dict({
       **{f"frozen.{k}":v for k,v in old.state_dict().items()},
       **extra},strict=True)
    return later.eval().requires_grad_(False)

def actual(tok,ids,answer):
    gold=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
    return {"tokens":ids,"gold":gold,"full_correct":ids==gold,
            "first_correct":bool(ids) and ids[0]==gold[0]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_immutable_CB59",type=Path)
    p.add_argument("S61_immutable_compressed",type=Path)
    p.add_argument("S74_frozen_controller",type=Path)
    p.add_argument("S74_frozen_physical_manifest",type=Path)
    p.add_argument("S80_frozen_controller",type=Path)
    p.add_argument("S80_frozen_physical_manifest",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S84 actual Apple MPS GPU needed, no fallback")
    torch.manual_seed(840084);torch.set_num_threads(2)
    raw=a.S45_immutable_CB59.read_bytes()
    compact=a.S61_immutable_compressed.read_bytes()
    b74=a.S74_frozen_controller.read_bytes()
    b80=a.S80_frozen_controller.read_bytes()
    m74=json.loads(a.S74_frozen_physical_manifest.read_text())
    m80=json.loads(a.S80_frozen_physical_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        m74["S74_GPU_full_controller_SHA256"]!=sha(b74) or
        m74["S61_unchanged_exact_compressed_source_SHA256"]!=sha(compact) or
        m74["physical_all_inclusive_bytes"]!=
            len(compact)+len(b74)+a.S74_frozen_physical_manifest.stat().st_size or
        sha(b74)!=m80["original_S74_full_controller_SHA256"] or
        sha(compact)!=m80["physically_paid_compressed_source_sha256"] or
        sha(b80)!=m80["new_controller_SHA256"] or
        m80["full_physical_bytes"]!=
            len(compact)+len(b80)+a.S80_frozen_physical_manifest.stat().st_size or
        float(m80["all_inclusive_physical_bpp"])>.7 or
        float(m74["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S84 immutable exact source/manifest/0.7bpp mismatch")
    w74=load_file(str(a.S74_frozen_controller))
    w80=load_file(str(a.S80_frozen_controller))
    if set(w80)-set(w74)!={
         "later_increment.down.weight","later_increment.up.weight",
         "later_increment.gate.weight","later_increment.gate.bias"}:
        raise AssertionError("S84 S80 has unexpected modified/additional source weights")
    if any(not torch.equal(w80[k],v) for k,v in w74.items()):
        raise AssertionError("S84 S80 original S74 frozen Green Tunnel was MUTATED")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    families={
       "original":s30.make_training_pairs(tok),
       "augmented":s54.build_new_train(tok),
       "history":build_pairs(tok,"TRAIN")[:4],
       "S69":s69.build_new_train(tok),
       "S73":s73.new_s73_train(tok),
       "S78":s78.new_train(tok)}
    if [len(v)*2 for v in families.values()]!=[64,64,8,32,32,32]:
        raise AssertionError("S84 reference original 232 TRAIN sample count drift")
    # Native compressed codebook, not an approximation through a dense
    # output projection; same original source on CPU + MPS.
    from calibri2.s0.bootstrap import build_config
    from calibri2.s0.train import create_small_paid_model
    from calibri2.s8.broad_replay import MODEL_SPEC
    cmodel,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    cmodel.eval().requires_grad_(False)
    head=cmodel.lm_head.source().detach().cpu().contiguous()
    oldfirst,oldlater=s78.controllers(w74)
    newlater=frozen_late_s80(oldlater,w80)
    dense=s30.dense_from_paid(CB59(raw)).to("mps").eval().requires_grad_(False)
    if sum(p.numel() for p in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S84 temporary GPU full source not correct 36M")
    firstg,laterg=s78.controllers(w74)
    newer=frozen_late_s80(laterg,w80)
    firstg=firstg.to("mps").eval().requires_grad_(False)
    laterg=laterg.to("mps").eval().requires_grad_(False)
    newer=newer.to("mps").eval().requires_grad_(False)
    gpuhead=dense.lm_head.weight.detach()
    if gpuhead.device.type!="mps":raise AssertionError("S84 GPU head not Apple MPS")
    records={}
    summary={}
    all_matches=0
    all_checked=0
    old_preserved=True
    for group,pairs in families.items():
        entries=[]
        for pair in pairs:
            for side in ("left","right"):
                item=pair[side]
                q=item["question"];expected=item["answer"]
                old_cpu=s56.greedy_first_active(cmodel,sources,oldfirst,oldlater,head,tok,q)
                new_cpu=s56.greedy_first_active(cmodel,sources,oldfirst,newlater,head,tok,q)
                old_gpu,_=dense_mps_cached_greedy(dense,tok,firstg,laterg,gpuhead,q)
                new_gpu,_=dense_mps_cached_greedy(dense,tok,firstg,newer,gpuhead,q)
                cpu_old=actual(tok,old_cpu,expected)
                cpu_new=actual(tok,new_cpu,expected)
                parity_old=old_cpu==old_gpu
                parity_new=new_cpu==new_gpu
                if cpu_old["full_correct"] and not cpu_new["full_correct"]:old_preserved=False
                if cpu_old["first_correct"] and not cpu_new["first_correct"]:old_preserved=False
                if not parity_old or not parity_new:
                    raise AssertionError(
                       f"S84 STRICT GPU/CPU token parity failure: {group}/{pair['pair_id']}/{side}"
                       f" S74 {old_cpu} != {old_gpu}; S80 {new_cpu} != {new_gpu}")
                all_matches+=1;all_checked+=1
                entries.append({
                  "id":f'{pair["pair_id"]}_{side}',
                  "old":cpu_old,"new":cpu_new,
                  "identical_real_compressed_CPU_vs_real_MPS_S74":parity_old,
                  "identical_real_compressed_CPU_vs_real_MPS_S80":parity_new,
                  "old_correct_full_preserved":
                    not cpu_old["full_correct"] or cpu_new["full_correct"],
                  "old_correct_first_preserved":
                    not cpu_old["first_correct"] or cpu_new["first_correct"],
                })
        summary[group]={
            "TRAIN_count":len(entries),
            "S74_full_correct":sum(r["old"]["full_correct"] for r in entries),
            "S80_full_correct":sum(r["new"]["full_correct"] for r in entries),
            "S74_first_correct":sum(r["old"]["first_correct"] for r in entries),
            "S80_first_correct":sum(r["new"]["first_correct"] for r in entries),
            "lost_previous_full_ids":[r["id"] for r in entries
                                      if not r["old_correct_full_preserved"]],
            "lost_previous_first_ids":[r["id"] for r in entries
                                       if not r["old_correct_first_preserved"]],
            "exact_original_CPU_and_GPU_tokens":len(entries),
            "exact_new_CPU_and_GPU_tokens":len(entries),
        }
        records[group]=entries
        print(json.dumps({"event":"S84_ALL_NATIVE_OLD_SKILL_BITWISE_IDENTITY_AUDIT",
             "family":group,**summary[group]}),flush=True)
    if not old_preserved or all_checked!=232:
        raise AssertionError("S84 old protected TRAIN identities lost or incomplete")
    if sum(v["S80_full_correct"] for v in summary.values())<=sum(
       v["S74_full_correct"] for v in summary.values()):
        raise AssertionError("S84 original S80 purported TRAIN gains did NOT reproduce")
    # Each sample was checked EXACT generated tokens under real CPU and
    # real GPU, not just aggregate counts or a loss proxy.
    report={
       "event":"S84_REAL_GPU_CPU_BITWISE_S80_REPRODUCTION_CERTIFICATE_TRAIN_ONLY",
       "all_232_full_old_S74_native_CPU_GPU_token_sequences_equal":all_matches==232,
       "all_232_full_new_S80_native_CPU_GPU_token_sequences_equal":all_matches==232,
       "original_S74_SHA256":sha(b74),"new_S80_SHA256":sha(b80),
       "S80_full_all_inclusive_bpp":m80["all_inclusive_physical_bpp"],
       "S74_full_all_inclusive_bpp":m74["physical_all_inclusive_bpp"],
       "immutability_old_weights_proven":True,
       "old_full_and_first_correct_original_232_TRAIN_ids_preserved":old_preserved,
       "TRAIN_correct_gain":sum(v["S80_full_correct"]-v["S74_full_correct"]
                                for v in summary.values()),
       "per_TRAIN_family":summary,
       "all_actual_native_generated_train_token_ids":records,
       "never_updated_any_weights":True,
       "read_previous_inspected_or_S79_QA":False,
       "no_independent_generalization_claim":True,
       "original_full_GPU_temporary_weights_saved":False,
       "main_LIVE_Jessica_Render_Supabase_untouched":True
    }
    (a.output_dir/"S84_ALL_232_REAL_GPU_CPU_S80_BITWISE_FULL_TRAIN_CERTIFICATE.json").write_text(
        json.dumps(report,indent=2))
    print(json.dumps({"event":"S84_FINAL_BITWISE_S80_TRAIN_CERTIFIED",
       "all232_exact_CPU_GPU":True,
       "old_correct_full_first_preserved":old_preserved,
       "physical_bpp":m80["all_inclusive_physical_bpp"],
       "new_FULL_TRAIN_gain":report["TRAIN_correct_gain"],
       "no_QA_or_train":True}),flush=True)

if __name__=="__main__":main()
