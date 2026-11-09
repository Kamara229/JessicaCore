"""CALIBRI S86 — read-only composition of two ALREADY frozen GPU-trained skills.

Independent TRAIN-only hypothesis preregistered WITHOUT seeing S85
one-time new-noun QA results. Prior inspected QA S85/S79/S72/S68/S55/
S53/S29 are PERMANENTLY EMBARGOED. This script imports NONE and
consults no heldout prompts, results, text or scores.

All 3 source controllers were already physically paid at <=0.7bpp:
S74 rank16 first + frozen rank128 after-first;
S81 (S74 + GPU monotone protected neural late-rank16);
S82 (S74 + GPU generic token-context pointer on second word).
Both S81 and S82 preserve every S74 originally right TRAIN identity.

S86 composes the exact frozen S81 later-rank16 with S82 generic neural
pointer (without ANY new gradients, checkpoint search, threshold
tuning, data-specific prompt rules, model training or QA). Neither
pre-existing model modified.

Evaluate the entire previously authored 456 TRAIN prompts, same
original 36M temporarily reconstructed FP32 on true MPS GPU, and run
four source candidates S74/S81/S82/S86 with exact full-autoregressive
token generation. Reject combined candidate if it loses any previously
right full OR first TRAIN response from the UNION of 3 sources; accept
only if it also adds at least one new FULL correct native TRAIN answer.
Upon provisional success, prove every generated token ID of the chosen
S86 identical to original compressed CB59 model on real CPU across
ALL 456 TRAIN prompts and CPU speed >=0.9x original S82 speed.
Pay compressed source+S81 sidecar+S82 pointer+manifest byte-perfect,
<=0.7bpp. Otherwise report SCIENTIFIC NEGATIVE without saving weights.

No independent unseen generalization, true FP32 baseline parity or
natural-language understanding claims. No LIVE/main/Jessica changes.
"""
from __future__ import annotations
import argparse,hashlib,json,time,statistics
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s84 import bitwise_certify_s80_physically_paid_232_real_cpu_mps as s84
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from v186.pairs import build_pairs

def sha(data):return hashlib.sha256(data).hexdigest()

def read_physical(a):
    raw=a.S45_immutable_CB59.read_bytes()
    cb61=a.S61_lossless_compressed.read_bytes()
    b74=a.S74_frozen_controller.read_bytes()
    b81=a.S81_frozen_late_controller.read_bytes()
    b82=a.S82_frozen_pointer_controller.read_bytes()
    m74=json.loads(a.S74_physical_manifest.read_text())
    m81=json.loads(a.S81_physical_manifest.read_text())
    m82=json.loads(a.S82_physical_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        m74["S74_GPU_full_controller_SHA256"]!=sha(b74) or
        m74["S61_unchanged_exact_compressed_source_SHA256"]!=sha(cb61) or
        m74["physical_all_inclusive_bytes"]!=len(cb61)+len(b74)+a.S74_physical_manifest.stat().st_size or
        float(m74["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S86 unchanged S74 core controller/physical proof invalid")
    if (m81["original_S74_full_controller_SHA256"]!=sha(b74) or
        m81["new_controller_SHA256"]!=sha(b81) or
        m81["physically_paid_compressed_source_sha256"]!=sha(cb61) or
        m81["full_physical_bytes"]!=len(cb61)+len(b81)+a.S81_physical_manifest.stat().st_size or
        float(m81["all_inclusive_physical_bpp"])>.7):
        raise AssertionError("S86 immutable GPU-trained S81 paid source invalid")
    if (m82["S74_original_frozen_controller_SHA256"]!=sha(b74) or
        m82["S82_full_new_controller_SHA256"]!=sha(b82) or
        m82["original_S61_physically_compressed_source_SHA256"]!=sha(cb61) or
        m82["all_inclusive_physical_bytes"]!=len(cb61)+len(b82)+a.S82_physical_manifest.stat().st_size or
        float(m82["all_inclusive_bpp"])>.7):
        raise AssertionError("S86 immutable GPU-trained S82 paid source invalid")
    w74=load_file(str(a.S74_frozen_controller))
    w81=load_file(str(a.S81_frozen_late_controller))
    w82=load_file(str(a.S82_frozen_pointer_controller))
    extra81=set(w81)-set(w74)
    extra82=set(w82)-set(w74)
    if extra81!={"later_increment.down.weight",
                 "later_increment.up.weight","later_increment.gate.weight",
                 "later_increment.gate.bias"}:
        raise AssertionError("S86 unexpected S81 late-rank16 weights")
    if extra82!={"generic_pointer.query.weight",
                 "generic_pointer.key.weight","generic_pointer.gate.weight",
                 "generic_pointer.gate.bias","generic_pointer.log_gain"}:
        raise AssertionError("S86 unexpected S82 attention-pointer weights")
    if any(k not in d or not torch.equal(v,d[k])
           for k,v in w74.items() for d in (w81,w82)):
        raise AssertionError("S86 source/frozen S74 original changed")
    return raw,cb61,(w74,w81,w82),{
        "source_CB59_sha256":sha(raw),
        "physical_CB61_sha256":sha(cb61),
        "S74_controller_sha256":sha(b74),
        "S81_controller_sha256":sha(b81),
        "S82_controller_sha256":sha(b82),
        "source_physical_bpps":{
            "S74":m74["physical_all_inclusive_bpp"],
            "S81":m81["all_inclusive_physical_bpp"],
            "S82":m82["all_inclusive_bpp"]},
    }

def frozen_pointer(w):
    pointer=s82.GenericContextPointer()
    state={k.removeprefix("generic_pointer."):v for k,v in w.items()
           if k.startswith("generic_pointer.")}
    if set(state)!=set(pointer.state_dict()):
        raise AssertionError("S86 pointer weights fail exact schema check")
    pointer.load_state_dict(state,strict=True)
    return pointer.eval().requires_grad_(False)

def score(ids,tok,answer):
    gold=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
    return {"ids":ids,"full_correct":ids==gold,
            "first_correct":bool(ids) and ids[0]==gold[0]}

def frozen_manifest(compact,side,metadata,output_dir):
    m={
      "stage":"CALIBRI_S86_REAL_GPU_FROZEN_S81_LATER_PLUS_S82_NEURAL_POINTER",
      "full_original_source_CB59_sha256":metadata["source_CB59_sha256"],
      "physical_source_CB61_sha256":sha(compact),
      "frozen_source_S74_controller_SHA256":metadata["S74_controller_sha256"],
      "frozen_S81_late_controller_SHA256":metadata["S81_controller_sha256"],
      "frozen_S82_pointer_controller_SHA256":metadata["S82_controller_sha256"],
      "combined_controller_SHA256":sha(side),
      "source_bytes":len(compact),
      "combined_controller_all_bytes":len(side),
      "added_training_steps":0,
      "all_parameters_frozen_prior_to_any_S85_QA_outcome":True,
      "all_previous_inspected_QA_permanently_embargoed":True,
      "no_independent_qa_or_FP32_parity_claim":True,
      "no_LIVE_Jessica_changes":True
    }
    for _ in range(35):
        b=json.dumps(m,indent=2,sort_keys=True).encode()
        size=len(compact)+len(side)+len(b)
        changes={"manifest_bytes":len(b),"full_all_inclusive_bytes":size,
                 "full_all_inclusive_bpp":f"{size*8/SOURCE_PARAMS:.12f}"}
        if all(m.get(k)==v for k,v in changes.items()):break
        m.update(changes)
    encoded=json.dumps(m,indent=2,sort_keys=True).encode()
    if m["full_all_inclusive_bytes"]!=len(compact)+len(side)+len(encoded):
        raise AssertionError("S86 all physical bytes manifest does not converge")
    if m["full_all_inclusive_bytes"]*8/SOURCE_PARAMS>.7:
        raise AssertionError("S86 all-in physical bpp >0.7")
    (output_dir/"S86_FROZEN_COMPOSED_SOURCE_ALL_INCLUSIVE_PAID_MANIFEST.json").write_bytes(encoded)
    return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_immutable_CB59",type=Path)
    p.add_argument("S61_lossless_compressed",type=Path)
    p.add_argument("S74_frozen_controller",type=Path)
    p.add_argument("S74_physical_manifest",type=Path)
    p.add_argument("S81_frozen_late_controller",type=Path)
    p.add_argument("S81_physical_manifest",type=Path)
    p.add_argument("S82_frozen_pointer_controller",type=Path)
    p.add_argument("S82_physical_manifest",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S86 must execute full native frozen 36M on REAL MPS GPU")
    torch.manual_seed(860086);torch.set_num_threads(2)
    raw,compact,(w74,w81,w82),refs=read_physical(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    newly,words=s82.newly_authored_train(tok)
    pairs={
       "S30":s30.make_training_pairs(tok),
       "S54":s54.build_new_train(tok),
       "history":build_pairs(tok,"TRAIN")[:4],
       "S69":s69.build_new_train(tok),
       "S73":s73.new_s73_train(tok),
       "S82_new":newly,
    }
    if [len(x)*2 for x in pairs.values()]!=[64,64,8,32,32,256]:
        raise AssertionError("S86 456 authorized TRAIN provenance changed")
    dense=s30.dense_from_paid(CB59(raw)).to("mps").eval().requires_grad_(False)
    if sum(p.numel() for p in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S86 wrong unique model parameter count")
    head=dense.lm_head.weight.detach()
    if head.device.type!="mps":raise AssertionError("S86 not on real GPU")
    f74,l74=s78.controllers(w74)
    f74=f74.to("mps").eval().requires_grad_(False)
    l74=l74.to("mps").eval().requires_grad_(False)
    l81=s84.frozen_late_s80(s78.controllers(w74)[1],w81)
    l81=l81.to("mps").eval().requires_grad_(False)
    ptr=frozen_pointer(w82).to("mps").eval().requires_grad_(False)
    systems={
       "S74_original":(l74,None),
       "S81_monotone_late":(l81,None),
       "S82_context_pointer":(l74,ptr),
       "S86_combined_no_training":(l81,ptr),
    }
    out={name:{} for name in systems}
    summaries={}
    first_loss=None
    new_uncovered_correct=0
    for fam,allpairs in pairs.items():
        for pair in allpairs:
            for side in ("left","right"):
                key=f'{pair["pair_id"]}_{side}'
                item=pair[side]
                for name,(later,pointer) in systems.items():
                    ids,_=s82.full_gpu_generate(
                        dense,f74,later,head,tok,item["question"],pointer)
                    out[name][key]=score(ids,tok,item["answer"])
        summary={}
        for name in systems:
            keys=[f'{p["pair_id"]}_{side}' for p in allpairs for side in ("left","right")]
            summary[name]={"full":sum(out[name][key]["full_correct"] for key in keys),
                           "first":sum(out[name][key]["first_correct"] for key in keys),
                           "out_of":len(keys)}
        summaries[fam]=summary
        print(json.dumps({"event":"S86_ACTUAL_FULL_GPU_READ_ONLY_COMPOSITION_TRAIN_FAMILY",
            "family":fam,"scores":summary}),flush=True)
    preexisting=("S74_original","S81_monotone_late","S82_context_pointer")
    lost=[]
    gains=[]
    for key in out["S86_combined_no_training"]:
        new=out["S86_combined_no_training"][key]
        prior=[out[n][key] for n in preexisting]
        if any(p["full_correct"] and not new["full_correct"] or
               p["first_correct"] and not new["first_correct"] for p in prior):
            lost.append(key)
        if new["full_correct"] and all(not p["full_correct"] for p in prior):
            gains.append(key)
    report={
      "stage":"S86_FROZEN_GPU_TRAIN_ONLY_S81_LATE_PLUS_S82_CONTEXTUAL_POINTER",
      "immutable_original_36M_reference":refs,
      "all_prior_source_controllers_bitwise_verified":True,
      "all_newly_authored_TRAIN_nouns_only_not_QA":words,
      "all_456_TRAIN_QUESTIONS_GPU_FULL_NATIVE":True,
      "by_family_FULL_and_FIRST_GPU_TRAIN":summaries,
      "old_right_UNION_S74_S81_S82_lost_on_combined":lost,
      "new_full_right_not_in_any_prerequisite_model":gains,
      "eligible_as_TRAIN_candidate_prior_to_CPU_test":not lost and bool(gains),
      "never_loaded_or_ran_S85_S79_or_any_prior_QA":True,
      "no_weights_trained_or_test_QA_run":True,
      "main_LIVE_Jessica_Supabase_Render_untouched":True,
    }
    if not lost and gains:
        # The exact accepted S81 original neural weights and S82 learned
        # pointer are combined byte-for-byte, no new learned fitting.
        w86=dict(w81)
        for k,v in w82.items():
            if k.startswith("generic_pointer."):
                w86[k]=v.detach().cpu().contiguous()
        saved=a.output_dir/"S86_COMPOSED_S81_LATE_PLUS_S82_POINTER_FROZEN_CANDIDATE.safetensors"
        save_file(w86,str(saved))
        reloaded=load_file(str(saved))
        if any(k not in reloaded or not torch.equal(v,reloaded[k])
               for k,v in w81.items()) or any(
                   not torch.equal(v,reloaded[k]) for k,v in w82.items()
                   if k.startswith("generic_pointer.")):
            raise AssertionError("S86 combined paid controller changed accepted originals")
        paid=frozen_manifest(compact,saved.read_bytes(),refs,a.output_dir)
        report["new_complete_physical_bpp"]=paid["full_all_inclusive_bpp"]
        cmodel,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
        cmodel.eval().requires_grad_(False)
        cpu_head=cmodel.lm_head.source().detach().cpu().contiguous()
        fc,_=s78.controllers(w74)
        cpu_late=s84.frozen_late_s80(s78.controllers(w74)[1],w81)
        cpu_ptr=frozen_pointer(w82)
        native_differences=[]
        for fam,allpairs in pairs.items():
            for pair in allpairs:
                for side in ("left","right"):
                    key=f'{pair["pair_id"]}_{side}'
                    q=pair[side]["question"]
                    ids=s82.full_cpu_generate(
                        cmodel,sources,fc,cpu_late,cpu_head,tok,q,cpu_ptr)
                    if ids!=out["S86_combined_no_training"][key]["ids"]:
                        native_differences.append(key)
        report["exact_456_cpu_gpu_full_token_sequence_mismatch_ids"]=native_differences
        if not native_differences:
            speed={}
            for label,late,pointer in (("S82",s78.controllers(w74)[1],cpu_ptr),
                                       ("S86",cpu_late,cpu_ptr)):
                durations=[]
                for _,_,q,_ in make_samples(tok):
                    start=time.perf_counter()
                    s82.full_cpu_generate(cmodel,sources,fc,late,cpu_head,tok,q,pointer)
                    durations.append(time.perf_counter()-start)
                speed[label]=statistics.median(durations)
            ratio=speed["S82"]/speed["S86"]
            report["real_cpu_speed_ratio_S86_to_S82"]=ratio
            report["physical_bpp_and_cpu_gate_pass"]=ratio>=.9
            if ratio>=.9:
                report["accepted_as_frozen_train_only_composition"]=True
            else:report["accepted_as_frozen_train_only_composition"]=False
        else:report["accepted_as_frozen_train_only_composition"]=False
        if not report["accepted_as_frozen_train_only_composition"]:
            saved.unlink(missing_ok=True)
            (a.output_dir/"S86_FROZEN_COMPOSED_SOURCE_ALL_INCLUSIVE_PAID_MANIFEST.json").unlink(missing_ok=True)
    else:report["accepted_as_frozen_train_only_composition"]=False
    (a.output_dir/"S86_ACTUAL_GPU_FULL_TRAIN_COMPOSITION_AND_CPU_ACCEPTANCE.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S86_FINAL_FROZEN_SKILL_COMPOSITION_TRAIN_ONLY",
        "candidate_accepted":report["accepted_as_frozen_train_only_composition"],
        "union_lost_old_correct":len(lost),
        "gained_NEW_full_correct_over_ALL_prior":len(gains),
        "accepted_bpp":report.get("new_complete_physical_bpp"),
        "S85_and_all_prior_independent_QA_not_read":True}),flush=True)

if __name__=="__main__":main()
