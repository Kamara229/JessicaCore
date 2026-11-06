"""CALIBRI v214: full-vocabulary native-logit Green Tunnel calibration.

v206 has 3/16 on public DEV; two of three are coincidental Yes/No successes
on examples the original FP32 actually got wrong. The compressed model
severely overproduces a handful of high-bias tokens ('No', 'red', '5').
v213 naive FP32-weight SVD worsened to 0/16, because compressed hidden
representations and packed tied head are co-adapted. So preserve embedding
and all original packed operators and calibrate the OUTPUT head against
the *actual compressed* hidden states and FP32 outputs on public TRAIN only.

Two physically accounted correction mechanisms:
 A. only learned 49k output bias (under 100 KB);
 B. trained rank-4 logit residual + output bias (~0.50 MB).
Both are within INCLUSIVE 0.700 bpp, and run on the actual native packed C
kernel. The only learned weights are the sealed FP16 tensors.
Hyperparameter gain selected on distinct held-out TRAIN records, NEVER QA
DEV/VALID/TEST; answers judged only after physical sealing. Not a 1:1 copy.
No live Jessica, GitHub main, Render, Supabase or APK changes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v186.run import speed_pairs
from v196.data import training_inputs, positions_and_targets
from v196.run import newly_frozen_public_dev_16
from v196 import protocol as p
from v212.run import SOURCE_V206_SHA256

RANK=4
RIDGE=2.0
CANDIDATE_GAINS=(0.125,0.25,0.5,1.0)
MAX_PHYSICAL_BPP=0.700
TRAIN_TEXT_HELDOUT_MOD=5
TRAIN_PAIR_HELDOUT_MOD=3
MAX_SEQ=512


def freeze_native_teacher_train_inputs(teacher,native,tok):
    texts,pairs,_,origin=training_inputs(tok)
    if len(texts)!=20 or len(pairs)!=24:
        raise AssertionError("FROZEN TRAIN inputs incomplete")
    fit_h=[];fit_y=[];hold_h=[];hold_y=[];hold_teacher=[];hold_base=[];hold_expected=[]
    counters={"fit_tokens":0,"holdout_tokens":0,"fit_pair_prompts":0,"holdout_pair_prompts":0}
    def collect(ids,positions,fit,expected_gold=None):
        if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]>MAX_SEQ:
            raise AssertionError("Forbidden missing or overlong TRAIN text")
        if positions.ndim!=1 or not len(positions) or int(positions.max())>=ids.shape[1]:
            raise AssertionError("TRAIN prediction positions invalid")
        capture=[]
        def hook(_module,args):
            capture.append(args[0].detach().cpu().clone())
        h=native.lm_head.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                compressed=native(input_ids=ids,use_cache=False).logits[0,positions].float().detach().cpu()
        finally:
            h.remove()
        if len(capture)!=1:
            raise AssertionError("Native packed source does not expose final hidden")
        feature=capture[0][0,positions].float().detach().cpu()
        with torch.inference_mode():
            teacher_logits=teacher(input_ids=ids,use_cache=False).logits[0,positions].float().detach().cpu()
        if feature.ndim!=2 or feature.shape[0]!=len(positions) or compressed.shape!=teacher_logits.shape:
            raise AssertionError("Mismatched FP32 / native TRAIN features")
        if not torch.isfinite(feature).all() or not torch.isfinite(compressed).all():
            raise AssertionError("Nonfinite TRUE native source")
        differences=teacher_logits-compressed
        if not torch.isfinite(differences).all():
            raise AssertionError("Nonfinite calibrated teacher signal")
        if fit:
            fit_h.append(feature)
            fit_y.append(differences)
            counters["fit_tokens"]+=len(positions)
        else:
            hold_h.append(feature)
            hold_y.append(differences)
            hold_teacher.append(teacher_logits)
            hold_base.append(compressed)
            # Use -1 for free text with no single gold; no QA label leakage.
            hold_expected.extend([expected_gold if expected_gold is not None else -1]*len(positions))
            counters["holdout_tokens"]+=len(positions)
    for i,row in enumerate(texts):
        if row["kind"] not in ("wiki","news","supervised_chat"):
            raise AssertionError("Forbidden external split")
        token_ids=row["ids"][None].long()
        positions,_=positions_and_targets(row)
        collect(token_ids,positions,fit=i%TRAIN_TEXT_HELDOUT_MOD!=0)
    for i,pair in enumerate(pairs):
        if pair.get("split")!="TRAIN":
            raise AssertionError("VALID or DEV entered correction fitting")
        fit=i%TRAIN_PAIR_HELDOUT_MOD!=0
        for side in ("left","right"):
            sample=pair[side]
            ids=sample["prompt_ids"][None].long()
            gold=int(sample["first_expected_id"])
            collect(ids,torch.tensor([ids.shape[1]-1],dtype=torch.long),fit,expected_gold=gold)
            counters["fit_pair_prompts" if fit else "holdout_pair_prompts"]+=1
    if counters["holdout_pair_prompts"]!=16 or counters["fit_pair_prompts"]!=32:
        raise AssertionError("Precommitted 2:1 TRAIN pairs holdout changed")
    return (torch.cat(fit_h),torch.cat(fit_y),torch.cat(hold_h),
            torch.cat(hold_base),torch.cat(hold_teacher),hold_expected,counters)


def fit_calibrations(h,delta):
    if h.ndim!=2 or delta.ndim!=2 or h.shape[0]!=delta.shape[0] or h.shape[1]!=576:
        raise AssertionError("Wrong authentic compressed TRAIN calibration dimensions")
    # Center residual, then learn low-rank modes in LOGIT space, rather than
    # fitting FP32 weight residual on mismatched native hidden representations.
    bias=delta.mean(0)
    centered=delta-bias
    torch.manual_seed(214)
    U,S,V=torch.svd_lowrank(centered,q=8,niter=2)
    modes=V[:,:RANK].contiguous()
    scores=centered@modes
    # MAP ridge regression from *native* compressed hidden to four target
    # logit residual modes. Does NOT change model or original source bits.
    gram=h.T@h
    ridge=torch.eye(h.shape[1],dtype=torch.float32)*RIDGE
    projection=torch.linalg.solve(gram+ridge,h.T@scores)
    # Each physical rank-4 correction is (native_hidden @ B.T) @ A.T.
    A=modes.to(torch.float16).contiguous()
    B=projection.T.to(torch.float16).contiguous()
    bias=bias.to(torch.float16).contiguous()
    if not all(torch.isfinite(x).all() for x in (A,B,bias)):
        raise AssertionError("Invalid FP16 learned Green Tunnel factors")
    return {"bias_only":{"bias":bias},
            "rank4_bias":{"A":A,"B":B,"bias":bias}}


def heldout_train_stats(features,logits,teacher_logits,golds,factors,gain):
    corrected=logits+gain*factors["bias"].float()[None]
    if "A" in factors:
        corrected=corrected+gain*((features@factors["B"].float().T) @ factors["A"].float().T)
    t=teacher_logits.argmax(-1)
    pred=corrected.argmax(-1)
    correct_pairs=sum(int(int(p)==int(golds[i])) for i,p in enumerate(pred) if golds[i]>=0)
    t_agree=int((pred==t).sum())
    distinct=len(set(pred.tolist()))
    return {"TRAIN_holdout_teacher_token_agreement":t_agree,
            "TRAIN_holdout_correct_pairs":correct_pairs,
            "TRAIN_holdout_distinct_tokens":distinct,
            "TRAIN_holdout_position_count":len(pred)}


def choose_on_TRAIN_only(features,base,teacher,golds,variants):
    result={}
    for name,parts in variants.items():
        scored=[{"gain":gain,**heldout_train_stats(features,base,teacher,golds,parts,gain)}
                for gain in CANDIDATE_GAINS]
        # The gold answer value comes ONLY from the frozen TRAIN split; no
        # heldout public DEV or public VALID checkpoint choice is allowed.
        # Teacher agreement and diversity prevent a constant single token.
        selected=max(range(len(scored)),key=lambda i:(
            scored[i]["TRAIN_holdout_teacher_token_agreement"],
            scored[i]["TRAIN_holdout_correct_pairs"],
            scored[i]["TRAIN_holdout_distinct_tokens"],-i))
        result[name]={"selected_gain":CANDIDATE_GAINS[selected],
                      "training_only_selection":scored[selected],
                      "all_gain_candidates_TRAIN_only":scored}
    return result


def attach_head_only_correction(model,weights,scale):
    bias=weights["bias"].float()*scale
    a=weights["A"].float() if "A" in weights else None
    b=weights["B"].float() if "B" in weights else None
    if bias.ndim!=1 or bias.numel()!=model.config.vocab_size:
        raise AssertionError("Incorrect native output vocabulary")
    def output_patch(_module,args,out):
        corrected=out+bias
        if a is not None:
            corrected=corrected+scale*((args[0].float()@b.T)@a.T)
        return corrected
    return model.lm_head.register_forward_hook(output_patch)


def run(v206_path,v161_path,v193_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(214)
    cb=Path(v206_path).read_bytes()
    if len(cb)!=p.SOURCE_BYTES or hashlib.sha256(cb).hexdigest()!=SOURCE_V206_SHA256:
        raise AssertionError("Exact original v206 3/16 source required")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as td:
        root=Path(td)
        (root/"train_control").mkdir()
        native,_,_=make_model("baseline",SimpleNamespace(original=Path(v206_path),folder=root/"train_control"))
        teacher=original_teacher().cpu().eval()
        h,d,hv,bv,tv,golds,counts=freeze_native_teacher_train_inputs(teacher,native,tok)
        del teacher,native
        variants=fit_calibrations(h,d)
        training_policy=choose_on_TRAIN_only(hv,bv,tv,golds,variants)
        precommit={
            "experiment":"CALIBRI_v214_calibrated_native_output_distribution",
            "v206_SHA256":SOURCE_V206_SHA256,
            "original_unique_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
            "unmodified_v206_packed_bytes":len(cb),
            "all_in_source_budget_bpp":MAX_PHYSICAL_BPP,
            "TRAIN_source_counts":counts,
            "TRAIN_holdout_procedure":"every 5th frozen text record and every 3rd frozen opposing pair",
            "no_output_embedding_correction":True,
            "teacher_FP32_distribution_seen_only_on_PUBLIC_TRAIN":True,
            "TRAIN_svd_rank":RANK,"TRAIN_ridge":RIDGE,
            "TRAIN_gain_candidates":list(CANDIDATE_GAINS),
            "policy_selected_from_TRAIN_HOLDOUT_only":training_policy,
            "NO_DEV_VALID_TEST_OR_JESSICA_to_select_weights_or_gain":True,
            "NO_LIVE_MAIN_RENDER_SUPABASE_APK":True,
            "sealed_variants":{}
        }
        for mode in ("bias_only","rank4_bias"):
            path=out/f"v214_{mode}_preQA_sealed.safetensors"
            save_file(variants[mode],str(path))
            full=8*(len(cb)+path.stat().st_size)/p.FROZEN_SOURCE_135M_UNIQUE
            if full>MAX_PHYSICAL_BPP:
                raise AssertionError(f"Unpaid source bits: {mode}: {full}")
            precommit["sealed_variants"][mode]={
                "rank":0 if mode=="bias_only" else RANK,
                "physical_sidecar_bytes":path.stat().st_size,
                "physical_all_in_bpp":full,
                "SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected_gain_FROM_TRAIN_ONLY":training_policy[mode]["selected_gain"],
                "physically_signed_BEFORE_any_QA":True
            }
        save(out/"SEALED_v214_calibrated_heads_BEFORE_DEV_QA.json",precommit)
        print(json.dumps({"stage":"v214_train_only_signed",
                          "TRAIN_counts":counts,
                          "physical":precommit["sealed_variants"],
                          "train_only_choices":training_policy}),flush=True)
        original=extract_original(Path(v161_path))
        if len(original.blob)!=p.SOURCE_BYTES:
            raise AssertionError("Missing original CB59 baseline for speed")
        (root/"cb59").mkdir();(root/"cb59"/"source.cb59").write_bytes(original.blob)
        cb59,_,_=make_model("baseline",SimpleNamespace(original=root/"cb59"/"source.cb59",folder=root/"cb59"))
        (root/"v206").mkdir()
        baseline,_,_=make_model("baseline",SimpleNamespace(original=Path(v206_path),folder=root/"v206"))
        models={"CB206":baseline}
        handles={}
        for mode in ("bias_only","rank4_bias"):
            (root/mode).mkdir()
            model,_,_=make_model("baseline",SimpleNamespace(original=Path(v206_path),folder=root/mode))
            if any(w.ndim>=2 for w in model.parameters()):
                raise AssertionError("Native packed source replaced by dense model")
            factors=load_file(str(out/f"v214_{mode}_preQA_sealed.safetensors"))
            handles[mode]=attach_head_only_correction(
                model,factors,precommit["sealed_variants"][mode]["selected_gain_FROM_TRAIN_ONLY"])
            models["GT214_"+mode]=model
        try:
            dev=newly_frozen_public_dev_16(models,tok,v193_path,out)
            if dev["scores"]["CB206"]["machine_confirmed"]!=3:
                raise AssertionError("Baseline v206 changed between runs")
            speed={}
            cases=freeze_cases(all_cases())
            for mode in ("bias_only","rank4_bias"):
                name="GT214_"+mode
                native=speed_pairs({"CB59":cb59,"CB186":models[name]},tok,cases)
                speed[name]={"median_speed_ratio_vs_CB59":native["median_speed_ratio_vs_CB59"],
                             "8_paired_runs":native["8_paired_runs"]}
        finally:
            for hook in handles.values():hook.remove()
        result={
            "experiment":"v214_original_native_CB59_full_vocabulary_calibration",
            "completed":True,
            "physically_signed_before_QA":precommit,
            "original_FP32_machine_confirmed":dev["scores"]["FP32_existing_135M"]["machine_confirmed"],
            "v206_machine_confirmed":dev["scores"]["CB206"]["machine_confirmed"],
            "v214_scores":{key:{"machine_confirmed":value["machine_confirmed"],
                        "machine_wrong":value["machine_recognized_wrong"],
                        "ungraded":value["ungraded_not_known_wrong"]}
                        for key,value in dev["scores"].items() if key.startswith("GT214_")},
            "matched_native_CPU_speed":speed,
            "public_DEV16_score_is_retrospective_not_checkpoint_selection":True,
            "original_packed_model_unmodified":True,
            "no_live_or_Jessica_runs":True
        }
        save(out/"v214_REAL_native_QA_0p7bpp_and_CPU.json",result)
        print(json.dumps({"stage":"v214_real_NATIVE_QA",
                          "fp32":result["original_FP32_machine_confirmed"],
                          "v206":result["v206_machine_confirmed"],
                          "v214":result["v214_scores"],
                          "speed":speed,
                          "all_in_bpp":{k:v["physical_all_in_bpp"]
                                      for k,v in precommit["sealed_variants"].items()}}),flush=True)
        return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("source_v206",type=Path)
    ap.add_argument("original_CB59",type=Path)
    ap.add_argument("original_v193_QA",type=Path)
    ap.add_argument("output",type=Path)
    x=ap.parse_args()
    run(x.source_v206,x.original_CB59,x.original_v193_QA,x.output)
