#!/usr/bin/env python3
"""CALIBRI v43: actual 12-MLP physical rate-distortion curve with TRAIN-only calibration.

On REAL pretrained Pythia-14m use WikiText TRAIN tokens 0:2048 for
all 12 MLP row-bit allocation, FP16 row-scale calibration, and the v39
joint full-model NLL optimizer. Use disjoint TRAIN tokens 2048:4096
ONLY ONCE to decide whether to keep the final optimizer candidate.
Only then persist all 12-MLP byte bundles and read TWO new public corpora.
The one-shot internal gate cannot be used to select individual swaps.
"""
from __future__ import annotations

import argparse,hashlib,json,os
from pathlib import Path

import numpy as np
import torch

from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from all_mlp_agnews_v36 import package,unpack_bundle,module_identity
from direct_tiles_v33 import encode as encode_tiles,decode as decode_tiles
from fixedbit_v15 import encode as encode_int4,decode as decode_int4
from calibrated_rows_v35 import calibrated_encode
from pareto_fixed_highbits_v43 import trained_fixedbit,smoke as fixedbit_smoke
from joint_train_loss_v39 import search_joint_train_nll

TRAIN_A=2048
TRAIN_B=2048
CONTEXT=128
NEW_TEST_TOKENS=2048
GLOBAL_BYTES=343840
MLP_WEIGHTS=786432
EXTERNAL_DATASETS=(
    ("NEW_OpenBookQA_validation","allenai/openbookqa","main","validation[:128]"),
    ("NEW_SNLI_validation","stanfordnlp/snli",None,"validation[:128]"),
)
GATE_MIN_IMPROVEMENT=1e-6


def new_first_use_tokens(tokenizer, dataset_name, config, subset, revision, token_count):
    """First-use fixed public question/inference texts, WITHOUT ground-truth labels."""
    from datasets import load_dataset
    opts={"split":subset,"revision":revision}
    ds=load_dataset(dataset_name,config,**opts) if config else load_dataset(dataset_name,**opts)
    parts=[]
    for row in ds:
        if dataset_name=="allenai/openbookqa":
            choices=row.get("choices") or {}
            text=("Fact: "+str(row.get("fact1","")).strip()+
                  "\nQuestion: "+str(row["question_stem"]).strip()+
                  "\nChoices: "+"; ".join(str(label)+") "+str(choice)
                     for label,choice in zip(choices.get("label",[]),choices.get("text",[]))))
        elif dataset_name=="stanfordnlp/snli":
            text=("Premise: "+str(row["premise"]).strip()+
                  "\nHypothesis: "+str(row["hypothesis"]).strip())
        else:
            raise AssertionError("not preregistered new domain")
        if text.strip():parts.append(text)
    if not parts:raise AssertionError("new-domain test contains no examples")
    ids=tokenizer("\n\n".join(parts),return_tensors="pt",
                  add_special_tokens=False).input_ids[0,:token_count]
    if len(ids)!=token_count:
        raise AssertionError("new-domain frozen 2048-token excerpt not available")
    return ids,len(parts)


def evaluate_dense(model, originals, variants, corpora):
    result={}
    try:
        with torch.inference_mode():
            for tag,recon in variants.items():
                for n,source in originals.items():
                    model.get_submodule(n).weight.copy_(source)
                if recon is not None:
                    if set(recon)!=set(originals):
                        raise AssertionError("all 12 source MLP modules must be present")
                    for n,q in recon.items():
                        if q.shape!=tuple(originals[n].shape) or not np.isfinite(q).all():
                            raise AssertionError("invalid decoded physical source MLP")
                        model.get_submodule(n).weight.copy_(torch.from_numpy(q))
                result[tag]={}
                for corpus,tokens in corpora.items():
                    p=ppl_eval(model,tokens,CONTEXT)
                    result[tag][corpus]=p
                    print(json.dumps({"variant":tag,"corpus":corpus,
                                      "PPL":p["ppl"],"NLL":p["nll"]}),flush=True)
    finally:
        with torch.no_grad():
            for n,source in originals.items():
                model.get_submodule(n).weight.copy_(source)
    return result


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi

    if (args.train_tokens!=TRAIN_A+TRAIN_B or
            args.dev_tokens!=1024 or args.new_tokens!=NEW_TEST_TOKENS or
            args.context!=CONTEXT):
        raise AssertionError("frozen v41 test/calibration split altered")
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("real selected CPU AVX2 dispatcher required")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    fixedbit_smoke()

    api=HfApi()
    model_revision=api.model_info(MODEL).sha
    wiki_revision=api.dataset_info(DATASET).sha
    # Metadata lookup is safe; NONE of the three EVALUATION texts is read until final
    # TRAIN-A optimizer, TRAIN-B gate and immutable binary files complete.
    external_revisions={name:api.dataset_info(repository).sha
                        for name,repository,config,split in EXTERNAL_DATASETS}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=model_revision)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=model_revision,torch_dtype=torch.float32)
    model.eval()

    modules=module_selection(model)
    names=sorted(modules,key=module_identity)
    originals={n:modules[n].weight.detach().clone() for n in names}
    weights={n:originals[n].cpu().numpy().astype(np.float32).copy() for n in names}
    if len(names)!=12 or sum(q.size for q in weights.values())!=MLP_WEIGHTS:
        raise AssertionError("unexpected REAL Pythia-14m architecture/MLP size")

    all_train=load_token_ids(tokenizer,"train",wiki_revision,TRAIN_A+TRAIN_B)
    train_a=all_train[:TRAIN_A]
    train_b=all_train[TRAIN_A:]
    assert len(train_a)==len(train_b)==2048
    hash_a=hashlib.sha256(train_a.numpy().tobytes()).hexdigest()
    hash_b=hashlib.sha256(train_b.numpy().tobytes()).hexdigest()
    if hash_a==hash_b:
        raise AssertionError("distinct TRAIN token windows are identical")
    with torch.inference_mode():
        original_activations=calibrate(model,modules,train_a,CONTEXT)
    if set(original_activations)!=set(names):
        raise AssertionError("TRAIN-A activation capture missing MLP")
    x={n:v.detach().cpu().numpy().astype(np.float32) for n,v in original_activations.items()}
    if any(v.shape!=(TRAIN_A,weights[n].shape[1]) for n,v in x.items()):
        raise AssertionError("TRAIN-A MLP input calibration shapes changed")

    # Never examine TRAIN-B, DEV or SQuAD when fitting the quantizer.
    local={}
    moments_by_module={}
    for n in names:
        source=weights[n]
        moments=(x[n]*x[n]).mean(axis=0,dtype=np.float32)
        moments_by_module[n]=moments
        local[n],_=encode_tiles(source,moments,int(source.size*3.5/8),
                                br=1,bc=source.shape[1])
    local_cal={n:calibrated_encode(weights[n],x[n],local[n])[0]
               for n in names}
    raw_joint,cal_joint,search=search_joint_train_nll(
        names,weights,x,model,originals,train_a,CONTEXT,local)
    if any(len(package(blobs,1))!=GLOBAL_BYTES for blobs in (local,local_cal,raw_joint,cal_joint)):
        raise AssertionError("TRAIN-A model compressed bundles do not satisfy exact physical bytes")

    # ONE independent TRAIN-B score for the final source-local and the
    # TRAIN-A-optimized proposal. Never use TRAIN-B within search/ranking.
    internal=evaluate_dense(model,originals,{
        "local_2048_trainA":{n:decode_tiles(blob) for n,blob in local_cal.items()},
        "joint_trainA_candidate":{n:decode_tiles(blob) for n,blob in cal_joint.items()},
    },{"WikiText_TRAIN_B_internal_gate":train_b})
    b_base=internal["local_2048_trainA"]["WikiText_TRAIN_B_internal_gate"]["nll"]
    b_joint=internal["joint_trainA_candidate"]["WikiText_TRAIN_B_internal_gate"]["nll"]
    accepted=b_joint < b_base-GATE_MIN_IMPROVEMENT
    decision="accept_joint" if accepted else "reject_joint_use_local"
    selected=cal_joint if accepted else local_cal
    print(json.dumps({
        "TRAIN_A_local_NLL":search["original_baseline_joint_train_NLL"],
        "TRAIN_A_joint_NLL":search["selected_joint_train_NLL"],
        "TRAIN_B_local_NLL":b_base,"TRAIN_B_joint_NLL":b_joint,
        "train_only_single_GATE":decision,
    }),flush=True)

    # Materialize every real physical variant and read back it with checksum.
    # Frozen Pareto points: physical ~3.498bpw, 4.081bpw, 5.081bpw,
    # 6.081bpw. THE EXPERIMENT MEASURES actual rate incl all overhead,
    # not nominal code symbol width. CF15 controls unchanged v15 codec,
    # recalibrated with v35 fixed scale grid + only TRAIN-A moments.
    variants={
        "local_uncalibrated":(1,local),
        "local_trainA_calibrated":(1,local_cal),
        "joint_trainA_calibrated_UNGATED":(1,cal_joint),
        "joint_trainA_calibrated_GATED":(1,selected),
    }
    highbit_train={}
    for bits in (4,5,6):
        baseline={}
        traincal={}
        proxy_before=0.
        proxy_after=0.
        changed=0
        for n in names:
            b,adj,meta=trained_fixedbit(weights[n],moments_by_module[n],bits)
            baseline[n]=b
            traincal[n]=adj
            proxy_before+=meta["TRAIN_diagonal_weighted_MSE_before"]
            proxy_after+=meta["TRAIN_diagonal_weighted_MSE_after"]
            changed+=meta["rows_calibrated_changed"]
        variants[f"CF15_INT{bits}_raw"]=(2,baseline)
        variants[f"CF15_INT{bits}_TRAIN_calibrated"]=(2,traincal)
        highbit_train[str(bits)]={
            "source_bitwidth":bits,
            "TRAIN_moment_weighted_MSE_before":proxy_before,
            "TRAIN_moment_weighted_MSE_after":proxy_after,
            "rows_changed":changed,
            "scale_grid":"the EXACT predeclared v35 FP16 scale factor grid",
        }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    saved={}
    decoded={}
    for tag,(mode,components) in variants.items():
        archive=package(components,mode)
        dest=output.parent/(tag+".bin")
        dest.write_bytes(archive)
        actual=dest.read_bytes()
        if archive!=actual:
            raise AssertionError("encoded physical disk did not match saved bytes")
        restored=unpack_bundle(actual,mode)
        if set(restored)!=set(names):
            raise AssertionError("physical archive missing modules")
        actual_bytes=dest.stat().st_size
        if mode==1 and actual_bytes!=GLOBAL_BYTES:
            raise AssertionError("unfair physical mixed-rate comparison")
        if mode==2 and actual_bytes<=GLOBAL_BYTES:
            raise AssertionError("higher-bit standard physically coded reference should be above 3.5bpw")
        if mode==2:
            bits=int(tag.split("_")[1].replace("INT",""))
            # Actual serialized CF15-CRC per-matrix header, packed symbols,
            # FP16 scales, plus CB36 outer header+descriptors and SHA.
            expected=401224+(bits-4)*(MLP_WEIGHTS//8)
            if actual_bytes!=expected:
                raise AssertionError("actual physical high-bit Pareto size changed")
        saved[tag]={
            "real_disk_bytes":actual_bytes,
            "real_sha256":hashlib.sha256(actual).hexdigest(),
            "bpw_for_12_MLP_only":8*actual_bytes/MLP_WEIGHTS,
            "module_count":len(restored),
        }
        decoded[tag]={n:(decode_int4(v) if mode==2 else decode_tiles(v))
                      for n,v in restored.items()}
    if (saved["joint_trainA_calibrated_GATED"]["real_sha256"] !=
       saved["joint_trainA_calibrated_UNGATED" if accepted else "local_trainA_calibrated"]["real_sha256"]):
        raise AssertionError("internal holdout gating not reflected in physical file")

    # FIRST time any NEW text is read, AFTER frozen training, internal
    # one-shot B gate and actual persisted/reloaded physical source weights.
    known_dev=load_token_ids(tokenizer,"validation",wiki_revision,args.dev_tokens)
    corpora={"WikiText2_development":known_dev}
    external_details={}
    for tag,repository,config,split in EXTERNAL_DATASETS:
        ids,items=new_first_use_tokens(
            tokenizer,repository,config,split,external_revisions[tag],args.new_tokens)
        corpora[tag]=ids
        external_details[tag]={
            "dataset":repository,"config":config,"split":split,
            "dataset_revision":external_revisions[tag],
            "texts_in_slice":items,
            "token_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "tokens":len(ids),
        }
    if len(corpora)!=3:
        raise AssertionError("both preregistered first-use corpora required")
    quality=evaluate_dense(model,originals,{"original_teacher":None,**decoded},corpora)
    baseline=quality["local_trainA_calibrated"]
    raw_candidate=quality["joint_trainA_calibrated_UNGATED"]
    chosen=quality["joint_trainA_calibrated_GATED"]
    diffs={d:{
        "raw_joint_minus_local_PPL":raw_candidate[d]["ppl"]-baseline[d]["ppl"],
        "gated_joint_minus_local_PPL":chosen[d]["ppl"]-baseline[d]["ppl"],
        "gated_strategy":decision,
    } for d in corpora}
    data={
        "experiment":"CALIBRI_v43_measured_real_12_MLP_Pareto_3p5_4_5_6_bits_first_use_domains",
        "source_model":MODEL,
        "source_revision":model_revision,
        "source_MLP_sha256":{n:hashlib.sha256(w.tobytes()).hexdigest() for n,w in weights.items()},
        "real_source_MLP_parameter_count":MLP_WEIGHTS,
        "datasets":{"WikiText2":{"name":DATASET,"revision":wiki_revision},
                    "two_new_untouched_external":external_details},
        "TRAIN_A":{"start":0,"end_exclusive":TRAIN_A,"sha256":hash_a},
        "TRAIN_B":{"start":TRAIN_A,"end_exclusive":TRAIN_A+TRAIN_B,"sha256":hash_b},
        "WikiText2_DEV_sha256":hashlib.sha256(known_dev.numpy().tobytes()).hexdigest(),
        "external_dataset_tokens_SHA256":{k:info["token_sha256"] for k,info in external_details.items()},
        "split_epoch":{"TRAIN_A_selection_uses_A_only":True,
                       "TRAIN_B_checked_only_once_after_complete_TRAIN_A_search":True,
                       "both_new_corpora_loaded_after_binary_freeze":True},
        "selection_trainA":search,
        "internal_gate_trainB":{
            "local_NLL":b_base,"ungated_joint_NLL":b_joint,
            "joint_improvement_required":GATE_MIN_IMPROVEMENT,
            "joint_accepted":accepted,"decision":decision,
            "TRAIN_B_scores":internal,
        },
        "physical_MLP_bundles":saved,
        "higher_bits_trainA_calibration":highbit_train,
        "physical_rate_quality_curve":{
            tag:{
                "real_MLP_only_bytes":info["real_disk_bytes"],
                "actual_bpw_including_CB36_and_row_scales":info["bpw_for_12_MLP_only"],
                "PPL_by_corpus":{corpus:score["ppl"] for corpus,score in quality[tag].items()},
                "is_training_calibrated":tag.endswith("TRAIN_calibrated") or tag=="joint_trainA_calibrated_GATED",
            }
            for tag,info in saved.items()
        },
        "full_model_decoded_PPL":quality,
        "quality_delta_against_equal_rate_local":diffs,
        "new_corpora_summary":{
            "primary_candidate":"joint_trainA_calibrated_GATED",
            "rate_matched_comparator":"local_trainA_calibrated",
            "success_count":sum(diffs[name]["gated_joint_minus_local_PPL"]<0.0
                                for name,*_ in EXTERNAL_DATASETS),
            "new_corpus_count":len(EXTERNAL_DATASETS),
            "improved_each":{name:diffs[name]["gated_joint_minus_local_PPL"]<0.0
                            for name,*_ in EXTERNAL_DATASETS},
        },
        "environment":{"torch":str(torch.__version__),
                       "cpu_capability":str(torch._C._get_cpu_capability()),
                       "onednn_max_requested":"AVX2",
                       "threads":torch.get_num_threads(),
                       "determinism":torch.are_deterministic_algorithms_enabled()},
        "limitations":[
          "TRAIN-B used only ONCE to accept or reject the single final candidate. It is no longer an untouched heldout corpus after this session.",
          "Only 2048+2048 training tokens and TWO 2048-token fixed first-use samples; exploratory PPL, not broad benchmarks.",
          "No inference speed from packed weights: dense decoders used for full model PPL.",
          "Actual CB36 files only store 12 source MLP matrices; NOT a full model checkpoint.",
          "3/4-bit mixed codec compared with fixed signed INT4/5/6 CF15 raw and train-moment calibrated. Different codec families and training objective mean only observed physical rate-quality points, NOT optimized Pareto envelope.",
          "Nominal 4/5/6 bit symbols have physical overhead and actual 4.081/5.081/6.081 bpw including CB36/CRC/scales, NOT strict <=4/5/6 bpw.",
          "No multidimensional trained weights, QMoE 0.8bpw or deployable runtime demonstrated.",
        ],
    }
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"stage":"FINAL_MEASURED",
       "train_A_local_NLL":search["original_baseline_joint_train_NLL"],
       "train_A_joint_NLL":search["selected_joint_train_NLL"],
       "train_B_local_NLL":b_base,"train_B_joint_NLL":b_joint,
       "gate":decision,
       "file_bytes":{k:x["real_disk_bytes"] for k,x in saved.items()},
       "PPL":{k:{j:round(s["ppl"],6) for j,s in q.items()} for k,q in quality.items()},
       "new_corpora_primary_deltas":{name:diffs[name] for name,*_ in EXTERNAL_DATASETS},
       "new_corpora_improved":data["new_corpora_summary"]["success_count"],
       "physical_rate_quality":{k:{"bytes":v["real_MLP_only_bytes"],
                                   "bpw":v["actual_bpw_including_CB36_and_row_scales"],
                                   "PPL":v["PPL_by_corpus"]}
                                for k,v in data["physical_rate_quality_curve"].items()},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/calibri_v43_physical_pareto.json")
    parser.add_argument("--train-tokens",type=int,default=4096)
    parser.add_argument("--dev-tokens",type=int,default=1024)
    parser.add_argument("--new-tokens",type=int,default=2048)
    parser.add_argument("--context",type=int,default=128)
    run(parser.parse_args())
