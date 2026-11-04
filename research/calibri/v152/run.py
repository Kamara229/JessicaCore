"""Causal 30-block oracle localization from ORIGINAL frozen packed CB59.

This is not a deployable candidate: every borrowed original FP32 2D layer
costs extra megabytes not counted in CB59 0.667bpp. We make this deliberate
cost explicit. NO parameters are fitted, compressed or saved as a model.

Goal: identify which original operator(s) must be redesigned in the
physically paid 0.7-bpp source itself rather than adding tiny head adapters
that improve next-token NLL but break SciQ/BoolQ semantics and diversity.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v147.codec import unpack as unpack_CB147
from v108.generation import build_chat
from .protocol import (check,locked_questions,LAYERS,ARMS,
                       SOURCE_CB147_SHA,ORIGINAL_V140_SHA)
from .oracles import replace_one_layer
from .scoring import comparison,aggregate


@torch.inference_mode()
def one_last_logits(model,ids):
    logits=model(input_ids=ids,use_cache=False).logits[0,-1].detach().clone()
    if logits.shape!=(49152,) or not torch.isfinite(logits).all():
        raise ValueError("invalid original source next-token logits")
    return logits


def run(physical_cb147,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    source=unpack_CB147(physical_cb147.read_bytes())
    if source.sha256!=SOURCE_CB147_SHA or source.paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not the physically pinned parent archive")
    if hashlib.sha256(source.paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("wrong original CB59 source")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    prepared=[]
    for name,side,question,opposite,gold,foil in locked_questions():
        prompt=build_chat(tokenizer,question)
        token_ids=tokenizer.encode(prompt,add_special_tokens=False)
        if not 1<len(token_ids)<160:
            raise ValueError("diagnostic source prompt changed")
        prepared.append((name,side,question,
                         torch.tensor([token_ids],dtype=torch.long)))
    teacher=original_teacher()
    if len(teacher.model.layers)!=LAYERS:
        raise ValueError("incorrect frozen FP32 teacher")
    teacher_logits=[one_last_logits(teacher,ids) for _,_,_,ids in prepared]
    dataset_labels=[{"name":name,"side":side,"question":question}
                    for name,side,question,ids in prepared]
    observed={key:[] for key in ARMS}
    eos=tokenizer.eos_token_id
    if eos is None:raise ValueError("missing frozen EOS")
    for logits in teacher_logits:
        observed["FP32_teacher"].append(comparison(logits,logits,eos))

    borrowed_info={}
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        raw=root/"original_CB59_native.cb59"
        raw.write_bytes(source.paid.original)
        model,meta,_=make_model("baseline",SimpleNamespace(original=raw,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("base native packed student has dense pretrained weights")
        if len(model.model.layers)!=LAYERS:
            raise ValueError("base native packed layer count wrong")
        original_layers=tuple(model.model.layers)
        with torch.inference_mode():
            for _,_,_,ids in prepared:
                idx=len(observed["genuine_packed_CB59"])
                prediction=one_last_logits(model,ids)
                observed["genuine_packed_CB59"].append(
                    comparison(teacher_logits[idx],prediction,eos))
            for depth in range(LAYERS):
                arm=f"oracle_FP32_layer_{depth:02d}"
                with replace_one_layer(model,teacher,depth) as borrowed:
                    borrowed_info[arm]=borrowed
                    for case_index,(_,_,_,ids) in enumerate(prepared):
                        trial=one_last_logits(model,ids)
                        observed[arm].append(
                            comparison(teacher_logits[case_index],trial,eos))
                if any(x is not y for x,y in zip(model.model.layers,original_layers)):
                    raise AssertionError("one borrowed FP32 oracle was not restored")
        if any(param.ndim>=2 for param in model.parameters()):
            raise AssertionError("teacher matrix survived removal from CB59 control")
    del teacher
    summary=aggregate(observed)
    report={
        "stage":"v152_NONDEPLOYABLE_FP32_30BLOCK_CAUSAL_LOCALIZATION",
        "genuine_compressed_source":"CB59 from pinned original complete CB147",
        "physically_packed_CB59_sha256":ORIGINAL_CB59_SHA,
        "parent_paid_CB147_sha256":SOURCE_CB147_SHA,
        "genuine_CB59_only_bpp":8*len(source.paid.original)/TOTAL_UNIQUE,
        "ALL_single_block_FP32_oracles_are_NOT_compressed":True,
        "all_30_oracle_single_layer_bytes_unpaid":borrowed_info,
        "strictly_fixed_prior_diagnostic_probes":dataset_labels,
        "no_new_training":True,
        "no_QA_or_next_token_source_heldout_validation":True,
        "no_official_v89_independent_TEST":True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes":True,
        "raw_FP32_KL_and_argmax_controls":observed,
        "summary":summary,
        "cannot_claim_restored_language_from_FP32_oracle":True,
        "candidate_next_step":"Evaluate sensitivity-aware, physically paid layer-wise compression using ORIGINAL source TRAIN FIT then disjoint TRAIN VALID; do not smuggle oracle matrices into source bpp.",
    }
    target=out/"v152_30block_FP32_oracle_ranking.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report["stage"],
        "CB59_KL":summary["controls"]["genuine_packed_CB59"]["mean_FP32_teacher_KL"],
        "most_responsive_5_single_FP32_blocks":
            summary["all_30_single_layer_FP32_oracle_ablation_ranking"][:5],
        "least_responsive_5_single_FP32_blocks":
            summary["all_30_single_layer_FP32_oracle_ablation_ranking"][-5:],
        "FP32_oracles_break_0_7bit":True,
        "report":str(target),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_paid_CB147",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.original_paid_CB147,a.out)
