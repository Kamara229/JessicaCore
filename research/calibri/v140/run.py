"""Build genuinely prepaid CB140, run direct-native C tied-embedding/head QA.

Original source locked and unmodified. The v119 paid adapter is NOT included;
this is a compact alternative under 0.7 bits/ORIGINAL unique parameter.
This previously used paired corpus is DIAGNOSTIC, not independent v89 TEST.
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
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v128.protocol import read_v119
from v132.protocol import all_examples
from v138.run import teacher_oracle
from v138.scoring import one_case
from .codec import pack,unpack,report
from .train import fit_from_original_pretrained,assign_full_extra_bits
from .runtime import install
from .scoring import summarize_native


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    paid_v119=read_v119(host)
    base=paid_v119.base
    if hashlib.sha256(base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original entire CB59 source not pinned")
    origin=CB59(base)
    tied=origin.handles["model.embed_tokens.weight"]
    if origin.handles["lm_head.weight"] is not tied:
        raise ValueError("original tied physical source not shared")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=tuple(all_examples())
    if len(cases)!=16:
        raise ValueError("not original paired control corpus")
    weights,teacher_logits=teacher_oracle(tok,cases)
    residual,index_codes=fit_from_original_pretrained(weights,tied)
    plane,source_rmse=assign_full_extra_bits(weights,tied,residual,index_codes)
    updated=pack(base,residual,plane)
    if len(updated)>MAX_SOURCE_BYTES:
        raise ValueError("physical CB140 source exceeds 0.7-bpp")
    path=out/"CALIBRI_v140_paid_tied_embedding_6bit.cb140"
    path.write_bytes(updated)
    physical=unpack(path.read_bytes())
    if physical.original!=base or physical.sha256!=hashlib.sha256(updated).hexdigest():
        raise ValueError("loss of original immutable CB59 base")
    result={
        "experiment":"v140 full prepaid 1bit tied embedding+head residual native CPU",
        "previous_source_v119_SHA":hashlib.sha256(host.read_bytes()).hexdigest(),
        "new_physical_source":report(updated),
        "physical_file_name":path.name,
        "physical_file_SHA256":physical.sha256,
        "original_CB59_file_is_unchanged":True,
        "v119_paid_sidecar_is_NOT_carried_into_new_archive":True,
        "original_source_FP32_teacher_used_only_for_SOURCE_FIT_and_EVAL":True,
        "no_answer_gold_used_before_generation":True,
        "no_v89_independent_TEST_opened":True,
        "no_Jessica_LIVE_Render_Supabase_APK_change":True,
        "input_embedding_and_output_head_share_one_physical_bitplane":True,
        "source_embedding_reconstruction":source_rmse,
        "arms":{"original_CB59":[],"paid_CB140_6bit_tied":[]},
    }
    with TemporaryDirectory(dir=out) as temporary:
        folder=Path(temporary)
        orig=folder/"original.cb59";orig.write_bytes(base)
        model,metadata,_=make_model("baseline",SimpleNamespace(original=orig,folder=folder))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise AssertionError("original source reappeared as dense")
        original_embedding=model.model.embed_tokens
        original_head=model.lm_head
        kernel=install(model,physical,folder)
        improved_embedding=model.model.embed_tokens
        improved_head=model.lm_head
        if kernel.rows!=49152 or kernel.dim!=576:
            raise ValueError("wrong C native geometry")
        for idx,(case,teacher_logit) in enumerate(zip(cases,teacher_logits)):
            pair,side,question,opposite,gold,foil=case
            # Alternate arm order to avoid systematic thermal/time confounds.
            order=("original_CB59","paid_CB140_6bit_tied") if idx%2==0 else (
                "paid_CB140_6bit_tied","original_CB59")
            for arm in order:
                use_paid=arm=="paid_CB140_6bit_tied"
                model.model.embed_tokens=improved_embedding if use_paid else original_embedding
                model.lm_head=improved_head if use_paid else original_head
                if model.model.embed_tokens.handle is not model.lm_head.handle:
                    raise AssertionError("tied compressed source identity deviated")
                with torch.inference_mode():
                    item=one_case(model,tok,question,opposite,gold,foil,
                                  pair,side,teacher_logit)
                result["arms"][arm].append(item)
        # Make the original branch canonical again.
        model.model.embed_tokens=original_embedding
        model.lm_head=original_head
    result["score"]=summarize_native(result["arms"])
    target=out/"v140_paid_native_6bit_full_model.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v140_PAID_NATIVE_0_7BIT_SOURCE_COMPLETE",
        "physical":result["new_physical_source"],
        "full_source_embedding_RMSE":source_rmse,
        "score":result["score"],
        "full_report":str(target)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("host",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.host,args.out)
