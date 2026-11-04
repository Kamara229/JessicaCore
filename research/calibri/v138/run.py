"""End-to-end re-encoded tied embedding/head with genuine CB59-native C kernels.

Two pre-registered source-only candidate codebooks selected in v137's
held-out SOURCE-WEIGHT pilot. No teacher/gold logits used for fitting.
Record real first-token teacher KL and complete autoregressive outputs
on 16 PREVIOUSLY USED diagnostic questions (NOT independent v89 TEST).
The pinned original CB59 and v119 remain unmodified on disk and in GitHub.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v108.generation import build_chat
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE, MAX_SOURCE_BYTES
from v128.protocol import read_v119
from v132.protocol import all_examples
from .repack import KEY,checked_source,encoded_all,patch_one_matrix
from .protocol import choose_source_centers
from .scoring import ARMS,one_case,summarize_cases


@torch.inference_mode()
def teacher_oracle(tok,examples):
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:raise ValueError("teacher architecture unexpectedly changed")
    original_weight=teacher.model.embed_tokens.weight.detach().float().cpu().numpy().copy()
    logits=[]
    try:
        for pair,side,question,opposite,gold,foil in examples:
            ids=tok.encode(build_chat(tok,question),add_special_tokens=False)
            if not 1<len(ids)<160:raise ValueError("unexpected frozen paired prompt")
            true=teacher(input_ids=torch.tensor([ids],dtype=torch.long),use_cache=False)
            logits.append(true.logits[0,-1].detach().clone())
    finally:
        del teacher
    return original_weight,logits


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    pinned=read_v119(host)
    base=pinned.base
    if hashlib.sha256(base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original complete CB59 source must be frozen")
    archive=checked_source(base)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=tuple(all_examples())
    if len(cases)!=16:raise AssertionError("paired diagnostic set changed")
    source_weight,teacher_logits=teacher_oracle(tok,cases)
    handle=archive.handles[KEY]
    if handle.shape!=source_weight.shape:
        raise ValueError("FP32 teacher and original packed source different shapes")

    learned=choose_source_centers(source_weight,handle)
    candidates={"original_CB59":base}
    for key in ARMS[1:]:
        if key not in learned:raise ValueError("unregistered source dictionary")
        packed_codes=encoded_all(source_weight,handle.row_scales,learned[key])
        candidate=patch_one_matrix(base,learned[key],packed_codes)
        if len(candidate)!=len(base) or len(candidate)>MAX_SOURCE_BYTES:
            raise ValueError("physical source no longer under 0.7bpw")
        if hashlib.sha256(candidate).hexdigest()==ORIGINAL_CB59_SHA:
            raise AssertionError("candidate re-encoding produced no new bits")
        candidates[key]=candidate

    # After fitting has terminated, gold answers enter only post-generation
    # scoring. Neither gold nor teacher logits participate in source fitting.
    result={
        "experiment":"v138 bit-exact reencoded K32V8 tied embedding and lm_head",
        "parent_v119_sha256":hashlib.sha256(host.read_bytes()).hexdigest(),
        "original_CB59_sha256":ORIGINAL_CB59_SHA,
        "real_original_unique_parameters":TOTAL_UNIQUE,
        "source_bytes_by_arm":{a:len(blob) for a,blob in candidates.items()},
        "all_in_bpp_by_arm":{a:8*len(blob)/TOTAL_UNIQUE for a,blob in candidates.items()},
        "original_source_remains_unmodified":True,
        "all_non_embedding_trained_matrices_remain_byte_identical":True,
        "lm_head_is_tied_to_embedding":True,
        "only_predeclared_v137_source_weight_centroids":True,
        "genuine_full_model_native_C_CPU":True,
        "locked_independent_v89_TEST_unopened":True,
        "sixteen_prior_counterfactual_DIAGNOSTIC_not_fresh_TEST":True,
        "no_training_on_gold_answers_or_teacher_logits":True,
        "no_GREEN_or_Jessica_or_LIVE_change":True,
        "physical_archives_SHA256":{},
        "arms":{},
    }
    for name,data in candidates.items():
        filename=out/("v138_"+name+".cb59")
        filename.write_bytes(data)
        if filename.read_bytes()!=data:raise ValueError("physical candidate not persisted")
        CB59(filename.read_bytes()) # independently verify CRC, SHA, aliases
        result["physical_archives_SHA256"][name]=hashlib.sha256(data).hexdigest()

    with TemporaryDirectory(dir=out) as temporary:
        scratch=Path(temporary)
        for arm in ARMS:
            storage=scratch/arm
            storage.mkdir()
            model,metadata,_=make_model("baseline",SimpleNamespace(
                original=out/("v138_"+arm+".cb59"),folder=storage))
            if not metadata["zero_dense_pretrained_2D_hot_weights"]:
                raise AssertionError("native source accidentally loaded dense")
            if metadata["physical_complete_model_bytes"]!=len(base):
                raise ValueError("candidate byte budget no longer matches original")
            entries=[]
            for row,t in zip(cases,teacher_logits):
                pair,side,question,opposite,gold,foil=row
                entries.append(one_case(model,tok,question,opposite,gold,foil,
                                        pair,side,t))
            del model
            result["arms"][arm]=entries
    result["score"]=summarize_cases(result["arms"])
    destination=out/"v138_repacked_native_CB59_quality.json"
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    compact={arm:{k:v for k,v in record.items() if k!="whole_reply_observation"}
             for arm,record in result["score"].items() if isinstance(record,dict)}
    print(json.dumps({"stage":"v138_REAL_NATIVE_REPACK_AND_QA",
          "archive_bpp":result["all_in_bpp_by_arm"],
          "scores":compact,
          "full_reply_metrics":{arm:result["score"][arm]["whole_reply_observation"]
                                for arm in ARMS},
          "report":str(destination)},
          ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("host",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.host,a.out)
