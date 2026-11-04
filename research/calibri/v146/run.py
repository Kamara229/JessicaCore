"""Preregistered untrained validation QA: FP32, CB59, paid full, paid head50.

We do not open official v89 TEST. No codebook/gain fitted from SciQ/BoolQ;
the two existing gains were fixed in v144 and confirmed in v145. Physical
CB140 source (0.6942 bpp) is frozen. We compare next-label likelihood,
not generated natural language, and report the FP32 teacher baseline.
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
from v140.codec import unpack
from v141.run import ORIGINAL_V140_SHA
from v144.operators import Paired

from .protocol import DOMAINS,ARMS,SELECTED_GAINS,validate
from .sources import load_valid
from .scoring import label_tokens,encode_prompt,measure_one,summary


@torch.inference_mode()
def score_model(model,tokenizer,tasks):
    answers=[]
    for task in tasks:
        tokens=label_tokens(tokenizer,task.options)
        prompt=encode_prompt(tokenizer,task.prompt)
        outcome=measure_one(model,prompt,tokens,task.correct_index)
        outcome["dataset_row_index"]=task.index
        outcome["prompt_SHA256"]=task.prompt_sha
        outcome["correct_choice_index_ONLY_FOR_POST_INFERENCE_SCORING"]=task.correct_index
        outcome["all_choice_tokens_single_id"]=True
        outcome["prompt_token_count"]=int(prompt.shape[1])
        answers.append(outcome)
    return answers


def run(archive,out):
    torch.set_num_threads(2)
    validate()
    out.mkdir(parents=True,exist_ok=True)
    archive_bytes=archive.read_bytes()
    physical=unpack(archive_bytes)
    if physical.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not frozen physically paid original v140 CB140 source")
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original complete CB59 bytes changed")
    tasks,sources=load_valid()
    if tuple(tasks)!=DOMAINS:
        raise ValueError("unregistered validation benchmark")

    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    data={
        "stage":"v146_SCIQ_BOOLQ_VALID_NEXT_LABEL_CHOICE_DIAGNOSTIC",
        "immutable_source_SHA256":physical.sha256,
        "original_CB59_SHA256":ORIGINAL_CB59_SHA,
        "all_in_CB140_bytes":physical.size,
        "all_in_CB140_bpp":8*physical.size/TOTAL_UNIQUE,
        "two_new_unseen_VALIDATION_slices_not_TEST":True,
        "source_provenance":sources,
        "no_codebook_gain_or_model_training":True,
        "gain_0_0p5_SELECTED_IN_prior_v144_NOT_retuned":True,
        "no_official_frozen_v89_TEST_access":True,
        "no_Jessica_chat_test_or_LIVE_changes":True,
        "forced_next_label_accuracy_NOT_free_response_accuracy":True,
        "results":{arm:{} for arm in ARMS},
    }

    teacher=original_teacher()
    try:
        if len(teacher.model.layers)!=30:
            raise ValueError("wrong reference model geometry")
        for domain in DOMAINS:
            data["results"]["FP32"][domain]=score_model(teacher,tokenizer,tasks[domain])
    finally:
        del teacher

    with TemporaryDirectory(dir=out) as temporary:
        root=Path(temporary)
        base=root/"native.cb59"
        base.write_bytes(physical.original)
        model,meta,_=make_model("baseline",SimpleNamespace(original=base,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("frozen CB59 original weights unexpectedly dense")
        paid=Paired(model,physical,root)
        try:
            for arm in ARMS[1:]:
                paid.configure(*SELECTED_GAINS[arm])
                for domain in DOMAINS:
                    data["results"][arm][domain]=score_model(
                        model,tokenizer,tasks[domain])
        finally:
            paid.restore()

    data["quality"]=summary(data["results"])
    path=out/"v146_sciq_boolq_choice_validation.json"
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":data["stage"],
        "quality":data["quality"],
        "physical_bpp":data["all_in_CB140_bpp"],
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("physically_paid_CB140",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.physically_paid_CB140,args.out)
