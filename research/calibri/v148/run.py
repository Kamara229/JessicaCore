"""New 16+16 SciQ/BoolQ VALID forced-choice without fitting any model.

This is a checkpoint read-only test, performed after real v147 construction
and after v145/v146 results. Vocab prior fitted on original TRAIN FIT only.
One frozen 11,721,959-byte CB147 file, no v89 TEST and no LIVE changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from time import perf_counter

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v141.run import ORIGINAL_V140_SHA
from v143.runtime import PackedFusedHead
from v147.codec import unpack,measured_rate
from v147.runtime import fused_half_head

from .protocol import DOMAINS,ARMS,COUNT,LABELS,FROZEN_CB147_SHA,verify_protocol
from .sources import load_fresh_validation
from .scoring import make_score,audit,label_tokens,encode_prompt


@torch.inference_mode()
def one_teacher(model,ids):
    logits=model(input_ids=ids,use_cache=False).logits[0,-1].detach()
    if logits.shape!=(49152,):raise ValueError("unexpected pretrained vocab")
    return logits


@torch.inference_mode()
def one_native_hidden(model,ids):
    h=model.model(input_ids=ids,use_cache=False).last_hidden_state
    if h.shape[:2]!=ids.shape or h.shape[-1]!=576:
        raise ValueError("packed hidden shape differs from original teacher")
    return h[:,-1:,:].contiguous()


@torch.inference_mode()
def score_head(head,hidden):
    logits=head(hidden)[0,-1].detach()
    if logits.shape!=(49152,) or not torch.isfinite(logits).all():
        raise ValueError("failed native packed output vocabulary")
    return logits


def run(packed_file,out):
    torch.set_num_threads(2)
    verify_protocol()
    out.mkdir(parents=True,exist_ok=True)
    blob=packed_file.read_bytes()
    physical=unpack(blob)
    if physical.sha256!=FROZEN_CB147_SHA:
        raise ValueError("CB147 SHA changed; not the pinned v147 artifact")
    if physical.paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not the frozen CB140 source")
    if hashlib.sha256(physical.paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original trained base source unexpectedly changed")
    tasks,sources=load_fresh_validation()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    labels={domain:label_tokens(tok,LABELS[domain]) for domain in DOMAINS}

    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("pretrained FP32 architecture changed")
    rows={a:{d:[] for d in DOMAINS} for a in ARMS}
    durations={a:0. for a in ARMS}
    with TemporaryDirectory(dir=out) as temporary:
        root=Path(temporary)
        source=root/"native_CB59.cb59"
        source.write_bytes(physical.paid.original)
        model,metadata,_=make_model(
            "baseline",SimpleNamespace(original=source,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("dense pretrained source materialized")
        original_embedding=model.model.embed_tokens
        cb59_head=model.lm_head
        half_head=fused_half_head(model,physical.paid,root)
        cb147_head=PackedFusedHead(half_head.handle,half_head.kernel,physical.bias)
        # Tied source codewords unchanged across all compressed conditions.
        if not (original_embedding.handle is cb59_head.handle
                is half_head.handle is cb147_head.handle):
            raise AssertionError("not genuinely tied original packed weights")

        for domain in DOMAINS:
            for task in tasks[domain]:
                ids=encode_prompt(tok,task.prompt)
                if ids.shape[1]>1023:
                    raise ValueError("new QA passage too long for declared limit")
                clock=perf_counter()
                with torch.inference_mode():
                    teacher_logits=one_teacher(teacher,ids)
                durations["FP32"]+=perf_counter()-clock
                clock=perf_counter()
                with torch.inference_mode():
                    hidden=one_native_hidden(model,ids)
                shared_prefill=perf_counter()-clock
                for arm,head in (
                        ("CB59",cb59_head),
                        ("CB140_head_half",half_head),
                        ("CB147_int8_paid",cb147_head)):
                    clock=perf_counter()
                    with torch.inference_mode():
                        logits=score_head(head,hidden)
                    durations[arm]+=perf_counter()-clock+shared_prefill/3
                    result=make_score(logits,labels[domain],task.correct_index)
                    result.update({"row":task.index,"prompt_sha":task.prompt_sha,
                                   "model_received_correct_answer_index":False,
                                   "prompt_length_tokens":int(ids.shape[1])})
                    rows[arm][domain].append(result)

                fp32=make_score(teacher_logits,labels[domain],task.correct_index)
                fp32.update({"row":task.index,"prompt_sha":task.prompt_sha,
                             "model_received_correct_answer_index":False,
                             "prompt_length_tokens":int(ids.shape[1])})
                rows["FP32"][domain].append(fp32)

        model.lm_head=cb59_head
        del model

    del teacher
    final={
        "stage":"v148_FRESH_VALID_PHYSICALLY_PAID_VOCAB_PRIOR_QA",
        "pinned_complete_CB147_SHA256":physical.sha256,
        "original_CB59_sha256":ORIGINAL_CB59_SHA,
        "whole_source_bytes_and_BPP":measured_rate(blob),
        "frozen_sciq_boolq_VALIDATION_slices_only":sources,
        "all_model_arms_frozen_before_validation":list(ARMS),
        "no_v89_official_independent_TEST_opened":True,
        "no_teacher_logits_fed_into_model_weights":True,
        "no_training_or_new_codebook":True,
        "no_Jessica_LIVE_Render_Supabase_APK_change":True,
        "per_arm_individual_native_timing_NOT_end_to_end_speed":durations,
        "responses_by_arm":rows,
        "quality":audit(rows),
        "no_semantic_or_release_success_claim_from_small_validation":True,
    }
    target=out/"v148_fresh_SciQ_BoolQ_paid_CB147_choice.json"
    target.write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":final["stage"],
                      "source":final["whole_source_bytes_and_BPP"],
                      "quality":final["quality"],"report":str(target)},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("pinned_CB147",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.pinned_CB147,args.out)
