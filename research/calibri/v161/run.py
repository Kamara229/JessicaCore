"""Physically pay one more bit for seven ORIGINAL source block-0 operators.

Previous measured evidence: v159 original CB59 input embeddings RMSE ~0.72;
block 0 adds a teacher-relative RMSE ~+0.23 in both original source FIT
domains, most damaging early increment. v160 (completed CI #41919065068)
paid original tied token embedding extra 1bit reduces input RMSE
0.724->0.687 Wiki, 0.720->0.683 news, but input-only on new text badly
WORSENS true NLL by +1.458 Wiki, +0.089 news and KL by +1.423/+0.214.
More accurate source embedding is not sufficient.

v161 changes the 7 real first transformer layer *weight operators*, not
logits or model priors: source FP32 weight-to-residual 2-vector dictionary,
old 5bit codes frozen, new single bit per group physically serialized
under strict <=0.7bpp. Compare against native CB59. A genuine FP32-only
first-block ORACLE is separately measured to estimate upper potential
but explicitly violates physical source budget and cannot be a candidate.

No test leakage: old v159/v160 evidence verified; after source-only
weight residual FIT, save complete archive with SHA/CRC; ONLY THEN open
completely new original TRAIN WikiText and AG News source VALID.
"""
import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from v140.codec import unpack as unpack_original_paid_tied
from v153.codec import pack,unpack,report
from v153.fit import train_block
from v153.runtime import install_paid_block
from v152.oracles import replace_one_layer
from v142.scoring import compare_logits

from .protocol import (
    check,LAYER,ARMS,DOMAINS,ORACLE_NAME,WINDOW,WINDOWS,TOKENS,
    ORIGINAL_V140_SHA,MATRIX_KEYS)
from .evidence import verify_old
from .sources import unopened_until_paid_archive_frozen
from .audit import summarize


@torch.inference_mode()
def full_source_logits(model,source):
    if len(source)!=TOKENS:
        raise ValueError("model source must use original 8x64 TRAIN-only VALID")
    out=[]
    seconds=0.
    for index in range(WINDOWS):
        ids=source[index*WINDOW:(index+1)*WINDOW][None,:]
        t0=perf_counter()
        values=model(input_ids=ids,use_cache=False).logits.detach().clone()
        seconds+=perf_counter()-t0
        if values.shape!=(1,WINDOW,49152) or not torch.isfinite(values).all():
            raise ValueError("original pretrained teacher output vocabulary changed")
        out.append(values)
    return out,seconds


def run(original_paid_CB140,v159_prior,v160_prior,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    prior=verify_old(v159_prior,v160_prior)
    old=unpack_original_paid_tied(original_paid_CB140.read_bytes())
    if old.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not exact original full physically paid CB140 source")
    original=old.original
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original 5bit pretrained source differs")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("unmatched pretrained 135M teacher architecture")

    # Only ORIGINAL FP32 pretrained SOURCE WEIGHTS are used here, no
    # source TRAIN tokens, no QA or validation text enters this fit.
    patches,fit_report=train_block(teacher,original,LAYER)
    if set(patches)!=set(MATRIX_KEYS):
        raise ValueError("original pretrained block0 7 linear operators missing")
    physical_bytes=pack(original,LAYER,patches)
    target=out/"CALIBRI_v161_original_block0_paid_onebit.cb153"
    target.write_bytes(physical_bytes)
    physical=unpack(target.read_bytes())
    if (physical.original!=original or physical.layer!=LAYER or
            physical.sha256!=hashlib.sha256(physical_bytes).hexdigest() or
            physical.size>MAX_SOURCE_BYTES):
        raise ValueError("under-budget source-only layer0 bitstream did not round-trip")

    frozen_before_validation=out/"v161_source_only_layer0_frozen_before_VALID.json"
    frozen_before_validation.write_text(json.dumps({
        "stage":"v161_PHYSICAL_SOURCE_LOCK",
        "pinned_original_CB59_SHA256":ORIGINAL_CB59_SHA,
        "physically_serialized_complete_CB161_SHA256":physical.sha256,
        "complete_paid_archive":report(physical_bytes),
        "parent_old_v159_and_v160_negative_evidence":prior,
        "original_source_only_FP32_weight_residual_training":fit_report,
        "NO_training_corpus_VALID_or_QA_access_before_file_written":True,
        "source_VALID_text_not_opened_yet":True,
        "FP32_teacher_oracle_not_a_compact_candidate":True,
    },indent=2,ensure_ascii=False)+"\n")

    # ONLY NOW read previously unobserved source TRAIN-only validation.
    new_source,provenance=unopened_until_paid_archive_frozen()
    reference={domain:full_source_logits(teacher,tokens)[0]
               for domain,tokens in new_source.items()}
    per_window={arm:{domain:[] for domain in DOMAINS} for arm in ARMS}
    elapsed={arm:{} for arm in ARMS}
    oracle_bytes=sum(p.numel()*p.element_size()
                     for p in teacher.model.layers[LAYER].parameters())
    if oracle_bytes<=100000:
        raise ValueError("FP32 oracle must be an explicitly large unpaid intervention")

    with TemporaryDirectory(dir=out) as temp:
        scratch=Path(temp)
        root=scratch/"original_CB59.cb59"
        root.write_bytes(original)
        student,metadata,_=make_model("baseline",SimpleNamespace(
            original=root,folder=scratch))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("CB59 original pretrained model loaded dense 2D tensors")
        embedding=student.model.embed_tokens
        head=student.lm_head
        original_layer=student.model.layers[LAYER]
        if embedding.handle is not head.handle:
            raise ValueError("source original input and output no longer tied")

        for arm in ARMS:
            if arm==ARMS[0]:
                intervention=nullcontext({"original_CB59":True})
            elif arm==ARMS[1]:
                intervention=install_paid_block(student,physical,scratch)
            else:
                intervention=replace_one_layer(student,teacher,LAYER)
            with intervention:
                for domain in DOMAINS:
                    actual,sec=full_source_logits(student,new_source[domain])
                    elapsed[arm][domain]=sec
                    for index,(ref,observed) in enumerate(zip(reference[domain],actual)):
                        ids=new_source[domain][index*WINDOW:(index+1)*WINDOW][None,:]
                        row=compare_logits(ref,observed,ids)
                        row["source_window"]=index
                        row["original_FP32_oracle_UNPAID"]=arm==ORACLE_NAME
                        per_window[arm][domain].append(row)
            if (student.model.layers[LAYER] is not original_layer
                    or student.model.embed_tokens is not embedding
                    or student.lm_head is not head):
                raise AssertionError("one original FP32/paid module intervention not restored")
            if any(p.ndim>=2 for p in student.parameters()):
                raise AssertionError("original dense matrix survives after source-only intervention")

    del teacher
    quality=summarize(per_window)
    result={
        "stage":"v161_GENUINE_PAID_ORIGINAL_BLOCK0_REPAIR_VS_UNPAID_FP32_ORACLE",
        "unchanged_original_CB59_SHA256":ORIGINAL_CB59_SHA,
        "physically_paid_CB161_0p7bpp_source":report(physical_bytes),
        "precommitted_layer0_because_v159_large_first_block_drift":True,
        "v160_paid_input_embedding_worsened_real_source_NLL_and_KL":prior,
        "original_trainable_pretrained_2D_residual_fit_only":fit_report,
        "real_new_untouched_TRAIN_split_validation_sources":provenance,
        "physical_source_7_original_block0_linear_ops_patched":True,
        "original_tied_input_and_output_head_UNCHANGED":True,
        "all_other_29_original_transformer_layers_UNCHANGED":True,
        "full_source_frozen_before_first_new_validation_token":frozen_before_validation.name,
        "complete_source_SHA256":physical.sha256,
        "unpaid_large_teacher_FP32_block0_oracle_bytes":oracle_bytes,
        "unpaid_oracle_must_NOT_be_reported_as_within_0p7_bpp":True,
        "per_arm_full_original_model_true_NLL_PPL_teacher_KL_top1":quality,
        "raw_matched_source_next_token_windows":per_window,
        "full_model_CPU_prefill_seconds_per_arm_domain":elapsed,
        "NO_frozen_v89_original_independent_TEST":True,
        "NO_untouched_SciQ_BoolQ_QA_or_greedy_generation_result":True,
        "NO_incorrect_semantic_or_performance_recovery_claim":True,
        "NO_LIVE_Jessica_Render_Supabase_APK_change":True,
    }
    path=out/"v161_original_block0_source_operator_valid.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":result["stage"],
        "real_all_in_bpp":result["physically_paid_CB161_0p7bpp_source"]["all_in_bpp_unique_original"],
        "paid_block0_GATE_PASS":quality[
            "physically_paid_block0_improves_true_NLL_and_teacher_KL_in_both_domains"],
        "new_source_real_gain_by_domain":quality["new_VALID_frozen_effects"],
        "oracle_BYTES_NOT_paid":oracle_bytes,
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_paid_CB140",type=Path)
    p.add_argument("real_v159_JSON",type=Path)
    p.add_argument("real_v160_JSON",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.original_paid_CB140,a.real_v159_JSON,a.real_v160_JSON,a.out)
