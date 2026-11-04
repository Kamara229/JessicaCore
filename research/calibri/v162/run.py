"""Real 2016 gold next tokens x two domains x two packed original models.

No new neural weights, operator tuning, validation-selected thresholds,
auxiliary 'green tunnel' modules, or extra bitstream. Compare frozen
fully paid original 0.67158-bpp CB161 (source-trained layer0 5+1bit)
against original direct-native CB59 across a fifth *completely disjoint*
original TRAIN-split text region. v161's tiny eight-window gains were
preliminary and teacher top1 agreement was nearly zero. Robustness must
withstand >=20/32 window improvements and leave-best-out tests on both
NLL and KL and both domains.

Opening source data must happen STRICTLY after physical SHA evidence
is verified and an immutable selection/threshold seal is saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v142.scoring import compare_logits

from .protocol import (NATIVE_ARMS,DOMAINS,WINDOW,WINDOWS,SOURCE_TOKENS,
                       check,ORIGINAL_CODEBOOK_SHA)
from .evidence import frozen_v161
from .sources import load_untouched_source
from .audit import summarize


@torch.inference_mode()
def one_domain(model,tokens):
    if len(tokens)!=SOURCE_TOKENS:raise ValueError("unexpected frozen source length")
    output=[]
    sec=0.
    for j in range(WINDOWS):
        window=tokens[j*WINDOW:(j+1)*WINDOW][None,:]
        start=perf_counter()
        logits=model(input_ids=window,use_cache=False).logits.detach().clone()
        sec+=perf_counter()-start
        if logits.shape!=(1,WINDOW,49152) or not torch.isfinite(logits).all():
            raise ValueError("wrong original source full vocabulary or NaN")
        output.append(logits)
    return output,sec


def run(archive,v161_full_report,v161_pre_valid_seal,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    blob=archive.read_bytes()
    physical=unpack(blob)
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CODEBOOK_SHA:
        raise ValueError("pretrained original CB59 source altered")
    prior=frozen_v161(physical,v161_full_report,v161_pre_valid_seal)
    if prior["complete_v161_model_SHA256"]!=physical.sha256:
        raise ValueError("archive changed during source selection")

    # Whole source was paid and persisted by v161; DO NOT repack or select
    # per-token correction, or count only active patches rather than all.
    seal=out/"v162_predeclared_gate_and_model_SHA_BEFORE_new_data.json"
    seal.write_text(json.dumps({
        "stage":"v162_PRECOMMITTED_REVALIDATION",
        "whole_parent_physical_source_SHA256":physical.sha256,
        "whole_parent_physical_source_bytes":len(blob),
        "physical_bpp_all_weights_counted":8*len(blob)/134515008,
        "prior_completed_v161_measured_evidence":prior,
        "model_arms":list(NATIVE_ARMS),
        "new_source_domains":list(DOMAINS),
        "windows_per_domain":WINDOWS,
        "tokens_per_window":WINDOW,
        "min_improved_windows_per_domain_PER_metric":20,
        "drop_single_most_favorable_window_per_metric":True,
        "new_validation_source_NOT_loaded_when_seal_written":True,
        "no_v89_TEST_ever_loaded":True,
    },ensure_ascii=False,indent=2)+"\n")

    source,provenance=load_untouched_source()
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong original pretrained model depth")
    records={arm:{domain:[] for domain in DOMAINS} for arm in NATIVE_ARMS}
    timings={arm:{} for arm in NATIVE_ARMS}

    with TemporaryDirectory(dir=out) as t:
        scratch=Path(t)
        cb59_path=scratch/"original_CB59_source.cb59"
        cb59_path.write_bytes(physical.original)
        student,meta,_=make_model(
            "baseline",SimpleNamespace(original=cb59_path,folder=scratch))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original 2D pretrained tensors leaked")
        embedding=student.model.embed_tokens
        head=student.lm_head
        if embedding.handle is not head.handle:
            raise ValueError("original tied input/head source must remain identical")
        original_0=student.model.layers[0]
        for domain in DOMAINS:
            ids=source[domain]
            truth,teacher_seconds=one_domain(teacher,ids)
            baseline,elapsed=one_domain(student,ids)
            timings[NATIVE_ARMS[0]][domain]=elapsed

            # Pay same original layer0 1bit physical codebook, once per
            # original source domain. All 7 original block0 ops patched.
            with install_paid_block(student,physical,scratch):
                candidate,elapsed=one_domain(student,ids)
            timings[NATIVE_ARMS[1]][domain]=elapsed
            if (student.model.layers[0] is not original_0 or
                    student.model.embed_tokens is not embedding or
                    student.lm_head is not head):
                raise ValueError("paid correction leaked into original control")
            if any(x.ndim>=2 for x in student.parameters()):
                raise ValueError("dense original source survived native correction")

            for j,(ref,base,paid) in enumerate(zip(truth,baseline,candidate)):
                tokens=ids[j*WINDOW:(j+1)*WINDOW][None,:]
                for arm,logits in ((NATIVE_ARMS[0],base),(NATIVE_ARMS[1],paid)):
                    comparison=compare_logits(ref,logits,tokens)
                    comparison["matched_original_source_window"]=j
                    comparison["whole_model_source_SHA256"]=physical.sha256
                    records[arm][domain].append(comparison)
            del truth,baseline,candidate

    del teacher
    audit=summarize(records,timings)
    report_data={
        "stage":"v162_REAL_FULLY_PAID_BLOCK0_32WINDOW_TWO_DOMAIN_ROBUSTNESS",
        "source_parent_frozen_v161_SHA256":physical.sha256,
        "all_original_physical_weights_paid":report(blob),
        "original_complete_CB59_source_SHA256":ORIGINAL_CB59_SHA,
        "prior_actual_v161_source_VALID_PASS_small":prior,
        "selection_and_robustness_thresholds_sealed_before_new_TRAIN_VALID":seal.name,
        "fifth_disjoint_original_source_TRAIN_slices":provenance,
        "no_train_or_weight_reencoding_or_new_tuning":True,
        "matched_true_gold_2016_next_tokens_per_domain_arm":True,
        "two_complete_native_0p7bpp_arms":list(NATIVE_ARMS),
        "original_FP32_is_reference_ONLY_not_a_compressed_candidate":True,
        "true_NLL_PPL_teacher_KL_early_teacher_argmax_results":audit,
        "all_paired_32_window_gold_records":records,
        "full_model_CPU_forward_time_NOT_autoregressive_Generation":timings,
        "frozen_independent_v89_TEST_UNTOUCHED":True,
        "no_SciQ_BoolQ_QA_or coherent_generation_result":True,
        "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    target=out/"v162_block0_full_paid_dual_corpus_32window_robustness.json"
    target.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "full_physical_model_bpp":report_data["all_original_physical_weights_paid"]["all_in_bpp_unique_original"],
        "strict_TWO_DOMAIN_gate_PASS":audit["both_domain_precommitted_STRONG_NLL_KL_robustness_pass"],
        "two_DOMAIN_quality_deltas":audit["per_domain_robustness"],
        "report":str(target),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("pinned_v161_CB161_archive",type=Path)
    parser.add_argument("actual_completed_v161_validation_JSON",type=Path)
    parser.add_argument("v161_pre_validation_model_SHA_seal",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.pinned_v161_CB161_archive,
        args.actual_completed_v161_validation_JSON,
        args.v161_pre_validation_model_SHA_seal,args.out)
