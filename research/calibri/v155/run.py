"""Unbiased new source VALID for best paid original operator from v154.

The decision comes from old eight diagnostic prompts ONLY; it is frozen
before new natural text from the original TRAIN split is loaded. No
weights are trained, no 0.67158-bpp source is reserialized, and no QA
gold/official v89 TEST are accessed. Count the entire physical CB153
archive including six unused one-bit correction matrices in every arm.
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
from v117.protocol import ORIGINAL_CB59_SHA
from v142.scoring import compare_logits,aggregate
from v153.codec import unpack,report
from v153.runtime import install_paid_block

from .protocol import ARM_NAMES,DOMAINS,WINDOW,WINDOWS,TOKENS,check,precommit_selection
from .selection import frozen_arms,holdout_gate
from .sources import new_source


@torch.inference_mode()
def logits_on_windows(model,source):
    output=[]
    elapsed=0.0
    if len(source)!=TOKENS:
        raise ValueError("wrong preregistered original TRAIN source length")
    for idx in range(WINDOWS):
        ids=source[idx*WINDOW:(idx+1)*WINDOW][None,:]
        t0=perf_counter()
        response=model(input_ids=ids,use_cache=False).logits.detach()
        elapsed+=perf_counter()-t0
        if response.shape!=(1,WINDOW,49152) or not torch.isfinite(response).all():
            raise ValueError("model source full vocabulary corrupted")
        output.append(response)
    return output,elapsed


def run(archive,diagnostic,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    complete=archive.read_bytes()
    physical=unpack(complete)
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original pretrained source changed")
    # Only old 8 historical diagnostic prompts guide the chosen operator.
    # No new source data opened until selection is finished and logged.
    decision=precommit_selection(diagnostic)
    arms=frozen_arms(physical,decision)
    registered=out/"v155_selection_LOCKED_BEFORE_NEW_DATA.json"
    registered.write_text(json.dumps({
        "selection":decision,"predeclared_arms":list(arms),
        "whole_source_sha256":physical.sha256,
        "selected_without_new_valid_or_test":True},indent=2)+"\n")
    if tuple(arms)!=ARM_NAMES:
        raise ValueError("source strategy changed after prior diagnostic")

    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("teacher arch differs from 0.671bpp source")
    # New source is opened after frozen arms were written above.
    originals,provenance=new_source()
    references={domain:logits_on_windows(teacher,originals[domain])[0]
                for domain in DOMAINS}
    del teacher

    records={name:{domain:[] for domain in DOMAINS} for name in ARM_NAMES}
    timings={name:{} for name in ARM_NAMES}
    with TemporaryDirectory(dir=out) as temporary:
        root=Path(temporary)
        base=root/"complete_original_CB59.cb59"
        base.write_bytes(physical.original)
        model,metadata,_=make_model("baseline",SimpleNamespace(original=base,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("fully packed pretrained matrix unexpectedly expanded")
        saved_layers=tuple(model.model.layers)
        original_embedding=model.model.embed_tokens
        original_head=model.lm_head
        if original_embedding.handle is not original_head.handle:
            raise ValueError("source original embeddings not tied")
        for name,partial in arms.items():
            if partial.patches:
                context=install_paid_block(model,partial,root)
            else:
                context=nullcontext({"original_CB59":True})
            with context:
                for domain in DOMAINS:
                    logit_windows,elapsed=logits_on_windows(model,originals[domain])
                    timings[name][domain]=elapsed
                    for i,actual in enumerate(logit_windows):
                        ids=originals[domain][i*WINDOW:(i+1)*WINDOW][None,:]
                        measurement=compare_logits(references[domain][i],actual,ids)
                        measurement["v155_new_VALID_window_index"]=i
                        records[name][domain].append(measurement)
            if any(a is not b for a,b in zip(saved_layers,model.model.layers)):
                raise AssertionError("paid code source intervention not restored")
            if model.model.embed_tokens is not original_embedding or model.lm_head is not original_head:
                raise AssertionError("paid source changed tied word operators")
            if any(p.ndim>=2 for p in model.parameters()):
                raise AssertionError("original FP32 matrix still loaded")

    aggregated={name:{domain:aggregate(records[name][domain])
                      for domain in DOMAINS} for name in ARM_NAMES}
    changes,passed=holdout_gate(aggregated)
    outcome={
        "stage":"v155_BEST_SINGLE_PAID_ORIGINAL_OPERATOR_UNSEEN_TWO_DOMAIN_VALID",
        "unchanged_frozen_original_complete_CB153":report(complete),
        "entire_paid_source_still_charged_for_all_seven_stored_operators":True,
        "selection_FROM_prior_v154_eight_reused_prompts_ONLY":decision,
        "immutable_pre_DATA_selection_file":registered.name,
        "new_source_TRAIN_validation_provenance":provenance,
        "different_context_windows_per_domain":WINDOWS,
        "gold_next_tokens_scored_each_domain_each_arm":WINDOWS*(WINDOW-1),
        "full_model_original_operator_arms":list(ARM_NAMES),
        "source_validation_true_NLL_KL_PPL_and_top1":aggregated,
        "paired_all_window_metrics":records,
        "one_operator_vs_original_CB59_and_all7":changes,
        "best_single_operator_improves_both_NLL_and_KL_on_both_new_domains":passed,
        "raw_full_model_CPU_seconds_for_each_arm":timings,
        "no_optimization_or_source_reencoding_after_new_source_opened":True,
        "no_SciQ_BoolQ_QA_data_or_model_answers_used":True,
        "no_official_v89_independent_TEST":True,
        "no_semantic_QA_or_coherent_generation_success_claim":True,
        "no_live_Jessica_Render_Supabase_APK_changes":True,
    }
    report_file=out/"v155_single_operator_crossdomain_validation.json"
    report_file.write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":outcome["stage"],"selection":decision,
        "complete_bpp":outcome["unchanged_frozen_original_complete_CB153"]["all_in_bpp_unique_original"],
        "best_single_operator_passed":passed,
        "new_VALID_contrasts":changes,
        "new_VALID_complete_scores":aggregated,
        "report":str(report_file),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_complete_CB153",type=Path)
    p.add_argument("v154_old_diagnostic_JSON",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.original_complete_CB153,args.v154_old_diagnostic_JSON,args.out)
