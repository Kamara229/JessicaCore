"""Fixed 25-arm tied one-bit gain diagnostic on VALID_SELECT then VALID_CONFIRM.

Same frozen physically paid original CB140 and original CB59 packed source.
No new training, no independent official v89 TEST, no model serialization.
Original train split provides four 64-token windows each for WikiText/AG
News; 0,1 may choose gain, 2,3 are only opened AFTER selecting gain.
This does not imply same target works for any independent corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import get_tokens,make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v140.codec import unpack,report
from v141.run import ORIGINAL_V140_SHA
from v142.scoring import compare_logits,aggregate
from .operators import Paired
from .protocol import (DOMAINS,WINDOW,GAIN_VALUES,GAIN_GRID,
                       SELECT_WINDOW_IDS,CONFIRM_WINDOW_IDS,
                       label,BASELINE,ORIGINAL_PAID)
from .selection import choose


@torch.inference_mode()
def get_native_logits(model,paired,token_ids,input_gain):
    """Cache the SAME decoder-hidden state across all 5 head gain choices."""
    paired.configure(input_gain,0.0)
    hidden=model.model(input_ids=token_ids,use_cache=False).last_hidden_state
    if hidden.shape[:2]!=token_ids.shape or hidden.shape[-1]!=576:
        raise ValueError("unexpected native decoder hidden shape")
    head_baseline=paired.head.original(hidden)
    paid_delta=torch.from_numpy(paired.head.kernel.linear(
        hidden.detach().cpu().numpy())).reshape(head_baseline.shape)
    if not torch.isfinite(head_baseline).all() or not torch.isfinite(paid_delta).all():
        raise ValueError("invalid packed logits or 1-bit residual")
    return head_baseline,paid_delta


@torch.inference_mode()
def evaluate_window(model,paired,token_ids,teacher_logits,input_gains,head_gains):
    records={}
    for i in input_gains:
        baseline,delta=get_native_logits(model,paired,token_ids,i)
        for h in head_gains:
            key=label(i,h)
            logit=baseline+h*delta
            records[key]=compare_logits(teacher_logits,logit,token_ids)
    return records


@torch.inference_mode()
def teacher_window(model,ids):
    return model(input_ids=ids,use_cache=False).logits.detach().clone()


def run(artifact,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    physical_blob=artifact.read_bytes()
    physical=unpack(physical_blob)
    if physical.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("frozen CB140 physical SHA mismatch")
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("frozen original CB59 source mismatch")
    raw,origin={},{}
    for domain in DOMAINS:
        ids,source=get_tokens(domain)
        if source.get("split")!="train" or len(ids)<4*WINDOW:
            raise ValueError("expected disjoint original TRAIN-split VALID windows")
        raw[domain]=ids
        origin[domain]=source

    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong fixed original FP32 model")
    data={
        "stage":"v144_PREDECLARED_GAIN_SELECT_THEN_UNSEEN_CONFIRM",
        "archived_CB140_SHA256":physical.sha256,
        "original_CB59_SHA256":ORIGINAL_CB59_SHA,
        "physical_bytes":physical.size,
        "physical_bpp_without_new_gain_metadata":8*physical.size/TOTAL_UNIQUE,
        "gain_metadata_NOT_physically_saved_in_CB140":True,
        "hypothetical_plus_4bytes_gain_storage_bpp":
            8*(physical.size+4)/TOTAL_UNIQUE,
        "still_under_theoretical_0_7bit_cap_if_4bytes_added":physical.size+4<=MAX_SOURCE_BYTES,
        "separate_input_and_head_gains_NOT_trained_weights":True,
        "candidate_selection_only_from_WINDOWS_0_and_1":True,
        "confirm_windows_2_3_untouched_until_gain_selected":True,
        "locked_independent_v89_TEST_unopened":True,
        "original_model_weights_unmodified":True,
        "no_live_Jessica_Render_Supabase_APK_changes":True,
        "sources":origin,
        "grid":[{"input_gain":i,"head_gain":h,"arm":label(i,h)}
                for i,h in GAIN_GRID],
        "selection":{},"confirm":{},
    }
    with TemporaryDirectory(dir=out) as directory:
        folder=Path(directory)
        original=folder/"real_CB59.cb59"
        original.write_bytes(physical.original)
        model,meta,_=make_model("baseline",
            SimpleNamespace(original=original,folder=folder))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("not native all-packed model")
        paired=Paired(model,physical,folder)
        try:
            fitted={label(i,h):{d:[] for d in DOMAINS} for i,h in GAIN_GRID}
            # FIRST phase: ONLY source VALID selection windows 0,1.
            for domain in DOMAINS:
                ids=raw[domain]
                for n in SELECT_WINDOW_IDS:
                    x=ids[n*WINDOW:(n+1)*WINDOW][None,:]
                    teacher_ref=teacher_window(teacher,x)
                    details=evaluate_window(model,paired,x,teacher_ref,
                                            GAIN_VALUES,GAIN_VALUES)
                    for arm,row in details.items():
                        fitted[arm][domain].append(row)
            selected,decision=choose(fitted)
            data["selection"]={"chosen_arm":selected,**decision}

            # SECOND phase: only after choosing fixed gains, read windows 2,3.
            selected_pair=next((i,h) for i,h in GAIN_GRID if label(i,h)==selected)
            confirm_arms=tuple(dict.fromkeys((BASELINE,ORIGINAL_PAID,selected)))
            confirmed={arm:{d:[] for d in DOMAINS} for arm in confirm_arms}
            for domain in DOMAINS:
                ids=raw[domain]
                for n in CONFIRM_WINDOW_IDS:
                    x=ids[n*WINDOW:(n+1)*WINDOW][None,:]
                    teacher_ref=teacher_window(teacher,x)
                    inputs=tuple(sorted({i for i,h in [
                        (0.0,0.0),(1.0,1.0),selected_pair]}))
                    for i in inputs:
                        baseline,delta=get_native_logits(model,paired,x,i)
                        for arm in confirm_arms:
                            pair=next((a,b) for a,b in GAIN_GRID
                                      if label(a,b)==arm)
                            if pair[0]!=i:continue
                            confirmed[arm][domain].append(
                                compare_logits(teacher_ref,baseline+pair[1]*delta,x))
            data["confirm"]={
                "never_tuned_against_confirm_windows":True,
                "per_arm_per_domain":{arm:{domain:aggregate(records[domain])
                                          for domain in DOMAINS}
                                      for arm,records in confirmed.items()},
            }
            selected_s=data["confirm"]["per_arm_per_domain"][selected]
            old_s=data["confirm"]["per_arm_per_domain"][BASELINE]
            data["confirm"]["NLL_improved_both_domains_vs_CB59"] = all(
                selected_s[d]["mean_TRUE_next_token_NLL"] <
                old_s[d]["mean_TRUE_next_token_NLL"] for d in DOMAINS)
            data["confirm"]["teacher_KL_no_regression_both_domains_vs_CB59"]=all(
                selected_s[d]["teacher_to_candidate_KL"] <=
                old_s[d]["teacher_to_candidate_KL"] for d in DOMAINS)
            data["confirm"]["meets_untouched_confirm_quality_gate"]=bool(
                selected!=BASELINE and
                data["confirm"]["NLL_improved_both_domains_vs_CB59"] and
                data["confirm"]["teacher_KL_no_regression_both_domains_vs_CB59"])
            data["semantic_QUALITY_still_unverified"]=True
            data["CPU_runtime_speed_still_requires_verification"]=True
        finally:
            paired.restore()
    del teacher
    target=out/"v144_tied_gain_two_domain_confirm.json"
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    compact={"stage":data["stage"],"selected":data["selection"]["chosen_arm"],
             "no_robust_selection":data["selection"]["no_robust_both_domain_candidate"],
             "confirm":data["confirm"],"source_bpp":data["physical_bpp_without_new_gain_metadata"],
             "report":str(target)}
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("paid_cb140",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.paid_cb140,a.out)
