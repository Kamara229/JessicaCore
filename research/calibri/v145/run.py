"""No-tuning validation of v144 selected paid residual gains on NEW train text.

v144 source selection/confirm: WikiText token [14336,14592) and AG News
records [8500,8600). v145 uses entirely different, preregistered WikiText
[19456,19968) and news records [9100,9200). No v89 independent TEST used.
Nothing is optimized, repacked, trained or altered after reading new data.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v140.codec import unpack
from v141.run import ORIGINAL_V140_SHA
from v142.scoring import compare_logits
from v144.operators import Paired
from v144.run import get_native_logits
from .protocol import (SOURCE_DOMAINS,ARMS,WINDOW,TOKEN_COUNT,SELECTED,
                       BASELINE,PAID,windows,verify_predeclared)
from .sources import load_new_train_holdout
from .audit import assemble


@torch.inference_mode()
def make_observation(teacher_logits, baseline, residual, head_gain, token_ids):
    return compare_logits(teacher_logits,baseline+head_gain*residual,token_ids)


def run(archive,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    verify_predeclared()
    physical=unpack(archive.read_bytes())
    if physical.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not the physically pinned existing CB140 archive")
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("locked original packed CB59 lost")
    if physical.size>MAX_SOURCE_BYTES:
        raise ValueError("paid source already above 0.7 bpp")
    # Immutable choices and source revision MUST be established before loading.
    source_data,provenance=load_new_train_holdout()
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong teacher architecture")
    results={key:{domain:[] for domain in SOURCE_DOMAINS} for key in ARMS}
    with TemporaryDirectory(dir=out) as temporary:
        scratch=Path(temporary)
        base=scratch/"locked_CB59.cb59"
        base.write_bytes(physical.original)
        model,meta,_=make_model("baseline",SimpleNamespace(original=base,folder=scratch))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("native source accidentally expanded to trained FP32 matrix")
        bridge=Paired(model,physical,scratch)
        try:
            for domain in SOURCE_DOMAINS:
                for number,ids in enumerate(windows(source_data[domain])):
                    with torch.inference_mode():
                        teacher_logits=teacher(input_ids=ids,use_cache=False).logits.detach()
                        # The selected gain is frozen: input=0, head=0.5
                        cb59_logits,head_delta=get_native_logits(model,bridge,ids,0.0)
                        cb140_logits,full_delta=get_native_logits(model,bridge,ids,1.0)
                        controls=(
                            (ARMS[0],cb59_logits,head_delta,0.0),
                            (ARMS[1],cb59_logits,head_delta,0.5),
                            (ARMS[2],cb140_logits,full_delta,1.0),
                        )
                        for name,baseline,extra,scale in controls:
                            record=make_observation(teacher_logits,baseline,extra,
                                                    scale,ids)
                            record["window_number"]=number
                            results[name][domain].append(record)
        finally:
            bridge.restore()
    del teacher
    report={
        "experiment":"v145 no-tuning new source-domain-shift confirm",
        "frozen_v144_selected_input_gain":SELECTED[0],
        "frozen_v144_selected_output_gain":SELECTED[1],
        "original_CB59_physically_sha256":ORIGINAL_CB59_SHA,
        "unchanged_paid_CB140_physically_sha256":physical.sha256,
        "paid_CB140_complete_bytes":physical.size,
        "paid_CB140_physical_bpp":8*physical.size/TOTAL_UNIQUE,
        "calibration_gains_NOT_in_physical_source_yet":True,
        "hypothetical_4_bytes_packed_gain_metadata_bpp":8*(physical.size+4)/TOTAL_UNIQUE,
        "no_weight_training_or_modify_archive":True,
        "no_teacher_logits_entered_training":True,
        "v144_prior_selection_and_confirm_not_reopened":True,
        "no_independent_official_v89_TEST":True,
        "new_text_only_train_source_splits":provenance,
        "full_next_token_predictions_each_domain_each_arm":len(windows(source_data[SOURCE_DOMAINS[0]]))*(WINDOW-1),
        "arms":results,
        "audit":assemble(results),
        "do_not_claim_semantic_QA_recovery":True,
        "no_live_Jessica_Render_Supabase_APK_modification":True,
    }
    path=out/"v145_unseen_domain_shift_NLL_KL.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":"v145_NEW_DOMAIN_SHIFT_CONFIRM_WITHOUT_RETUNING",
        "frozen_v144_selected_gains":SELECTED,
        "physical_bpp":report["paid_CB140_physical_bpp"],
        "audit":report["audit"],
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("frozen_paid_CB140",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.frozen_paid_CB140,a.out)
