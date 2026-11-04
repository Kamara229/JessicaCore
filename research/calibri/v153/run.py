"""First real original-operator remediation: 1 paid bit/group in one block.

v152's single-FP32-block oracle sensitivity ranks 30 original blocks on
*existing* diagnostic questions. Choose exactly its best KL-ranked block
BEFORE fitting source-only FP32 original weight residuals, freezing fully
paid CB153 and opening NEW wiki/news source TRAIN VALID. Source is actual
11.2MB original CB59 plus every 1bit plane and its per-matrix codebook.
NO original 2D pretrained source is expanded in runtime or reused as oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from quality import load_token_ids
from v142.scoring import compare_logits,aggregate
from v147.codec import unpack as unpack_cb147
from .protocol import (PARENT_CB147_SHA,PARENT_CB59_SHA,check,select_layer,
    NEXT_VALID,NEXT_TOKENS,NEXT_WINDOWS,WINDOW,WIKI_REVISION,
    NEWS_REVISION,WIKI_NEXT_START,NEWS_NEXT_START,NEWS_NEXT_END)
from .codec import pack,unpack,report
from .fit import train_block
from .runtime import install_paid_block


def new_source_valid(tok):
    """Access ONLY after the physical CB153 has been saved and verified."""
    raw_wiki=load_token_ids(tok,"train",WIKI_REVISION,
                            WIKI_NEXT_START+NEXT_TOKENS)
    wiki=raw_wiki[WIKI_NEXT_START:WIKI_NEXT_START+NEXT_TOKENS].clone()
    news_rows=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
        split=f"train[{NEWS_NEXT_START}:{NEWS_NEXT_END}]")
    text="\n\n".join(str(r["text"]) for r in news_rows)
    news=tok(text,add_special_tokens=False,return_tensors="pt").input_ids[0,:NEXT_TOKENS].clone()
    if len(wiki)!=NEXT_TOKENS or len(news)!=NEXT_TOKENS:
        raise ValueError("new heldout source data too small")
    return dict(zip(NEXT_VALID,(wiki,news))),{
      NEXT_VALID[0]:{"source":"Salesforce/wikitext","split":"train",
        "revision":WIKI_REVISION,"tokens":[WIKI_NEXT_START,WIKI_NEXT_START+NEXT_TOKENS],
        "sha256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
      NEXT_VALID[1]:{"source":"fancyzhx/ag_news","split":"train",
        "revision":NEWS_REVISION,"records":[NEWS_NEXT_START,NEWS_NEXT_END],
        "sha256":hashlib.sha256(news.numpy().tobytes()).hexdigest()}
    }


@torch.inference_mode()
def all_logits(model,tokens,domain):
    values=[]
    elapsed=0.0
    for i in range(NEXT_WINDOWS):
        ids=tokens[i*WINDOW:(i+1)*WINDOW][None,:]
        t0=perf_counter()
        x=model(input_ids=ids,use_cache=False).logits.detach().clone()
        elapsed+=perf_counter()-t0
        if x.shape!=(1,WINDOW,49152):
            raise ValueError("wrong original source output vocabulary")
        values.append(x)
    return values,elapsed


def run(cb147_path,v152_diagnostic_json,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    source=unpack_cb147(cb147_path.read_bytes())
    if source.sha256!=PARENT_CB147_SHA:
        raise ValueError("not original physical CB147 wrapper")
    original=source.paid.original
    if hashlib.sha256(original).hexdigest()!=PARENT_CB59_SHA:
        raise ValueError("original CB59 source bytes tampered")
    target,ranking=select_layer(v152_diagnostic_json)
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("unexpected pretrained FP32 teacher geometry")

    # Only ORIGINAL FP32 trained source 2D weights; NO TEST, QA, teacher logits
    # or held-out v153 data guide residual dictionary or bits.
    patches,training=train_block(teacher,original,target)
    paid=pack(original,target,patches)
    file=out/"CALIBRI_v153_original_layer_paid_onebit.cb153"
    file.write_bytes(paid)
    physically_verified=unpack(file.read_bytes())
    if physically_verified.layer!=target or physically_verified.original!=original:
        raise ValueError("original source changed in whole 0.7-bpp physical patch")

    # Only NOW open untouched source TRAIN VALID.
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    val,origins=new_source_valid(tok)
    truth={domain:all_logits(teacher,tokens,domain)[0]
           for domain,tokens in val.items()}
    results={arm:{domain:[] for domain in NEXT_VALID}
             for arm in ("original_CB59","physically_paid_CB153")}
    timings={arm:{} for arm in results}
    with TemporaryDirectory(dir=out) as t:
        root=Path(t)
        path=root/"locked_original.cb59"
        path.write_bytes(original)
        model,meta,_=make_model(
            "baseline",SimpleNamespace(original=path,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("native original CB59 source unexpectedly expanded")
        for domain in NEXT_VALID:
            scores,elapsed=all_logits(model,val[domain],domain)
            timings["original_CB59"][domain]=elapsed
            for n,(actual,fp32) in enumerate(zip(scores,truth[domain])):
                ids=val[domain][n*WINDOW:(n+1)*WINDOW][None,:]
                record=compare_logits(fp32,actual,ids)
                record["window_index"]=n
                results["original_CB59"][domain].append(record)
        with install_paid_block(model,physically_verified,root) as meta_paid:
            for domain in NEXT_VALID:
                scores,elapsed=all_logits(model,val[domain],domain)
                timings["physically_paid_CB153"][domain]=elapsed
                for n,(actual,fp32) in enumerate(zip(scores,truth[domain])):
                    ids=val[domain][n*WINDOW:(n+1)*WINDOW][None,:]
                    record=compare_logits(fp32,actual,ids)
                    record["window_index"]=n
                    results["physically_paid_CB153"][domain].append(record)
        if any(p.ndim>=2 for p in model.parameters()):
            raise ValueError("FP32 oracle leaked into paid runtime")

    metrics={arm:{d:aggregate(records[d]) for d in NEXT_VALID}
             for arm,records in results.items()}
    changes={}
    for domain in NEXT_VALID:
        base=metrics["original_CB59"][domain]
        refined=metrics["physically_paid_CB153"][domain]
        changes[domain]={
          "new_minus_original_CB59_gold_NLL":
            refined["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
          "new_minus_original_CB59_teacher_KL":
            refined["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
          "new_minus_original_CB59_teacher_top1":
            refined["teacher_next_token_argmax_agreement_fraction"]-
            base["teacher_next_token_argmax_agreement_fraction"],
        }
    pass_both=all(changes[d]["new_minus_original_CB59_gold_NLL"]<0 and
                  changes[d]["new_minus_original_CB59_teacher_KL"]<=0
                  for d in NEXT_VALID)
    summary={
      "stage":"v153_ONE_ORIGINAL_FP32_SENSITIVE_BLOCK_PAID_1BIT",
      "chosen_layer_FROM_preexisting_v152_diagnostics":ranking,
      "original_source_CB59_sha256":PARENT_CB59_SHA,
      "full_0p7bpp_model":report(paid),
      "original_model_2D_weights_frozen":True,
      "7_actual_original_2D_ops_patched_not_output_logit_bias":True,
      "only_source_FP32_weights_used_to_train_residual_tables":True,
      "original_source_FIT_matrix_weight_RMSE":training,
      "new_unseen_source_TRAIN_VALID_provenance":origins,
      "raw_matching_window_results":results,
      "true_next_token_NLL_PPL_and_teacher_KL":metrics,
      "per_domain_differences_vs_CB59":changes,
      "improves_both_NLL_and_KL_on_both_unseen_domains":pass_both,
      "same_runner_timing_full_64token_original_vs_paid":timings,
      "new_physical_archive_file":file.name,
      "no_teacher_FP32_matrix_in_paid_archive_or_runtime":True,
      "NO_official_v89_independent_TEST":True,
      "NO_SciQ_BoolQ_QA_training_or_achievement_claim":True,
      "NO_live_Jessica_Render_Supabase_APK_changes":True,
    }
    report_file=out/"v153_paid_onebit_original_layer_NEW_valid.json"
    report_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
      "stage":summary["stage"],"chosen_layer":target,
      "physical":summary["full_0p7bpp_model"],
      "PASS_both_new_domains":pass_both,
      "contrasts":changes,
      "metrics":metrics,
      "report":str(report_file),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_CB147",type=Path)
    p.add_argument("v152_oracle_report_JSON",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.original_CB147,args.v152_oracle_report_JSON,args.out)
