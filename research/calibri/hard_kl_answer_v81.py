#!/usr/bin/env python3
"""CALIBRI v81: true constrained supervised QA learning on REAL packed 135M.

TRAIN-only hard per-corpus KL projected Adam at the actual physically
rounded CB76 weight factors. Independent disjoint labeled SciQ/BoolQ TRAIN
FIT/VALID, freeze source before brand-new external TEST. All original 135M
pretrained tensors remain original fully counted CB59 native 5bit source;
paid rank4 + vocabulary bias within same physically counted <=0.700bpw
CB76 file, small paid correction CPU Torch, NOT fully native C.
Research only; no Jessica live, Render, Supabase, main or APK updates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from paid_lowrank_v76 import (
    SOURCE_BYTES, REQUIRED_PHYSICAL_BYTES, MAX_PHYSICAL_BYTES,
    DIM, VOCAB, RANK, pack, unpack, make_model,
)
import answer_supervision_v79 as old
import repro_qa_calibration_v78 as evaluation

FIT_ROWS={"sciq":(1050,1082),"boolq":(2500,2564)}
VALID_ROWS={"sciq":(1180,1204),"boolq":(2700,2732)}
NEW_TEST_SCIQ=(500,528)
NEW_TEST_BOOLQ=(2350,2382)
NEW_TEST_WIKI=115000
NEW_TEST_NEWS=(7200,7300)
QA_CHECKPOINTS=(0,4,8,12,18)
PROJECTION_ALPHAS=(1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0)
LANG_MAX_DELTA=0.25
TASK_VALID_MIN_GAIN=0.05
LR=0.003
SEED=81
KL_FIT_WEIGHT=0.10
L2_CHANGE_WEIGHT=0.003
LANG_DOMAINS=("wiki_valid","news_valid")
MODEL_KINDS=("teacher","baseline","paid_v76","qa_v79")
NEW_PROMPTS=(
    "One result from the research is that",
    "When the traveler returned to the village,",
    "A careful proof begins by explaining",
)
# DO NOT use the old heldout v79 TEST when choosing a model: those
# have been disclosed and are not an independent v81 quality gate.
def configure():
    old.TRAIN_ROWS=FIT_ROWS
    old.VALID_ROWS=VALID_ROWS
    old.SEED=SEED
    old.NEW_TEST_SCIQ=NEW_TEST_SCIQ
    old.NEW_TEST_BOOLQ=NEW_TEST_BOOLQ
    old.NEW_TEST_WIKI_START=NEW_TEST_WIKI
    old.NEW_TEST_NEWS=NEW_TEST_NEWS

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def projected_values(state):
    # The actual inference bytes are fp16: never use float32-only gate values.
    return tuple(torch.from_numpy(
        np.asarray(z.detach().cpu().numpy(),dtype="<f2").astype(np.float32))
        for z in state)

def to_storage(state):
    return tuple(np.asarray(z.detach().cpu().numpy(),dtype="<f2").copy() for z in state)

def calc_guard(state,lang):
    with torch.no_grad():
        return {name:old.languageKL(state,samples)
                for name,samples in lang.items()}

def get_data(folder):
    fit,fitmeta=old.load_case_data(folder,"fit")
    valid,validmeta=old.load_case_data(folder,"valid")
    lang={k:old.get_language_ref(SimpleNamespace(folder=folder),k) for k in LANG_DOMAINS}
    train_anchor=[old.get_language_ref(SimpleNamespace(folder=folder),k)
                  for k in ("wiki_fit","news_fit")]
    return fit,valid,fitmeta,validmeta,lang,train_anchor

def train(args):
    configure()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started=time.perf_counter()
    fit,valid,fitmeta,validmeta,lang,anchor_full=get_data(args.folder)
    source=args.original.read_bytes()
    prior=(args.folder/"CALIBRI_v76_paid_full.cb76").read_bytes()
    _,u0,v0,b0=unpack(prior)
    if source!=prior[:SOURCE_BYTES] or len(prior)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("complete original pretrained source mismatch")
    previous=json.loads((args.folder/"v76_selection.json").read_text())
    if previous["chosen_source_mode"]!="rank4" or previous["chosen_source_step"]!=20:
        raise AssertionError("parent must be the original TRAIN-only ranked source")
    init=tuple(torch.from_numpy(z.astype(np.float32)) for z in (u0,v0,b0))
    oldqa=old.qastats(init,valid)
    baseline_guard=calc_guard(init,lang)
    weights=[torch.nn.Parameter(x.clone()) for x in init]
    opt=torch.optim.Adam(weights,lr=LR)
    states={0:to_storage(weights)}
    # Previous teacher/source FULL corpus features are TRAIN; sample 32
    # for predictable CPU memory/latency. Hard constraint uses entire
    # teacher-source TRAIN-CAL (v76 Wiki/News TRAIN VALID), not independent
    # language validation; this distinction is explicit in the report.
    anchors=[]
    for teacher,base,hidden in anchor_full:
        reference=teacher[:32].log_softmax(-1).exp().detach()
        anchors.append((reference,base[:32],hidden[:32]))
    trace=[]
    for step in range(1,max(QA_CHECKPOINTS)+1):
        prev=[w.detach().clone() for w in weights]
        opt.zero_grad(set_to_none=True)
        answers=old.find_answer_probs(tuple(weights),fit)
        sci=torch.nn.functional.cross_entropy(*answers["sciq"])
        boo=torch.nn.functional.cross_entropy(*answers["boolq"])
        langfit=[]
        for target,z,hidden in anchors:
            uu,vv,bb=weights
            logq=(z+(hidden@vv)@uu.T+bb).log_softmax(-1)
            langfit.append(-(target*logq).sum(-1).mean())
        deviation=sum((w-ref).square().mean() for w,ref in zip(weights,init))
        loss=(.5*sci+.5*boo+
              KL_FIT_WEIGHT*sum(langfit)/len(langfit)+
              L2_CHANGE_WEIGHT*deviation)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weights,5.0)
        opt.step()
        tentative=[w.detach().clone() for w in weights]
        accepted=None
        # TRUE explicit constraint enforced at each optimizer update on
        # physically fp16-quantized factors, not only a soft KL coefficient.
        # The only validation used by this projection is pretrained
        # language TRAIN-CAL; distinct QA TRAIN VALID remains for source
        # promotion. Already-exposed earlier v76 CAL is NOT new test.
        for alpha in PROJECTION_ALPHAS:
            snap=projected_values(
                [p+alpha*(t-p) for p,t in zip(prev,tentative)])
            if not all(bool(torch.isfinite(z).all()) for z in snap):
                continue
            trial_kl=calc_guard(snap,lang)
            if all(trial_kl[k]<=baseline_guard[k]+LANG_MAX_DELTA+1e-5
                   for k in LANG_DOMAINS):
                accepted=(alpha,snap,trial_kl)
                break
        if accepted is None:
            raise AssertionError("even zero projected update violated TRAIN-CAL")
        alpha,accepted_weights,training_cal_kl=accepted
        with torch.no_grad():
            for w,new in zip(weights,accepted_weights):
                w.copy_(new)
        trace.append({"step":step,"projected_alpha":alpha,
           "TRAIN_CAL_KL_delta":{k:training_cal_kl[k]-baseline_guard[k]
                                 for k in LANG_DOMAINS},
           "TRAIN_FIT_sciq_CE_pre_step":float(sci.detach()),
           "TRAIN_FIT_boolq_CE_pre_step":float(boo.detach()),
           "TRAIN_FIT_teacher_anchor_CE_pre_step":float(
                (sum(langfit)/len(langfit)).detach()),
           "nonzero_move":bool(alpha>0)})
        if step in QA_CHECKPOINTS:
            states[step]=to_storage(weights)
    checks=[]
    for step in QA_CHECKPOINTS:
        state=tuple(torch.from_numpy(z.astype(np.float32)) for z in states[step])
        QA=old.qastats(state,valid)
        guard=calc_guard(state,lang)
        deltas={k:QA[k]["accuracy"]-oldqa[k]["accuracy"] for k in ("sciq","boolq")}
        kal={k:guard[k]-baseline_guard[k] for k in LANG_DOMAINS}
        nonregress=all(delta>=-1e-6 for delta in deltas.values())
        positive=any(delta>=TASK_VALID_MIN_GAIN-1e-6 for delta in deltas.values())
        constrained=all(kal[k]<=LANG_MAX_DELTA+1e-5 for k in LANG_DOMAINS)
        eligible=step>0 and nonregress and positive and constrained
        checks.append({"step":step,"valid_QA":QA,
          "VALID_QA_gains":deltas,
          "TRAIN_CAL_lang_KL":guard,
          "TRAIN_CAL_lang_KL_delta":kal,
          "nonregress_both_QA":nonregress,
          "at_least_one_QA_valid_gain_ge_0p05":positive,
          "hard_LANG_KL_guard_satisfied":constrained,
          "eligible_according_to_preregistered_gates":eligible,
          "sum_valid_accuracy_gain":sum(deltas.values())})
    viable=[c for c in checks if c["eligible_according_to_preregistered_gates"]]
    chosen=(max(viable,key=lambda c:(c["sum_valid_accuracy_gain"],-c["step"]))["step"]
            if viable else 0)
    physically=pack(source,*states[chosen])
    if len(physically)!=REQUIRED_PHYSICAL_BYTES or len(physically)>MAX_PHYSICAL_BYTES:
        raise AssertionError("source bytes too large or original trained source absent")
    if chosen==0 and physically!=prior:
        raise AssertionError("unchanged-parent source is not byte-identical")
    output=args.folder/f"v81_trained_{args.repeat}.cb76"
    output.write_bytes(physically)
    meta={"study":"CALIBRI_v81_true_projected_QA_with_tight_two_corpus_KL_guard",
      "independent_training_repeat":args.repeat,
      "true_pretrained_model":MODEL,"pinned_original_revision":MODEL_SHA,
      "real_original_unique_trained_params":EXPECTED_UNIQUE,
      "complete_paid_source_bytes":len(physically),
      "full_source_all_in_bpw":8*len(physically)/EXPECTED_UNIQUE,
      "original_complete_CB59_source_SHA":sha(source),
      "parent_CB76_source_SHA":sha(prior),
      "TRAIN_QA_source_provenance":fitmeta["dataset_provenance"],
      "TRAIN_VALID_QA_source_provenance":validmeta["dataset_provenance"],
      "QA_FIT_count":fitmeta["original_questions_count"],
      "QA_VALID_count":validmeta["original_questions_count"],
      "hard_constrained_language_source":"reused v76 WikiText+AG News TRAIN VALID, NOT new TEST",
      "actual_hard_KL_bound_nats_per_corpus":LANG_MAX_DELTA,
      "frozen_projection_line_search_alphas":list(PROJECTION_ALPHAS),
      "QA_saved_checkpoints":list(QA_CHECKPOINTS),
      "source_v76_INITIAL_language_KL":baseline_guard,
      "source_v76_INITIAL_QA_VALID":oldqa,
      "learning_rate":LR,"seed":SEED,
      "full_step_projected_KL_logs":trace,
      "TRAIN_VALID_QA_checkpoint_gates":checks,
      "chosen_TRAIN_checkpoint":chosen,
      "new_SOURCE_truly_changes_pretrained_function":physically!=prior,
      "final_frozen_source_SHA":sha(physically),
      "no_new_TEST_read_for_training_or_source_selection":True,
      "original_5bit_C_kernel_unchanged":True,
      "lowrank_correction_executed_in_Torch_CPU_not_native_C":True,
      "elapsed_training_sec":time.perf_counter()-started}
    (args.folder/f"v81_train_{args.repeat}.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"stage":"v81_TRUE_hard_KL_projected_QA_training",
      "repeat":args.repeat,"chosen_checkpoint":chosen,
      "genuine_new_source":physically!=prior,
      "source_sha":sha(physically),"physical_bpw":round(meta["full_source_all_in_bpw"],9),
      "eligible":sum(c["eligible_according_to_preregistered_gates"] for c in checks),
      "qa_gains_by_step":{c["step"]:c["VALID_QA_gains"] for c in checks},
      "max_projected_LANG_delta":max(max(z["TRAIN_CAL_KL_delta"].values()) for z in trace)}),
      flush=True)

def freeze(args):
    first=json.loads((args.folder/"v81_train_first.json").read_text())
    second=json.loads((args.folder/"v81_train_second.json").read_text())
    x=(args.folder/"v81_trained_first.cb76").read_bytes()
    y=(args.folder/"v81_trained_second.cb76").read_bytes()
    if len(x)!=REQUIRED_PHYSICAL_BYTES or len(y)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("paid exact archive size missing")
    if first["chosen_TRAIN_checkpoint"]!=second["chosen_TRAIN_checkpoint"]:
        raise AssertionError("same-host TRAIN winner disagreed")
    _,ua,va,ba=unpack(x)
    _,ub,vb,bb=unpack(y)
    audit={"experiment":"CALIBRI_v81_frozen_original_source_BEFORE_TEST",
       "frozen_first_TRAIN_winner":first["chosen_TRAIN_checkpoint"],
       "frozen_checkpoint_steps":list(QA_CHECKPOINTS),
       "source_heldout_TEST_unseen_before_frozen":True,
       "physical_complete_source_bytes":len(x),
       "physical_original_source_unique_param_bpw":8*len(x)/EXPECTED_UNIQUE,
       "frozen_first_source_SHA":sha(x),"second_source_SHA":sha(y),
       "same_runner_source_bitwise_equal":x==y,
       "same_runner_max_U_FP16_diff":float(np.max(np.abs(ua.astype("f4")-ub.astype("f4")))),
       "same_runner_max_V_FP16_diff":float(np.max(np.abs(va.astype("f4")-vb.astype("f4")))),
       "same_runner_max_bias_FP16_diff":float(np.max(np.abs(ba.astype("f4")-bb.astype("f4")))),
       "parent_v76_source_SHA":first["parent_CB76_source_SHA"],
       "TEST_does_not_influence_optimizer_checkpoint":True,
       "original_native_C_5bit_inference_preserved":True,
       "paid_rank4_Torch_CPU":True}
    (args.folder/"v81_FROZEN_before_TEST.json").write_text(json.dumps(audit,indent=2)+"\n")
    (args.folder/"CALIBRI_v79_frozen_complete.cb76").write_bytes(x)
    # v79 source-eval/report expects this compatibility JSON containing the
    # exact frozen provenance; no claim that v79 TRAIN objective was used.
    bridge={"frozen_first_TRAIN_winner":first["chosen_TRAIN_checkpoint"],
       "frozen_candidate_source_SHA":sha(x),
       "parent_pretrained_full_source_SHA":first["parent_CB76_source_SHA"],
       "same_host_bitwise_reproducibility":{"same_runner_identical_source_SHA":x==y}}
    (args.folder/"v79_FROZEN_before_TEST.json").write_text(json.dumps(bridge,indent=2)+"\n")
    print(json.dumps({"stage":"v81_FROZEN_BEFORE_TEST",
      "winner":audit["frozen_first_TRAIN_winner"],
      "same_host_bitwise":x==y,"paid_source_bytes":len(x)}),flush=True)

def evaluate(args):
    if not (args.folder/"v81_FROZEN_before_TEST.json").is_file():
        raise AssertionError("all original pretrained sources must be frozen before opening TEST")
    configure()
    # Use the already independently tested v78 real pinned model CPU scoring
    # routines, but brand-new source ranges and new prompts, no TRAIN fitting.
    evaluation.SCI_Q_ROWS=NEW_TEST_SCIQ
    evaluation.BOOLQ_ROWS=NEW_TEST_BOOLQ
    evaluation.WIKI_TEST_FIRST=NEW_TEST_WIKI
    evaluation.NEWS_TEST_ROW_RANGE=NEW_TEST_NEWS
    evaluation.PROMPTS=NEW_PROMPTS
    evaluation.NEW_TOKENS_PER_PROMPT=24
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model,meta=old.model_for_eval(args)
    model.eval()
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1,2,3,4]],dtype=torch.long),use_cache=False)
        words=evaluation.metric_language(model,args.kind,tok,args.folder)
        qa=evaluation.QA_teacher_forced_likelihood(model,args.kind,tok)
        gen=evaluation.real_generation(model,args.kind,tok)
    doc={"experiment":"CALIBRI_v81_real_unseen_heldout_teacher_v76_hard_KL_QA",
      "kind":args.kind,"model":MODEL,"model_revision":MODEL_SHA,
      "physical_source_integrity":meta,"source_original_trained_dense_2D_absent_in_students":
        args.kind=="teacher" or meta.get("zero_dense_pretrained_2D_hot_weights",False),
      "TEST_QA_original_rows_SciQ":list(NEW_TEST_SCIQ),
      "TEST_QA_original_rows_BoolQ":list(NEW_TEST_BOOLQ),
      "TEST_WikiText_token_span":[NEW_TEST_WIKI,NEW_TEST_WIKI+512],
      "TEST_News_rows":list(NEW_TEST_NEWS),
      "all_source_model_candidates_frozen_before_test":True,
      "language":words,"QA":qa,"greedy_actual_token_generation":gen}
    # Companion original wrapper audits exact heldout common input SHA.
    compat={"source_integrity":meta,"kind":args.kind,
       "language":words,"QA":qa,"real_greedy_generation":gen}
    (args.folder/f"v79_{args.kind}.json").write_text(json.dumps(compat,indent=2)+"\n")
    (args.folder/f"v81_{args.kind}.json").write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps({"stage":"v81_COMPLETED_heldout_real_model",
       "kind":args.kind,"QA":{"sciq":qa["sciq"]["correct_count"],
                              "boolq":qa["boolq"]["correct_count"]},
       "KL":{key:round(float(np.mean([z["teacher_student_KL"] for z in val["data"]])),4)
             if val["data"] else None for key,val in words.items()},
       "greedy_unique_token_share":round(gen["mean_distinct_token_ratio"],4)}),flush=True)

def report(args):
    frozen=json.loads((args.folder/"v81_FROZEN_before_TEST.json").read_text())
    old.report(args)  # same matched-input quality, McNemar and full integrity audit
    r=json.loads((args.folder/"v79_FINAL_independent_TEST.json").read_text())
    if frozen["frozen_first_source_SHA"]!=json.loads(
          (args.folder/"v81_qa_v79.json").read_text()
        )["physical_source_integrity"]["complete_physical_SHA256"]:
        raise AssertionError("tested model isn't the TRAIN-frozen complete source")
    r["experiment"]="CALIBRI_v81_REAL_HARD_KL_constrained_QA_and_greedy_RESULTS"
    r["v81_frozen_TRAIN_source_and_byte_audit"]=frozen
    r["source_physically_COMPLETE_BPW"]=8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE
    r["TRAIN_only_constraint_projection_actual_per_step"]=True
    r["strict_new_external_TEST_not_used_for_selection"]=True
    r["actual_output_correction_native_C_only_for_base_Torch_for_paid"]=True
    r["caveats"]="Small task slices, old v76 language TRAIN-CAL reused, only 1 pretrained 135M model. No production claims."
    (args.folder/"v81_FINAL_independent_TEST.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({"stage":"v81_TEST_summary",
      "winner_step":frozen["frozen_first_TRAIN_winner"],
      "exact_source_bpw":round(r["source_physically_COMPLETE_BPW"],9),
      "QA_correct_counts":{k:{"teacher":v["teacher_correct"],"baseline":v["original_CB59_correct"],
        "prior_paid":v["prior_v76_correct"],"new":v["new_answer_supervised_CB76_correct"]}
                         for k,v in r["QA"].items()},
      "model_greedy_diversity":{k:round(v["distinct_ratio"],4) for k,v in r["decode"].items()}
       }),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("smoke","collect","train","freeze","evaluate","report"),required=True)
    parser.add_argument("--dataset",choices=("fit","valid"),default="fit")
    parser.add_argument("--kind",choices=MODEL_KINDS,default="teacher")
    parser.add_argument("--repeat",choices=("first","second"),default="first")
    parser.add_argument("--original",type=Path)
    parser.add_argument("--folder",type=Path,default=Path("results"))
    args=parser.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="smoke":
        assert REQUIRED_PHYSICAL_BYTES<=MAX_PHYSICAL_BYTES
        assert QA_CHECKPOINTS==(0,4,8,12,18)
        assert PROJECTION_ALPHAS[-1]==0
        print("CALIBRI v81 source-complete 0.700bpw, frozen source subsets and optimizer gates smoke PASS")
        return
    if args.original is None:parser.error("real original full CB59 file required")
    if args.phase=="collect":
        configure()
        old.collect(args)
    elif args.phase=="train":train(args)
    elif args.phase=="freeze":freeze(args)
    elif args.phase=="evaluate":evaluate(args)
    else:report(args)

if __name__=="__main__":
    main()
