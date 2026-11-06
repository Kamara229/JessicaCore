"""CALIBRI v215 — teacher-output imitation on wide native TRAIN contexts.

Diagnosed failure: v206 gets only 3/16 DEV (two are coincidental No answers)
and v213/v214 correction of only a few TRAIN positions does not generalize.
v215 never changes original 135M 5-bit packed source; its low-rank
rank-4 + bias head correction is trained on *hundreds of distinct context
positions* from frozen public TRAIN ONLY, directly matching FP32's
full 49,152-token output distribution and teacher top-one prediction.

TRAIN-only held-out source records choose one checkpoint / shrinkage from
precommitted options. Save actual FP16 sidecar and SHA BEFORE opening public
DEV/VALID QA or native speed. Physical source + entire sidecar <=0.700 bpp.
No bigger models or LIVE changes. Public DEV is already repeatedly viewed
in earlier experiments, and cannot be claimed as independent final TEST.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v186.run import speed_pairs
from v196 import protocol as p
from v196.data import training_inputs, positions_and_targets
from v196.run import newly_frozen_public_dev_16
from v212.run import SOURCE_V206_SHA256
from v214.calibrate import attach_head_only_correction

RANK = 4
SOURCE_EVAL_MAX_TOKENS = 192
NATURAL_TEXT_POSITION_STRIDE = 2
TRAIN_BATCH_SIZE = 24
TRAIN_STEPS = 120
CHECKPOINTS = (30, 60, 120)
GAIN_CHOICES = (0.25, 0.5, 1.0)
TRAIN_RESEARCH_SEED = 215
TRAIN_HELDOUT_TEXT_MOD = 5
TRAIN_HELDOUT_PAIRS_MOD = 3
LR = 0.009
REGULARIZATION = 0.0002
MAX_ALL_INCLUSIVE_BPP = 0.700


@torch.no_grad()
def collect_native_teacher_slices(fp32, native, tokenizer):
    corpus, pairs, _, provenance = training_inputs(tokenizer)
    if len(corpus)!=20 or len(pairs)!=24:
        raise AssertionError("The only allowed source is the frozen TRAIN")
    storage={x:{"hidden":[],"baseline":[],"teacher":[],"answer":[]}
             for x in ("FIT","HELDOUT")}
    by_kind=Counter()
    for index, row in enumerate(corpus):
        if row["kind"] in ("wiki","news"):
            token_ids=row["ids"][:SOURCE_EVAL_MAX_TOKENS].long()
            positions=torch.arange(8,len(token_ids)-1,NATURAL_TEXT_POSITION_STRIDE,dtype=torch.long)
        elif row["kind"]=="supervised_chat":
            positions,_=positions_and_targets(row)
            if int(positions.max())>=SOURCE_EVAL_MAX_TOKENS:
                raise AssertionError("Supervised TRAIN chat prompt exceeds frozen 192-token cap")
            token_ids=row["ids"][:max(int(positions.max())+2,20)].long()
        else:
            raise AssertionError("Unsupported or held-out data source")
        if not len(positions):
            raise AssertionError("Missing positions on frozen TRAIN text")
        split="HELDOUT" if index%TRAIN_HELDOUT_TEXT_MOD==0 else "FIT"
        _capture(fp32,native,token_ids[None],positions,storage[split],None)
        by_kind[split+"_"+row["kind"]]+=len(positions)
    for pair_index,pair in enumerate(pairs):
        if pair.get("split")!="TRAIN":
            raise AssertionError("Attempted to train on VALID")
        split="HELDOUT" if pair_index%TRAIN_HELDOUT_PAIRS_MOD==0 else "FIT"
        for side in ("left","right"):
            row=pair[side]
            ids=row["prompt_ids"].long()
            if len(ids)>SOURCE_EVAL_MAX_TOKENS:
                raise AssertionError("Opposing TRAIN prompt too long")
            position=torch.tensor([len(ids)-1],dtype=torch.long)
            _capture(fp32,native,ids[None],position,storage[split],
                     int(row["first_expected_id"]))
            by_kind[split+"_opposing_pairs"]+=1
    output={}
    for split in ("FIT","HELDOUT"):
        data=storage[split]
        output[split]={key:torch.cat(value,dim=0) for key,value in data.items()}
        n=len(output[split]["hidden"])
        if n < 60 or output[split]["baseline"].shape!=(n,p.VOCAB_SIZE):
            raise AssertionError("Insufficient broad public TRAIN or missing full vocabulary")
        if output[split]["hidden"].shape!=(n,576):
            raise AssertionError("Real 135M native model hidden dimension changed")
    if by_kind["HELDOUT_opposing_pairs"]!=16 or by_kind["FIT_opposing_pairs"]!=32:
        raise AssertionError("Leaked holdout TRAIN pairs into FIT")
    return output,dict(by_kind),provenance


@torch.no_grad()
def _capture(teacher,native,ids,positions,dest,gold):
    if ids.ndim!=2 or len(ids[0])>SOURCE_EVAL_MAX_TOKENS:
        raise AssertionError("Unsafe TRAIN model window")
    x=[]
    def capture(_module,inputs):
        x.append(inputs[0].detach().float().cpu())
    handle=native.lm_head.register_forward_pre_hook(capture)
    try:
        compact=native(input_ids=ids,use_cache=False).logits[0,positions].float().cpu()
    finally:
        handle.remove()
    if len(x)!=1:
        raise AssertionError("Native packed C output prehook missed")
    hidden=x[0][0,positions]
    full=teacher(input_ids=ids,use_cache=False).logits[0,positions].float().cpu()
    if compact.shape!=full.shape or hidden.shape[0]!=len(positions):
        raise AssertionError("Nonmatching original FP32/native TRAIN windows")
    if not torch.isfinite(compact).all() or not torch.isfinite(full).all():
        raise AssertionError("Nonfinite native / teacher output")
    dest["hidden"].append(hidden.detach().contiguous())
    dest["baseline"].append(compact.detach().contiguous())
    dest["teacher"].append(full.detach().contiguous())
    dest["answer"].append(torch.full((len(positions),),gold if gold is not None else -1,dtype=torch.int64))


def frozen_cpu_weights(parameters):
    return {key:obj.detach().to(torch.float16).cpu().contiguous() for key,obj in parameters.items()}


@torch.no_grad()
def heldout_metric(dataset,weights,gain,batched=32):
    # TRAIN-only split. Quantize the *actual stored FP16* weights BEFORE the
    # checkpoint decision, so no imaginary extra precision is credited.
    vals={k:v.float() for k,v in weights.items()}
    n=len(dataset["hidden"])
    teacher_choice=0
    gold_correct=0
    distinct=set()
    for start in range(0,n,batched):
        h=dataset["hidden"][start:start+batched]
        baseline=dataset["baseline"][start:start+batched]
        teacher=dataset["teacher"][start:start+batched]
        pred=baseline+gain*vals["bias"][None,:]
        pred=pred+gain*((h@vals["B"])@vals["A"].T)
        chosen=pred.argmax(-1)
        original=teacher.argmax(-1)
        teacher_choice+=int((chosen==original).sum())
        distinct.update(chosen.tolist())
        gold=dataset["answer"][start:start+batched]
        keep=gold>=0
        gold_correct+=int((chosen[keep]==gold[keep]).sum())
    return {"teacher_agreement":teacher_choice,
            "frozen_TRAIN_gold_correct":gold_correct,
            "distinct_output_tokens":len(distinct),
            "n_positions":n,
            "n_gold_pairs":int((dataset["answer"]>=0).sum())}


def choose_candidate(samples,heldout):
    torch.manual_seed(TRAIN_RESEARCH_SEED)
    # A,B are the only rank-4 factors in the physically paid sidecar.
    A=torch.nn.Parameter(torch.randn(p.VOCAB_SIZE,RANK)*0.001)
    B=torch.nn.Parameter(torch.randn(576,RANK)*0.01)
    bias=torch.nn.Parameter(torch.zeros(p.VOCAB_SIZE))
    optim=torch.optim.Adam([A,B,bias],lr=LR)
    n=len(samples["hidden"])
    generator=torch.Generator().manual_seed(TRAIN_RESEARCH_SEED)
    batches=[]
    selected=[]
    for step in range(1,TRAIN_STEPS+1):
        ids=torch.randint(0,n,(TRAIN_BATCH_SIZE,),generator=generator)
        hidden=samples["hidden"][ids]
        raw=samples["baseline"][ids]
        teacher=samples["teacher"][ids]
        logits=raw+((hidden@B)@A.T)+bias[None]
        # Each TRAIN position uses the full FP32 vocabulary distribution.
        # Teacher argmax term prioritizes actual first-token behavior.
        soft=F.softmax(teacher,dim=-1)
        distill=-(soft*F.log_softmax(logits,dim=-1)).sum(-1).mean()
        top=F.cross_entropy(logits,teacher.argmax(-1))
        gold=samples["answer"][ids]
        mask=gold>=0
        if mask.any():
            answer=F.cross_entropy(logits[mask],gold[mask])
        else:
            answer=logits.sum()*0.0
        regularization=REGULARIZATION*(bias.square().mean()+
                      A.square().mean()+B.square().mean())
        loss=0.55*distill+0.30*top+0.15*answer+regularization
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite v215 full-vocab TRAIN loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A,B,bias],max_norm=5.0,error_if_nonfinite=True)
        optim.step()
        if step in CHECKPOINTS:
            weights=frozen_cpu_weights({"A":A,"B":B,"bias":bias})
            for gain in GAIN_CHOICES:
                measured=heldout_metric(heldout,weights,gain)
                selected.append({"checkpoint":step,"gain":gain,
                                 "weights":weights,"TRAIN_heldout":measured})
                print(json.dumps({"stage":"v215_precommit_TRAIN_checkpoint",
                    "step":step,"gain":gain,"TRAIN_holdout":measured}),flush=True)
    # Checkpoint choice is 100% on the disjoint public TRAIN holdout.
    # The first goal is teacher first-token response diversity/generalization,
    # not a memorized single Yes/No answer.
    best=max(selected,key=lambda c:(
        c["TRAIN_heldout"]["teacher_agreement"],
        c["TRAIN_heldout"]["frozen_TRAIN_gold_correct"],
        c["TRAIN_heldout"]["distinct_output_tokens"],
        -c["checkpoint"],-c["gain"]))
    report={"candidates":[{k:v for k,v in c.items() if k!="weights"} for c in selected],
            "selection_rule":"TRAIN_heldout_teacher_top1_then_frozen_TRAIN_gold_then_diversity",
            "selected_checkpoint":best["checkpoint"],"selected_gain":best["gain"],
            "baseline_TRAIN_heldout":heldout_metric(heldout,{
                "A":torch.zeros_like(best["weights"]["A"]),
                "B":torch.zeros_like(best["weights"]["B"]),
                "bias":torch.zeros_like(best["weights"]["bias"])},1.0)}
    return best["weights"],report


def run(v206_path,v161_path,v193_path,out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(TRAIN_RESEARCH_SEED);torch.set_num_threads(2)
    raw=Path(v206_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_V206_SHA256 or len(raw)!=p.SOURCE_BYTES:
        raise AssertionError("Authentic original v206 frozen complete 135M required")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as td:
        tmp=Path(td)
        for name in ("TRAIN_source","CB59","CB206","GT215"):(tmp/name).mkdir()
        native,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_path),folder=tmp/"TRAIN_source"))
        teacher=original_teacher().cpu().eval()
        samples,sample_origins,provenance=collect_native_teacher_slices(teacher,native,tok)
        del native,teacher
        weights,selection=choose_candidate(samples["FIT"],samples["HELDOUT"])
        saved=out/"GREEN_TUNNEL_v215_rank4_full_vocab_native_activated.safetensors"
        save_file(weights,str(saved))
        total_bytes=len(raw)+saved.stat().st_size
        bpp=total_bytes*8/p.FROZEN_SOURCE_135M_UNIQUE
        if bpp>MAX_ALL_INCLUSIVE_BPP or saved.stat().st_size<=0:
            raise AssertionError("Unpaid Green Tunnel runtime weights or missing sidecar")
        seal={
            "experiment":"v215_true_native_hidden_full_vocab_teacher_behavior_TRAIN_only",
            "v206_source_sha256":SOURCE_V206_SHA256,
            "v206_source_bytes":len(raw),
            "source_param_count":p.FROZEN_SOURCE_135M_UNIQUE,
            "real_GreenTunnel_sidecar_file":saved.name,
            "real_GreenTunnel_sidecar_sha256":hashlib.sha256(saved.read_bytes()).hexdigest(),
            "real_GreenTunnel_sidecar_bytes":saved.stat().st_size,
            "all_including_sidecar_bpp":bpp,
            "fp16_rank":RANK,
            "trained_NATIVE_hidden_from_unchanged_5bit_source":True,
            "full_vocabulary_teacher_training_token_count":len(samples["FIT"]["hidden"]),
            "full_vocabulary_teacher_heldout_token_count":len(samples["HELDOUT"]["hidden"]),
            "TRAIN_origin_counts":sample_origins,
            "chosen_checkpoint_and_gain_from_TRAIN_HOLDOUT_ONLY":selection,
            "NO_DEV_VALID_TEST_or_Jessica_used_to_fit_or_select":True,
            "no_unpaid_FP32_teacher_on_inference":True,
            "no_LIVE_main_Render_Supabase_APK_changes":True,
            "SEALED_BEFORE_QA":True
        }
        save(out/"SEALED_v215_green_tunnel_BEFORE_ANY_QA.json",seal)
        print(json.dumps({"stage":"v215_physically_sealed_before_QA",
                          "bpp":bpp,"checkpoint":selection["selected_checkpoint"],
                          "gain":selection["selected_gain"],"TRAIN_baseline":selection["baseline_TRAIN_heldout"],
                          "TRAIN_selected":next(c["TRAIN_heldout"] for c in selection["candidates"]
                               if c["checkpoint"]==selection["selected_checkpoint"] and c["gain"]==selection["selected_gain"])
                          }),flush=True)
        original=extract_original(Path(v161_path))
        if len(original.blob)!=p.SOURCE_BYTES:
            raise AssertionError("Native CB59 CPU baseline missing")
        (tmp/"CB59"/"source.cb59").write_bytes(original.blob)
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=tmp/"CB59"/"source.cb59",folder=tmp/"CB59"))
        cb206,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_path),folder=tmp/"CB206"))
        student,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_path),folder=tmp/"GT215"))
        if any(param.ndim>=2 for param in student.parameters()):
            raise AssertionError("Native packed C source replaced by dense model")
        handle=attach_head_only_correction(student,load_file(str(saved)),selection["selected_gain"])
        try:
            qa=newly_frozen_public_dev_16({"CB206":cb206,"GT215":student},tok,v193_path,out)
            if qa["scores"]["CB206"]["machine_confirmed"]!=3:
                raise AssertionError("Frozen v206 public DEV changed")
            speed=speed_pairs({"CB59":cb59,"CB186":student},tok,freeze_cases(all_cases()))
        finally:
            handle.remove()
        result={
            "completed":True,
            "v215_model":"original_135M_native_CB59_plus_physically_paid_rank4_head_only",
            "physical_all_in_bpp":bpp,
            "v206_correct_PUBLIC_DEV_16":qa["scores"]["CB206"]["machine_confirmed"],
            "FP32_correct_PUBLIC_DEV_16":qa["scores"]["FP32_existing_135M"]["machine_confirmed"],
            "v215_correct_PUBLIC_DEV_16":qa["scores"]["GT215"]["machine_confirmed"],
            "v215_wrong_PUBLIC_DEV_16":qa["scores"]["GT215"]["machine_recognized_wrong"],
            "v215_ungraded_PUBLIC_DEV_16":qa["scores"]["GT215"]["ungraded_not_known_wrong"],
            "native_CPU_speed_ratio_CB59":speed["median_speed_ratio_vs_CB59"],
            "PUBLIC_DEV16_previously_opened_iterative_diagnostics_NOT_INDEPENDENT_TEST":True,
            "not_selected_based_on_dev_score":True,
            "LIVE_untouched":True
        }
        save(out/"v215_REAL_NATIVE_135m_full_answer_QA_CPU.json",result)
        print(json.dumps({"stage":"v215_final_real_native_QA",**result}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("v206_source",type=Path)
    ap.add_argument("v161_original_CB59",type=Path)
    ap.add_argument("v193_original_FP32_QA",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.v206_source,args.v161_original_CB59,args.v193_original_FP32_QA,args.output)
