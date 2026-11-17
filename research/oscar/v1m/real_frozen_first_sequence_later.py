#!/usr/bin/env python3
"""Oscar v1M REAL original FP32 next-token-sequence Green Tunnel research.

Only physically paid ORIGINAL rank32 LATER GT tensors are trained on
teacher-greedy AND actual compressed student-generated prefixes. FIRST GT
and independently pretrained 42M original 8×512 source K32/V8 (including
four v1K corrected source codebooks) are FROZEN bitwise. No CALIBRI S142.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from oscar.v1e.audit_checkpoint import source as external_fp32
from oscar.v1e.safe_pretrained_llama import (
    from_checkpoint, materialize_decoded_cb59,
)
from oscar.v1f.green_tunnel import FullLayerGreenTunnel,DEPTH,WIDTH
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,TOKENIZER,TOKENIZER_REV,
    EXPECTED_ORIGINAL_UNIQUE,TRAIN_STORIES,EVAL_OLD,EVAL_OTHER,
    prompt_ids,true_greedy,
)
from oscar.v1g.real_onpolicy_gt8 import (
    V1F_GT_SHA,DEBUG_NEW,
)
from oscar.v1h.real_dual_phase_inference import (
    V1G_SHA,NEW_DEBUG as V1H_PUBLIC_DEBUG,PHASES,dual_phase_greedy,
)
from oscar.v1i.real_functional_codebook import NEW_TEST_DEBUG
from oscar.v1k.real_joint_source_gt8 import EXTRA_DEBUG
from oscar.v1l.causal_source_gt_ablation import (
    NEW_CB59_SHA,NEW_CB61_SHA,
)

SEED=4219119
STEPS=112
TEMP=2.
MAX_POLICY_NEW=5
MANIFEST_BYTES=3072
LIMIT=int(.7*EXPECTED_ORIGINAL_UNIQUE/8)

# Structured authored TRAIN-only calibration, not actual original training
# distribution, not a blinded QA. Each of 10 names and 8 templates unique.
NAMES=("Alfred","Beth","Clara","Daisy","Eddie",
       "Frances","George","Helen","Ivan","Julia")
TEMPLATES=(
   "On Tuesday, {name} noticed a small butterfly outside the school.",
   "In the summer, {name} took a warm loaf of bread to the neighbor.",
   "Before bedtime, {name} remembered a funny story about a hungry bear.",
   "At the park, {name} watched two rabbits play beneath the trees.",
   "During lunch, {name} shared a bright red apple with a friend.",
   "When the thunder ended, {name} opened the window to see the flowers.",
   "After a long walk, {name} put the muddy boots beside the door.",
   "In the morning, {name} helped wash the little blue bicycle.",
)
TRAIN=tuple(TRAIN_STORIES)+tuple(
   template.format(name=name)
   for name in NAMES for template in TEMPLATES
)
NEW_DEBUG=(
    "The cheerful farmer carried fresh eggs to the village market.",
    "After the snow melted, a tiny frog jumped into the pond.",
    "Every evening, the baker closed his shop before sunset.",
    "The curious girl found a secret note under her pillow.",
    "A family of squirrels gathered nuts near the oak tree.",
    "The small sailboat drifted slowly across the quiet lake.",
    "Henry helped his grandfather build a wooden birdhouse.",
    "At dawn, the little fox followed footprints through the forest.",
)
COHORTS={
    "previously_exposed_36_public_debug":
        EVAL_OLD+EVAL_OTHER+DEBUG_NEW+V1H_PUBLIC_DEBUG+NEW_TEST_DEBUG+EXTRA_DEBUG,
    "eight_new_predeclared_v1M_public_debug":NEW_DEBUG,
}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def dump(path,x):
    Path(path).write_text(json.dumps(x,ensure_ascii=False,
                         sort_keys=True,indent=2)+"\n",encoding="utf8")


def log(stage,**data):
    print(json.dumps({"stage":stage,**data},
                     ensure_ascii=False),flush=True)


def init_gt(student,old_first,later_old,anchor_later=None):
    first=FullLayerGreenTunnel().install(student)
    later=FullLayerGreenTunnel().install(student)
    first.restore_paid(load_file(str(old_first)))
    later.restore_paid(load_file(str(later_old)))
    first.requires_grad_(False)
    # The actual original paid later route gets grads only during training.
    if anchor_later is None:
        return first,later,None
    frozen_later=FullLayerGreenTunnel().install(student)
    frozen_later.restore_paid(load_file(str(anchor_later)))
    frozen_later.requires_grad_(False)
    if not torch.any(frozen_later.up):
        raise AssertionError("Oscar v1M trust anchor must be genuine pretrained v1G GT")
    return first,later,frozen_later


def unhook(*routes):
    for route in reversed(routes):
        if route is not None:route.uninstall()


def real_greedy_student(student,first,later,tok,prompt,max_new=MAX_POLICY_NEW,
                        anchor=None):
    """Actual compressed model's own on-policy tokens: NEVER gold spliced."""
    generated=[]
    ids=prompt.clone()
    with torch.no_grad():
        for i in range(max_new):
            with first.scope(i==0),later.scope(i>0):
                if anchor is None:
                    logits,_=student(ids)
                else:
                    with anchor.scope(False):
                        logits,_=student(ids)
            if (first.last_coverage!=tuple(range(DEPTH)) or
                later.last_coverage!=tuple(range(DEPTH))):
                raise AssertionError("Oscar v1M 8layer real recurrent FIRST/LATER hook not executed")
            next_id=int(logits[0,-1].argmax())
            generated.append(next_id)
            if next_id==tok.eos_token_id:
                break
            ids=torch.cat((ids,ids.new_tensor([[next_id]])),dim=1)
    return generated


@torch.no_grad()
def reference_teacher_tokens(teacher,tok,prompt,max_new=MAX_POLICY_NEW):
    ids=prompt.clone()
    tokens=[]
    for _ in range(max_new):
        logits,_=teacher(ids)
        next_id=int(logits[0,-1].argmax())
        tokens.append(next_id)
        if next_id==tok.eos_token_id:
            break
        ids=torch.cat((ids,ids.new_tensor([[next_id]])),dim=1)
    return tokens


def context_from_continuation(prompt,tokens,tok):
    if (not tokens or len(tokens)>MAX_POLICY_NEW or
        tok.eos_token_id in tokens[:-1]):
        raise AssertionError("Oscar v1M sequential original policy continuation malformed")
    if len(tokens)==1:return prompt.clone()
    return torch.cat((prompt,
        prompt.new_tensor([tokens[:-1]])),dim=-1)


def original_teacher_on_same_history(teacher,ctx):
    with torch.no_grad():
        logits,states=teacher(ctx,hidden=True)
    if (logits.shape!=(1,ctx.shape[1],32000) or
        len(states)!=DEPTH+2):
        raise AssertionError("Oscar v1M independently pretrained FP32 full8 teacher target shape invalid")
    return logits.detach(),[x.detach() for x in states[1:DEPTH+1]]


def trust_and_teacher_loss(student,first,later,anchor,teacher,ctx,stem):
    """Teacher branch and student's own policy use SAME input history.
    
    Use ONLY token positions AFTER first generated token:
        if prompt length P and full ctx P+K-1, target slots [P,...,P+K-2].
    FIRST route does not receive any gradient, and cannot change token#1.
    """
    if ctx.shape[1]<=stem.shape[1]:
        return None
    pos=torch.arange(stem.shape[1],ctx.shape[1],
                     device=ctx.device)
    if pos.numel()<1:
        raise AssertionError("Oscar v1M training has no later tokens")
    teacher_logits,teacher_states=original_teacher_on_same_history(teacher,ctx)
    with torch.no_grad():
        with first.scope(False),later.scope(False),anchor.scope(True):
            anchor_logits,_=student(ctx)
        frozen_policy=anchor_logits[0,pos].detach().float()
    with first.scope(False),later.scope(True),anchor.scope(False):
        student_logits,_=student(ctx)
    if (later.last_coverage!=tuple(range(DEPTH)) or
        len(anchor.latest_layers)!=DEPTH):
        raise AssertionError("Oscar actual original trained LATER 8layer recurrent hook missing")
    s=student_logits[0,pos,:].float()
    t=teacher_logits[0,pos,:].float()
    old=frozen_policy
    if s.shape!=t.shape or s.shape!=old.shape:
        raise AssertionError("Original 42M pretrained same-prefix full32k tensor shape mismatch")
    log_s=F.log_softmax(s/TEMP,dim=-1)
    log_t=F.log_softmax(t/TEMP,dim=-1)
    log_old=F.log_softmax(old/TEMP,dim=-1)
    original_teacher_kl=(
        log_t.exp()*(log_t-log_s)).sum(-1).mean()*TEMP**2
    frozen_policy_kl=(
        log_old.exp()*(log_old-log_s)).sum(-1).mean()*TEMP**2
    correct=t.argmax(-1)
    CE=F.cross_entropy(s,correct)
    competing=s.detach().clone()
    competing.scatter_(1,correct[:,None],-torch.inf)
    highest_other=competing.max(-1).values
    margin=F.softplus(0.6+highest_other-s.gather(
        -1,correct[:,None]).squeeze(-1)).mean()
    state_sim=[]
    for i in range(DEPTH):
        h=anchor.latest_layers[i][0,pos].float()
        ref=teacher_states[i][0,pos].float()
        if h.shape!=ref.shape or h.shape[-1]!=WIDTH:
            raise AssertionError("Original 42M pretrained FP32 all8 hidden-state alignment broken")
        state_sim.append(
            F.cosine_similarity(
                F.layer_norm(h,(WIDTH,)),
                F.layer_norm(ref,(WIDTH,)),dim=-1).mean())
    inner=(1-torch.stack(state_sim)).mean()
    loss=(0.62*original_teacher_kl+
          0.085*CE+
          0.12*margin+
          0.60*inner+
          1.20*frozen_policy_kl)
    if not bool(torch.isfinite(loss)):
        raise AssertionError("Original independently pretrained FP32 later Green Tunnel loss nonfinite")
    details={
      "positions_later_than_first":int(pos.numel()),
      "original_FP32_32k_teacher_KL_T2":float(original_teacher_kl.detach()),
      "old_pretrained_v1G_later_route_TRUST_KL_T2":float(frozen_policy_kl.detach()),
      "original_FP32_teacher_TOP1_cross_entropy":float(CE.detach()),
      "original_teacher_margin":float(margin.detach()),
      "mean_original_FP32_eight_layer_hidden_loss":float(inner.detach()),
    }
    return loss,details


def weight_anchor(later,original_paid):
    # source_paid_tensors detaches and is ONLY for exporting; gradients
    # must flow to real trainable nn.Parameters of this same GT instance.
    real={
       "oscar_v1f.down_fp16":later.down,
       "oscar_v1f.up_fp16":later.up,
       "oscar_v1f.gains_fp16":later.gain,
       "oscar_v1f.mix_fp16":later.mix_logits,
    }
    if set(real)!=set(original_paid):
        raise AssertionError("Original pretrained LATER GT original FP16 anchor keys mismatch")
    return sum((w-original_paid[k].float()).square().mean()
               for k,w in real.items())/len(real)


@torch.inference_mode()
def public_benchmark(teacher,student,first,later,tok):
    cases=[]
    for cohort,prompts in COHORTS.items():
        for q in prompts:
            ids=prompt_ids(tok,q)
            t_ans=true_greedy(teacher,tok,ids)
            s_ans=dual_phase_greedy(student,teacher,first,later,tok,ids,PHASES[2])
            first_score=None
            with first.scope(True),later.scope(False):
                score,_=student(ids)
                first_score=score[0,-1].detach().float().cpu().contiguous()
            if first.last_coverage!=tuple(range(DEPTH)):
                raise AssertionError("Original pretrained FIRST GT does not actually traverse all8 layers")
            emitted=s_ans["generated_original_Oscar_token_ids"]
            if not emitted:
                raise AssertionError("Oscar real original 5bit 42M generated zero tokens")
            cases.append({
               "cohort":cohort,
               "original_prompt_SHA256":sha(q.encode()),
               "prompt":q,
               "original_independent_pretrained_FP32_8token_and_EOS":t_ans,
               "original_5bit_model_FULL8_GT_own_greedy":s_ans,
               "exact_pretrained_FP32_FIRST":emitted[0]==t_ans["ids"][0],
               "exact_pretrained_FP32_COMPLETE8_and_EOS":emitted==t_ans["ids"],
               "frozen_source_and_first_GT_next32k_logit_sha256":sha(
                   first_score.numpy().tobytes()),
            })
        log("OSCAR_V1M_ORIGINAL_FP32_REAL_PUBLIC_DEBUG",
            debug_group=cohort,seen=len(cases),
            first=sum(int(r["exact_pretrained_FP32_FIRST"]) for r in cases),
            full=sum(int(r["exact_pretrained_FP32_COMPLETE8_and_EOS"])
                     for r in cases))
    return cases


def scores(rows):
    n=len(rows)
    return {
      "public_DEBUG_n":n,
      "original_FP32_exact_first_count":sum(
          int(r["exact_pretrained_FP32_FIRST"]) for r in rows),
      "original_FP32_exact_full8_EOS_count":sum(
          int(r["exact_pretrained_FP32_COMPLETE8_and_EOS"]) for r in rows),
      "mean_original_FP32_last_layer_cos_on_actual_student_prefix":sum(
          r["original_5bit_model_FULL8_GT_own_greedy"][
          "mean_last_hidden_cosine_on_OWN_student_prefix"]
          for r in rows)/n,
      "mean_unigram_repeat":sum(
          r["original_5bit_model_FULL8_GT_own_greedy"][
          "actual_repetition"]["unigram_repeat"] for r in rows)/n,
      "mean_bigram_repeat":sum(
          r["original_5bit_model_FULL8_GT_own_greedy"][
          "actual_repetition"]["bigram_repeat"] for r in rows)/n,
    }


def run(a):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1M research output path must be empty")
    paths={
       "FOUR_MATRIX_original_FP32_TRAIN_calibrated_K32V8_CB59":
          (Path(a.original_CB59),NEW_CB59_SHA),
       "FOUR_MATRIX_original_FP32_TRAIN_calibrated_K32V8_CB61":
          (Path(a.original_CB61),NEW_CB61_SHA),
       "ORIGINAL_UNTOUCHED_v1F_FIRST_GT8":
          (Path(a.v1f_first_FP16),V1F_GT_SHA),
       "ORIGINAL_UNTOUCHED_v1G_LATER_GT8":
          (Path(a.v1g_later_FP16),V1G_SHA),
    }
    for name,(p,h) in paths.items():
        if not p.is_file() or sha(p.read_bytes())!=h:
            raise AssertionError("Oscar v1M original independent 42M source/pretrained GT mismatch: "+name)
    if (len(TRAIN)!=92 or len(set(TRAIN))!=92 or
        len(COHORTS["previously_exposed_36_public_debug"])!=36 or
        len(COHORTS["eight_new_predeclared_v1M_public_debug"])!=8 or
        set(TRAIN)&set(sum((tuple(g) for g in COHORTS.values()),()))):
        raise AssertionError("Oscar v1M genuine TRAIN 92 and all44 public DEBUG have overlap")
    precommit={
      "stage":"Oscar v1M original pretrained 42M frozen FIRST and 4codebook source, train LATER only",
      "original_real_independent_FP32_source_SHA256":SOURCE_SHA,
      "original_real_pretrained_unique":EXPECTED_ORIGINAL_UNIQUE,
      "original_8layer_512width_network_NOT_CALIBRI_12layer":True,
      "original_5bit_after_v1K_source_sha":NEW_CB59_SHA,
      "original_CB61_after_v1K_sha":NEW_CB61_SHA,
      "original_pretrained_v1F_FIRST_GT_SHA":V1F_GT_SHA,
      "original_pretrained_v1G_LATER_GT_SHA":V1G_SHA,
      "original_FIRST_all8_unchanged_and_not_trained":True,
      "original_CB59_CB61_source_unchanged_AND_NOT_TRAINED":True,
      "only_existing_LATER8_GT_FP16_original_4_tensors_trainable":True,
      "training_corpus":"12 existing TRAIN + 80 manually templated calibration sentences; NOT original model training data",
      "train_only_92_stories":list(TRAIN),
      "public_DEBUG44_never_used_for_gradient_or_epoch_selection":COHORTS,
      "optimizer_steps":STEPS,
      "training_both_real_greedy_teacher_and_student_contexts":True,
      "teacher_output_tokens_never_spliced_into_student_own_history":True,
      "all8_decoder_state_matching_after_initial_token":True,
      "literal_second_LATER_GT_frozen_REFERENCE_train_only_unshipped":True,
      "real_pretrained_FP32_FIRST_bitwise_after_reloaded_eq_before_required":True,
      "trained_NEW_GT_ONLY_physically_charged":True,
      "original_FP32_teacher_NOT_at_inference":True,
      "full_source_plus_TWO_GT_manifest_rate_max_bpp":.7,
      "no_CALIBRI_S142_live_NRA2_main_Render_Supabase_APK_616J7_changes":True,
      "native_5bit_decoder_runtime_and_tokenizer_NOT_INCLUDED":True,
      "no_independent_clean_unseen_quality_assessment":True,
    }
    dump(out/"OSCAR_V1M_PRECOMMIT_BEFORE_ACTUAL_ORIGINAL_FP32_MODEL.json",precommit)
    _,source,audit=external_fp32(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Original pretrained TinyStories FP32 41.6896M source not authenticated")
    teacher=from_checkpoint(source).eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Original independently pretrained 42M proxy tokenizer invalid")
    core=paths["FOUR_MATRIX_original_FP32_TRAIN_calibrated_K32V8_CB59"][0]
    first_file=paths["ORIGINAL_UNTOUCHED_v1F_FIRST_GT8"][0]
    later_file=paths["ORIGINAL_UNTOUCHED_v1G_LATER_GT8"][0]
    original=materialize_decoded_cb59(teacher,core.read_bytes()).eval().requires_grad_(False)
    first,later,_=init_gt(original,first_file,later_file)
    before=public_benchmark(teacher,original,first,later,tok)
    initial=scores(before[:36])
    if (initial["original_FP32_exact_first_count"]!=19 or
        initial["original_FP32_exact_full8_EOS_count"]!=0):
        raise AssertionError("Oscar v1M frozen v1K-source plus original first+later control not true v1L 19/36,0/36")
    unhook(first,later)
    model=materialize_decoded_cb59(teacher,core.read_bytes()).eval().requires_grad_(False)
    first,later,anchor=init_gt(model,first_file,later_file,later_file)
    later.requires_grad_(True)
    first.requires_grad_(False)
    anchor.requires_grad_(False)
    originally=load_file(str(later_file))
    opt=torch.optim.Adam([
       {"params":[later.up],"lr":.00006},
       {"params":[later.down],"lr":.000025},
       {"params":[later.gain,later.mix_logits],"lr":.000015},
    ],weight_decay=0)
    trace=[]
    start=time.monotonic()
    try:
        for step in range(STEPS):
            text=TRAIN[(step*37)%len(TRAIN)]
            ids=prompt_ids(tok,text)
            actual=real_greedy_student(
                model,first,later,tok,ids,anchor=anchor)
            reference=reference_teacher_tokens(
                teacher,tok,ids)
            if not actual or not reference:
                raise AssertionError("Student/real pretrained FP32 teacher both must emit genuine continuations")
            own_ctx=context_from_continuation(ids,actual,tok)
            teacher_ctx=context_from_continuation(ids,reference,tok)
            opt.zero_grad(set_to_none=True)
            branch_loss=[]
            metrics={}
            for name,ctx,scale in (
                ("student_real_self_history",own_ctx,1.0),
                ("original_FP32_teacher_own_greedy_history",teacher_ctx,.65),
            ):
                result=trust_and_teacher_loss(
                    model,first,later,anchor,teacher,ctx,ids)
                if result is None:
                    metrics[name]={"no_later_tokens_due_to_actual_EOS":True}
                    continue
                loss,d=result
                branch_loss.append(scale*loss)
                metrics[name]=d
            if not branch_loss:
                raise AssertionError("Oscar v1M both actual autoregressive continuation histories generated no later token")
            anchor_loss=weight_anchor(later,originally)
            objective=sum(branch_loss)+.25*anchor_loss
            if not torch.isfinite(objective):
                raise AssertionError("Original pretrained FP32 teacher real sequence objective nonfinite")
            objective.backward()
            grads={
               "original_pretrained_full8_LATER_UP_grad":float(later.up.grad.norm())
                   if later.up.grad is not None else 0,
               "original_pretrained_full8_LATER_DOWN_grad":float(later.down.grad.norm())
                   if later.down.grad is not None else 0,
            }
            if (not all(v>0 and torch.isfinite(torch.tensor(v)) for v in grads.values())
                or first.up.grad is not None or anchor.up.grad is not None):
                raise AssertionError("Oscar v1M gradient reached forbidden FIRST GT / frozen FP32 model OR not LATER")
            torch.nn.utils.clip_grad_norm_(
                [later.up,later.down,later.gain,later.mix_logits],
                .35,error_if_nonfinite=True)
            opt.step()
            # Explicit trust REGION from real previously pretrained v1G
            # initialization: maximum absolute displacement of any source
            # GT tensor value limited to 0.025 (not just a soft KL loss).
            params={
               "oscar_v1f.up_fp16":later.up,
               "oscar_v1f.down_fp16":later.down,
               "oscar_v1f.gains_fp16":later.gain,
               "oscar_v1f.mix_fp16":later.mix_logits,
            }
            with torch.no_grad():
                for name,p in params.items():
                    p.copy_(torch.minimum(
                        torch.maximum(p,originally[name].float()-.025),
                        originally[name].float()+.025))
            trace.append({
                "step":step+1,
                "train_prompt_sha256":sha(text.encode()),
                "true_student_onpolicy_ids":actual,
                "true_original_FP32_teacher_greedy_ids":reference,
                "same_context_teacher_differentiable_later_branch_metrics":metrics,
                "full_train_objective":float(objective.detach()),
                "reference_original_GT8_weight_anchor":float(anchor_loss.detach()),
                "FIRST_GT_and_original_source_FROZEN":True,
                **grads,
            })
            if (step+1)%16==0:
                log("OSCAR_V1M_ORIGINAL_FP32_LATER_ONLY_SEQUENCE_TRAIN",
                    step=step+1,
                    corpus_used=len({x["train_prompt_sha256"] for x in trace}),
                    original_FP32_teacher_policy_length=len(reference),
                    compressed_student_policy_length=len(actual),
                    loss=round(float(objective.detach()),5),
                    teacher_own_context_KL=round(
                        metrics.get("original_FP32_teacher_own_greedy_history",{}).get(
                            "original_FP32_32k_teacher_KL_T2",-1),4),
                    real_student_policy_KL=round(
                        metrics.get("student_real_self_history",{}).get(
                            "original_FP32_32k_teacher_KL_T2",-1),4),
                    gradient_LATER=round(grads[
                        "original_pretrained_full8_LATER_UP_grad"],4))
        new_paid=later.source_paid_tensors()
        changed={k:int(torch.count_nonzero(v!=originally[k]))
                 for k,v in new_paid.items()}
        if (changed["oscar_v1f.up_fp16"]==0 or
            changed["oscar_v1f.down_fp16"]==0):
            raise AssertionError("No actually changed full8 pretrained LATER UP/DOWN GT FP16 after sequence training")
        trained_path=out/"OSCAR_V1M_ORIGINAL_FP32_SEQUENCE_TRAINED_LATER_GT8_FP16.safetensors"
        save_file(new_paid,str(trained_path))
        reloaded=load_file(str(trained_path))
        if (set(reloaded)!=set(new_paid) or
            any(not torch.equal(reloaded[k],v) for k,v in new_paid.items())):
            raise AssertionError("Oscar v1M new LATER original pretrained FP16 checkpoint not precisely reloaded")
    finally:
        unhook(first,later,anchor)
    dump(out/"OSCAR_V1M_FULL_REAL_PRETRAINED_FP32_112_STEP_LATER_TOKEN_TRAINING.json",trace)
    # Repeat ALL 44 PUBLIC DEBUG after authentic physical FPGA GT and source
    # reconstructed from original CORRECT 5bit source + reloaded FP16.
    fresh=materialize_decoded_cb59(teacher,core.read_bytes()).eval().requires_grad_(False)
    ff,fl,_=init_gt(fresh,first_file,trained_path)
    try:
        after=public_benchmark(teacher,fresh,ff,fl,tok)
    finally:
        unhook(ff,fl)
    if len(before)!=len(after) or len(before)!=44:
        raise AssertionError("Oscar v1M predeclared 36+8 real original pretrained debug incomplete")
    for i,(b,r) in enumerate(zip(before,after)):
        if b["frozen_source_and_first_GT_next32k_logit_sha256"]!=\
           r["frozen_source_and_first_GT_next32k_logit_sha256"]:
            raise AssertionError("Oscar v1M FROZEN FIRST original FP32-pretrained GT first logits changed at debug index "+str(i))
        if b["exact_pretrained_FP32_FIRST"]!=r["exact_pretrained_FP32_FIRST"]:
            raise AssertionError("Oscar v1M FIRST original pretrained 42M token changed despite frozen source/FIRST GT")
    for name,(p,expected) in paths.items():
        if sha(p.read_bytes())!=expected:
            raise AssertionError("Original pretrained model packed source or v1F FIRST/later reference GT modified: "+name)
    # Strict physical: only ONE genuine original encoded source CB61,
    # original real v1F FIRST GT, new real trained LATER GT, and fixed
    # actually charged runtime ABI config. Training-only anchor excluded.
    package=out/"OSCAR_V1M_REAL_ORIGINAL42M_FROZEN_FIRST_NEW_LATER_GT_PAID"
    package.mkdir(parents=True,exist_ok=True)
    shipped={
      "ORIGINAL_PRETRAINED42M_4CODEBOOK_FIXED_CORE.cb61":
          paths["FOUR_MATRIX_original_FP32_TRAIN_calibrated_K32V8_CB61"][0],
      "ORIGINAL_PRETRAINED42M_FROZEN_FIRST_8L_GT_FP16.safetensors":first_file,
      "ORIGINAL42M_NEW_LATER_8L_SEQUENCE_GT_FP16.safetensors":trained_path,
    }
    physical={}
    for label,path in shipped.items():
        dest=package/label
        shutil.copyfile(path,dest)
        physical[label]={"bytes":dest.stat().st_size,
                         "sha256":sha(dest.read_bytes())}
    total=sum(x["bytes"] for x in physical.values())+MANIFEST_BYTES
    if total>LIMIT:
        raise AssertionError("Oscar v1M REAL original 42M FP32 source+2GT physical >.7 bpp")
    old36=scores(before[:36]);new36=scores(after[:36])
    old8=scores(before[36:]);new8=scores(after[36:])
    receipt={
      "stage":"OSCAR v1M originally pretrained FP32 42M frozen-FIRST real LATER sequential GT",
      "original_42M_unique_FP32_trained_params":EXPECTED_ORIGINAL_UNIQUE,
      "original_real_8decoder_layer_hidden512_not_Calibri12layer":True,
      "frozen_original_source_CB61_sha256":NEW_CB61_SHA,
      "frozen_original_first8_GT_FP16_sha256":V1F_GT_SHA,
      "new_later8_GT_retrained_sha256":sha(trained_path.read_bytes()),
      "new_later8_GT_only_physically_modified":True,
      "all_original_source_and_FIRST_logits_BITWISE_IDENTICAL":True,
      "full_complete_model_physical_files":physical,
      "actual_paid_runtime_manifest_bytes":MANIFEST_BYTES,
      "actual_original_5bit_source_plus_2GT_and_config_bytes":total,
      "original_noninflated_41p6896M_bpp":8*total/EXPECTED_ORIGINAL_UNIQUE,
      "strict_0p7_bpp_source_only_PASS":total<=LIMIT,
      "original_pretrained_FP32_teacher_NOT_in_inference":True,
      "tokenizer_and_executable_native_CPU_kernel_NOT_counted_or_TESTED":True,
      "real_teacher_FULL8_greedy36_match":new36["original_FP32_exact_full8_EOS_count"],
      "model_not_accepted_or_deployed":True,
      "no_clean_sealed_QA_or_original616J7":True,
    }
    manifest=(json.dumps(receipt,indent=2,sort_keys=True,
                          ensure_ascii=False)+"\n").encode()
    if len(manifest)>MANIFEST_BYTES:
        raise AssertionError("Oscar v1M physical runtime configuration reservation insufficient")
    (package/"OSCAR_V1M_SOURCE_AND_GT_PHYSICAL_MANIFEST.json").write_bytes(
        manifest+b" "*(MANIFEST_BYTES-len(manifest)))
    if total!=sum(p.stat().st_size for p in package.iterdir()):
        raise AssertionError("Oscar v1M full model physical inference byte accounting falsified")
    report={
       **precommit,
       "real_original_pretrained_FP32_model_source_authenticated":True,
       "original_frozen_FIRST_exact_bitwise_logits_on_44_PUBLIC_DEBUG":True,
       "full_real_original_FP32_teacher_GREEDY_PUBLIC_DEBUG_BEFORE":before,
       "full_real_original_FP32_teacher_GREEDY_PUBLIC_DEBUG_AFTER":after,
       "old36_metrics_before":old36,
       "old36_metrics_after":new36,
       "new8_metrics_before":old8,
       "new8_metrics_after":new8,
       "all44_metrics_before":scores(before),
       "all44_metrics_after":scores(after),
       "all_old_GT_FP16_file_hashes_unchanged":True,
       "new_trained_LATER_GT_FP16_changed_cells":changed,
       "trained_new_LATER_GT_physical_SHA256":sha(trained_path.read_bytes()),
       "total_real_CPU_training_seconds":round(time.monotonic()-start,3),
       "all_real_full_original_SOURCE_plus_BOTH_GT_paid_physical":receipt,
       "clean_QA_and_original616_J7_not_evaluated":True,
       "original_CALIBRI_12layer_43M_FP32_conversion_not_achieved":True,
       "full_original_teacher_language_quality_accepted":False,
    }
    dump(out/"OSCAR_V1M_FINAL_REAL_42M_ORIGINAL_FP32_FROZEN_FIRST_SEQUENCE_LATER_REPORT.json",report)
    log("OSCAR_V1M_FINAL_REAL_PRETRAINED42M_FP32_FROZEN_FIRST_SEQUENCE_LATER",
        train_steps=STEPS,
        train_prompts_distinct_used=len({x["train_prompt_sha256"] for x in trace}),
        old36_first_before=old36["original_FP32_exact_first_count"],
        old36_first_after=new36["original_FP32_exact_first_count"],
        old36_full_before=old36["original_FP32_exact_full8_EOS_count"],
        old36_full_after=new36["original_FP32_exact_full8_EOS_count"],
        new8_first_before=old8["original_FP32_exact_first_count"],
        new8_first_after=new8["original_FP32_exact_first_count"],
        old36_unigram_repeat_before=old36["mean_unigram_repeat"],
        old36_unigram_repeat_after=new36["mean_unigram_repeat"],
        old36_final_hidden_cos_before=old36[
            "mean_original_FP32_last_layer_cos_on_actual_student_prefix"],
        old36_final_hidden_cos_after=new36[
            "mean_original_FP32_last_layer_cos_on_actual_student_prefix"],
        real_total_source_plus2GT_bytes=total,
        actual_true_pretrained_41689600_bpp=8*total/EXPECTED_ORIGINAL_UNIQUE,
        FIRST_logits_bitwise_preserved_on44=True,
        original_FP32_full_language_skills_NOT_PROVEN=True,
        original_CALIBRI_S142_LIVE_UNTOUCHED=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("original_CB59","original_CB61","v1f_first_FP16",
              "v1g_later_FP16","out"):
        p.add_argument(n,type=Path)
    run(p.parse_args())
