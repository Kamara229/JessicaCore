"""CALIBRI2 S0: train the *small* 5bit architecture from randomly packed source.

All actual neural parameters start only in S0 bootstrap CB59 with no teacher
weights. Train existing paid 32x8 FP16 dictionaries (STE) in every layer.
Train/index-propose only using 100% WikiText TRAIN and 24 frozen balanced
synthetic TRAIN pairs. No DEV, VALID, QA, TEST for step or model selection.
Export real full packed archive; probe only original TRAIN questions.
This first bounded run is an engineering/pedagogy milestone, not a claimed
full pretrain or equivalence to original 135M pretrained FP32.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v176.operators import TrainWeight,TrainLinear,TrainEmbedding,clear
from v185.codec import replace_discrete
from v185.optim import propose_codes,apply_codes
from v186.fit import paired_loss
from v186.pairs import build_pairs
from v200.rollout import probe_greedy_train,summarize_sequences
from quality import DATASET,CONFIG
from calibri2.s0.bootstrap import build_config,CANDIDATES

SEED=2021
STEPS=64
MAX_STEPS=128
LR=0.008
TRAIN_CONTEXT=48
PROBE_PAIR_IDS=(0,1,6,7,12,13,18,19)
TRAIN_GREEDY_TOKENS=8
BPP_LIMIT=0.700

def log(**items):
    print(json.dumps(items,ensure_ascii=False,default=str),flush=True)

def create_small_paid_model(archive,config):
    student=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32).cpu().eval()
    if set(student.state_dict())!=set(archive.meta["full_original_state_keys"]):
        raise AssertionError("Changing S0 model architecture/weights without explicit growth phase")
    student.requires_grad_(False)
    names={k:TrainWeight(archive.handles[k]) for k,d in
           archive.meta["tensor_descriptors"].items() if d["dtype"]=="CB59_K32V8"}
    aliases=archive.meta["source_alias_keys"]
    # Train only the centroids stored in this immutable physical archive.
    for s in names.values():
        s.log_scales.requires_grad_(False)
        s.centroids.requires_grad_(True)
    for path in archive.handles:
        key=aliases.get(path,path)
        src=names[key]
        module_path=path.removesuffix(".weight")
        parent,_,leaf=module_path.rpartition(".")
        owner=student.get_submodule(parent) if parent else student
        old=getattr(owner,leaf)
        if isinstance(old,nn.Embedding):
            repl=TrainEmbedding(src)
        elif isinstance(old,nn.Linear) and old.bias is None:
            repl=TrainLinear(src)
        else:
            raise AssertionError("Unsupported original type "+module_path)
        setattr(owner,leaf,repl)
    for path,w in archive.raw.items():
        parent,_,leaf=path.rpartition(".")
        target=getattr(student.get_submodule(parent),leaf)
        with torch.no_grad():
            target.copy_(torch.tensor(w.copy(),dtype=torch.float32))
    if student.model.embed_tokens.source is not student.lm_head.source:
        raise AssertionError("Shared tied embedding/head lost")
    trainable={id(s.centroids) for s in names.values()}
    real={id(p) for p in student.parameters() if p.requires_grad}
    if trainable!=real or not 1+7*config.num_hidden_layers==len(names):
        raise AssertionError("Unexpected unpaid weight training or missing unique operators")
    return student,names

def freeze_train(tok):
    revision=HfApi().dataset_info(DATASET).sha
    # This public training text is explicitly NOT validation/test.
    dataset=load_dataset(DATASET,CONFIG,split="train[:1000]",
                         revision=revision,trust_remote_code=False)
    plain="\n\n".join(x["text"] for x in dataset if x["text"].strip())
    ids=tok.encode(plain,add_special_tokens=False)
    if len(ids)<TRAIN_CONTEXT*50:
        raise AssertionError("Not enough actual public TRAIN text")
    train_tokens=torch.tensor(ids[:TRAIN_CONTEXT*64],dtype=torch.long)
    pairs=build_pairs(tok,"TRAIN")
    if len(pairs)!=24 or len({p["pair_id"] for p in pairs})!=24:
        raise AssertionError("Cannot trust frozen original TRAIN synthetic paired questions")
    hashes=[hashlib.sha256(q["question"].encode()).hexdigest()
            for p in pairs for q in (p["left"],p["right"])]
    return train_tokens,pairs,{"wiki_dataset":DATASET,
        "wiki_config":CONFIG,"split":"train[:1000]",
        "wiki_revision":revision,"actual_tokens":len(train_tokens),
        "TRAIN_token_ids_sha256":hashlib.sha256(
            train_tokens.numpy().tobytes()).hexdigest(),
        "synthetic_training_pairs":len(pairs),"synthetic_question_SHA256":hashes,
        "no_QA_DEV_VALID_TEST_during_training":True,
        "no_original_FP32_teacher_weights_or_teacher_targets":True}

def freeze_training_probe(model,sources,pairs):
    # Exact FULL answer + EOS, no metric based on public QA!
    selected=[pairs[i] for i in PROBE_PAIR_IDS]
    full=probe_greedy_train(
        model,sources,selected,max_new_tokens=TRAIN_GREEDY_TOKENS)
    summary=summarize_sequences(full)
    direct=[]
    for pair in selected:
        for side in ("left","right"):
            row=full[pair["pair_id"]]["sides"][side]
            direct.append({"pair_id":pair["pair_id"],"domain":pair["domain"],
                "side":side,"gold":row["gold_tokens_with_EOS"],
                "generated":row["generated_tokens"],
                "strict_full_correct":row["exact_with_EOS"],
                "first_token_matches_gold":bool(
                  row["generated_tokens"] and row["gold_tokens_with_EOS"] and
                  row["generated_tokens"][0]==row["gold_tokens_with_EOS"][0])})
    return {
        "fully_correct_answers":len(summary["exact"]),
        "correct_gold_prefix_tokens":summary["prefix_tokens"],
        "adjacent_repeats_excess":summary["repetition_excess"],
        "first_gold_token_correct":sum(x["first_token_matches_gold"] for x in direct),
        "generated_first_token_diversity":len(set(
              x["generated"][0] for x in direct if x["generated"])),
        "all_16_training_answers":direct},full

def main(src,out,steps):
    if not 1<=steps<=MAX_STEPS:raise AssertionError("Unsafe number of unverified S0 pilot steps")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    src=Path(src)
    model_dir=src.parent
    plan=json.loads((model_dir/"ARCHITECTURE_SELECTED.json").read_text())
    raw=src.read_bytes()
    cb=CB59(raw)
    if cb.hash()!=plan["physical_sha256"]:
        raise AssertionError("Original S0 paid checkpoint mismatches bootstrap")
    if 8*len(raw)/plan["unique_original_architecture_parameters"]>BPP_LIMIT:
        raise AssertionError("Original random source exceeds 0.7 bpp")
    spec=next((x for x in CANDIDATES if x[0]==plan["candidate"]),None)
    if spec is None:raise AssertionError("Non-precommitted S0 architecture")
    cfg=build_config(spec)
    if cfg.hidden_size!=plan["config"]["hidden_size"]:
        raise AssertionError("S0 architecture drift")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    train_tokens,train_pairs,train_provenance=freeze_train(tok)
    source_contract={
        "reference_random_source_sha256":plan["physical_sha256"],
        "complete_bytes":len(raw),"physical_initial_bpp":8*len(raw)/plan["unique_original_architecture_parameters"],
        "architecture":plan["config"],"trained_params_random_only":True,
        "training_data":train_provenance,"total_steps_precommitted":steps,
        "max_new_tokens_TRAIN_probe":TRAIN_GREEDY_TOKENS,
        "optimizer":"Adam_32x8_paid_FP16_dictionary_STE",
        "learning_rate":LR,"all_original_operators_trainable":True,
        "model_growth_NOT_performed_in_S0":True,
        "neither_original_FP32_nor_v228_neural_weights_loaded":True,
        "no_QA_benchmark_for_training_or_weight_selection":True}
    save(out/"PRECOMMITTED_S0_TRAIN_SOURCE_CONTRACT.json",source_contract)
    student,sources=create_small_paid_model(cb,cfg)
    source_keys=sorted(sources)
    before,raw_before=freeze_training_probe(student,sources,train_pairs)
    save(out/"CALIBRI2_S0_TRAIN_answer_probe_before.json",before)
    optimizer=torch.optim.Adam(
        [s.centroids for s in sources.values()],lr=LR)
    log(stage="CALIBRI2_S0_TRAIN_READY",
        source=plan["candidate"],paid_ops=len(sources),
        real_params=plan["unique_original_architecture_parameters"],
        physical_bpp=source_contract["physical_initial_bpp"],
        original_full_training_answers=before["fully_correct_answers"])
    attempts=[]
    for step in range(steps):
        # 3:1 supervised balanced explicit paired messages to raw WikiText,
        # corresponding to a preliminary instruction-tuning curriculum.
        if step%4==0:
            kind="WikiText_public_TRAIN"
            index=(step//4)%(len(train_tokens)//TRAIN_CONTEXT)
            ids=train_tokens[index*TRAIN_CONTEXT:(index+1)*TRAIN_CONTEXT]
            pair=None
        else:
            kind="opposing_synthetic_TRAIN"
            pair=train_pairs[(step-1-step//4)%len(train_pairs)]
            ids=None
        target_name=source_keys[step%len(source_keys)]
        target=sources[target_name]
        seen=[]
        def capture(_module,_input,reconstruction):
            if not any(reconstruction is previous for previous in seen):
                reconstruction.retain_grad()
                seen.append(reconstruction)
        handle=target.register_forward_hook(capture)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            if pair is None:
                logits=student(input_ids=ids[None],use_cache=False).logits
                objective=F.cross_entropy(logits[0,:-1],ids[1:])
                details={"wiki_token_window":index}
            else:
                objective,pinfo=paired_loss(student,target,pair)
                details={"pair_id":pair["pair_id"],"domain":pair["domain"],
                         "expected_first_tokens":pinfo["expected_first_tokens"]}
            if not torch.isfinite(objective):
                raise AssertionError("Nonfinite paid weight gradient objective")
            before_loss=float(objective.detach())
            objective.backward()
        finally:
            handle.remove()
        if len(seen)!=1 or seen[0].grad is None:
            raise AssertionError("Missing full model projected gradient for "+target_name)
        gradient=seen[0].grad.detach().cpu().numpy().copy()
        active=sum(s.centroids.grad is not None for s in sources.values())
        if active!=len(sources):
            raise AssertionError(f"Missing gradients {active}/{len(sources)}")
        grad=float(torch.nn.utils.clip_grad_norm_(
            [x.centroids for x in sources.values()],1.0,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():s.centroids.clamp_(-16.,16.)
        clear(sources)
        # Every step tests 5-bit code proposals from OWN current scratch
        # compact weights only, NOT any pretrained FP32 teacher weights.
        with torch.inference_mode():
            if pair is None:
                after_logits=student(input_ids=ids[None],use_cache=False).logits
                post_ce=F.cross_entropy(after_logits[0,:-1],ids[1:])
            else:
                post_ce,_=paired_loss(student,target,pair)
            loss_after_centroids=float(post_ce)
            prior=target().detach().cpu().numpy().copy()
        candidate=propose_codes(target,gradient,prior)
        n=candidate["proposed_groups"]
        if n>512:raise AssertionError("Excessive code proposals")
        passed=False
        candidate_loss=loss_after_centroids
        if n:
            with torch.no_grad():apply_codes(target,candidate,True)
            clear(sources)
            with torch.inference_mode():
                if pair is None:
                    changed_logits=student(input_ids=ids[None],use_cache=False).logits
                    changed_loss=F.cross_entropy(changed_logits[0,:-1],ids[1:])
                else:
                    changed_loss,_=paired_loss(student,target,pair)
                candidate_loss=float(changed_loss)
            passed=np.isfinite(candidate_loss) and candidate_loss+1e-7<loss_after_centroids
            if not passed:
                with torch.no_grad():apply_codes(target,candidate,False)
        clear(sources)
        row={"step":step+1,"kind":kind,"origin":details,
             "target_original_paid_operator":target_name,
             "source_paid_all_layer_backprop_grad_count":active,
             "source_paid_grad_norm":grad,
             "TRAIN_loss_before":before_loss,
             "TRAIN_loss_after_centroids":loss_after_centroids,
             "TRAIN_loss_after_code_candidate":candidate_loss,
             "candidate_5bit_reassignments":n,
             "accepted_5bit_reassignments":n if passed else 0,
             "codes_adopted_only_if_same_TRAIN_loss_improves":passed}
        attempts.append(row)
        if (step+1)%4==0:
            log(stage="CALIBRI2_S0_TRAIN_STEP",**row)
        if (step+1)%16==0:
            # Actual source snapshot at fixed milestones; QA NEVER used.
            current=out/f"CALIBRI2_S0_step{step+1:03d}.cb59"
            compressed,info=replace_discrete(cb,sources,strict=False)
            current.write_bytes(compressed)
            if len(compressed)!=len(raw) or 8*len(compressed)/plan["unique_original_architecture_parameters"]>BPP_LIMIT:
                raise AssertionError("S0 learned physical model broke its strict full rate cap")
            # Do not use these TRAIN outcomes to change steps / model size.
            actual_answers,_=freeze_training_probe(student,sources,train_pairs)
            log(stage="CALIBRI2_S0_TRAIN_MILESTONE",
                steps=step+1,full_TRAIN_correct=actual_answers["fully_correct_answers"],
                first_token_correct=actual_answers["first_gold_token_correct"],
                repeats=actual_answers["adjacent_repeats_excess"],
                unique_first_tokens=actual_answers["generated_first_token_diversity"],
                total_changed_fp16_dictionary_values=info["actually_changed_FP16_codebook_cells"],
                physical_bpp=info["complete_bpp"])
            save(out/f"CALIBRI2_S0_TRAIN_probe_step{step+1:03d}.json",actual_answers)
    trained_file=out/"CALIBRI2_S0_AFTER_TRAIN_PAID.cb59"
    trained,change=replace_discrete(cb,sources,strict=False)
    trained_file.write_bytes(trained)
    after,raw_after=freeze_training_probe(student,sources,train_pairs)
    if not ((len(trained)==len(raw)) and
        hashlib.sha256(trained).hexdigest()==change["sha256"] and
        change["complete_bpp"]<=BPP_LIMIT):
        raise AssertionError("S0 compact physical source not reproducibly saved")
    report={"source":source_contract,
        "training_step_count":len(attempts),
        "TRAIN_only_real_answers_before":before,
        "TRAIN_only_real_answers_after":after,
        "all_paid_centroids_grads_every_training_step":all(
            x["source_paid_all_layer_backprop_grad_count"]==len(sources)
            for x in attempts),
        "training_progress":attempts,
        "physical_change":change,
        "actual_original_random_source_sha256":cb.hash(),
        "actual_trained_source_sha256":change["sha256"],
        "no_teacher_or_pretrained_original_weights_ever_used":True,
        "not_a_fully_trained_language_model_and_not_FP32_parity":True}
    save(out/"ACTUAL_CALIBRI2_S0_PAID_TRAIN_AND_ANSWER_PROBE.json",report)
    log(stage="CALIBRI2_S0_ACTUAL_END",
        steps=len(attempts),new_model_sha256=change["sha256"],
        physically_paid_bpp=change["complete_bpp"],
        actual_FP16_centroids_changed=change["actually_changed_FP16_codebook_cells"],
        actually_accepted_5bit_indices=change["actual_changed_5bit_group_indices"],
        correct_full_TRAIN_answers_before=before["fully_correct_answers"],
        correct_full_TRAIN_answers_after=after["fully_correct_answers"],
        first_gold_tokens_before=before["first_gold_token_correct"],
        first_gold_tokens_after=after["first_gold_token_correct"],
        output_first_token_diversity_after=after["generated_first_token_diversity"],
        no_test_leakage=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--steps",type=int,default=STEPS)
    args=parser.parse_args()
    main(args.source,args.out,args.steps)
