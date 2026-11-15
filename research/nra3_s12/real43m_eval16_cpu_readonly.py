"""NRA2 3.0 S12b: original 43M Calibri read-only CPU post-GT layer8 16 prereg EVAL.

Single existing S194 456+128+32 native CPU protected J7 run, with one
additional READ-ONLY layer8 hook during 16 additional SHA-pinned Russian EVAL
prompt prefills. The read hook is installed AFTER real S93 GT and all-layer
bridge scopes. NO new model writes, no Russian gold in model input, and
NO scored prompts, no reserved prior independent EVAL. Original 43M + GT weights frozen.

S12b is read-only original CPU EVAL16 hidden collection: NO scoring, optimization, model selection or deployment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file,save_file

from calibri2.s194.actual_cpu_j7_bridge import (
    actual_fp16_checkpoint_sha,output_digest,real_j7_baseline,
)
from research.nra3_s12.heldout_ru_eval16 import (
    EVAL16_FULL_CANONICAL_SHA,build_eval16,sealed_eval16,
)
from research.nra3_s12.locked_real_eval16_token_alignment import (
    align_heldout16,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort


def sha_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    argv=list(sys.argv)
    if len(argv)<27:
        raise SystemExit("S12b requires original verified S194 43M CPU J7 CLI")
    out=Path(argv[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S12b never overwrites EVAL16 hidden evidence")
    if torch.cuda.is_available() or sys.platform!="linux":
        raise RuntimeError("S12b requires original REAL Linux CPU only")
    # ONLY separately preregistered SHA-pinned 16 EVAL row texts.
    rows=build_eval16()
    evidence=sealed_eval16()
    if (len(rows)!=16 or evidence["rows"]!=16
            or evidence["complete_annotation_sha256"]!=EVAL16_FULL_CANONICAL_SHA
            or evidence["evaluation_started"] is not False
            or evidence["optimizer_steps_on_eval"]!=0
            or evidence["all_previous_five_consumed_EVAL128_untouched"] is not True
            or evidence["original_sixth_S3CI_EVAL128_untouched"] is not True
            or evidence["prior_S11_router_fp16_physical_sha256"]!=
               "05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"):
        raise AssertionError("S12b immutable new EVAL16 prereg or frozen S11 source drift")
    previous=json.loads(Path(
        "inputs/nra3_s12/NRA3_S12_HELDOUT_EVAL16_SHA_PREREG_ONLY.json").read_text(encoding="utf-8"))
    if (previous!=evidence or previous["complete_annotation_sha256"]!=
            "f2d2d5beba48dbe294a90f6918a3fd15870aa103abbd569e907c44729a43977c"):
        raise AssertionError("S12b exact SHA-pinned prereg before EVAL forward missing")

    s194=json.loads(Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s194.get("all_616_exact_real_CPU_generated_tokens_captured") is not True
            or s194.get("physical_FP16_reload_verified") is not True
            or s194.get("actual_43M_training_steps")!=0
            or s194.get("model_accepted") is not False):
        raise AssertionError("S12b original real S194 616 J7 source missing")

    from calibri2.s112 import prompt_conditioned_first_token_mps as s112
    from calibri2.s160 import real_parallel_slots_j7 as s160
    from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
    from v176.operators import clear

    collector=FrozenLayer8ReadPort()
    original_load=s112.load_cpu_original
    original_probe=s190.probe_true_frozen_cpu
    actual_model=[]
    capture_count=0
    real_tokenizer=[]

    def load_original_with_read_only_port(*args,**kwargs):
        cpu=original_load(*args,**kwargs)
        if actual_model:
            raise AssertionError("S12b reloaded original 43M CPU twice")
        model,sources,*_=cpu
        if model.training or len(model.model.layers)!=12:
            raise AssertionError("S12b must read original frozen 12-layer 43M")
        actual_model.append(id(model))
        # Do NOT install S12b EVAL hook before original protected CPU J7.
        return cpu

    def extra_eval16_after_original_J7(
            cpu,old,old_ids,routes,tok,bank,copy_net,bridge,full,
            expansion,parent_old,head):
        nonlocal capture_count
        # Complete original 32 new TRAIN CPU decode before any Russian
        # capture, *after* original 456+128 protected J7. Keep source
        # modules and full 616 original output captures untouched.
        original_probe(cpu,old,old_ids,routes,tok,bank,copy_net,bridge,
                       full,expansion,parent_old,head)
        if capture_count or len(actual_model)!=1 or id(cpu[0])!=actual_model[0]:
            raise AssertionError("S12b duplicated original CPU model")
        real_tokenizer.append(tok)
        model,sources,*_=cpu
        s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
        for row in rows:
            q=row.text # raw prereg EVAL prompt only; NO UD/semantic gold to Calibri
            ids=list(s160.prompt_tokens(tok,q))
            aligned=align_heldout16(tok,row,expected_input_ids=ids)
            if aligned.original_prompt_sha256!=row.sha256() or not aligned.independent_eval_labels_loaded:
                raise AssertionError("S12b original Calibri EVAL tokenizer/source SHA drift")
            prompt_vec=torch.as_tensor(s160.prompt_only_vector(tok,q),dtype=torch.float32)
            with (torch.inference_mode(),expansion.prefill_only(model),
                  s160.prefill_only_tunnel(model,full)):
                # Critical: register actual readport INSIDE actual S93
                # scope so S103 / GT residual hooks run before NRA read.
                with s160.frozen_cpu_phase_scope(
                        model,bridge,"S93",prompt_vec,phase="prefill_only"):
                    with collector.capture(model,q,ids):
                        clear(sources)
                        result=model(input_ids=torch.tensor([ids],dtype=torch.long),
                                     past_key_values=None,use_cache=True,
                                     output_hidden_states=True)
                        if (result.hidden_states[-1].shape[-1]!=320
                                or result.past_key_values is None):
                            raise AssertionError("S12b original 43M CPU eval prefill drift")
                        del result
                        clear(sources)
            capture_count+=1

    s112.load_cpu_original=load_original_with_read_only_port
    s190.probe_true_frozen_cpu=extra_eval16_after_original_J7
    try:
        old,new,expected128,old_gold,novel,novel_gold,eos,s190_source=(
            real_j7_baseline(argv,old_s87=Path(argv[-2])))
    finally:
        s112.load_cpu_original=original_load
        s190.probe_true_frozen_cpu=original_probe
    # Posthoc ONLY: verify exact protected 616 original tokens from known S194.
    joined={**old,**new}
    source_cert=s194["J7Retention584_frozen_baseline"]
    fp16_src=actual_fp16_checkpoint_sha([Path(argv[-7]),Path(argv[-5])])
    if (len(old)!=456 or len(new)!=128 or len(novel)!=32 or len(joined)!=584
            or capture_count!=16 or collector.observed_calls!=16
            or len(actual_model)!=1 or len(real_tokenizer)!=1
            or sum(old[k]==old_gold[k] for k in old)!=456
            or sum(new[k]==expected128[k] for k in new)!=128
            or sum(novel[k]==novel_gold[k] for k in novel)!=0
            or s190_source["NEW_TRAIN_first_correct"]!=16
            or source_cert["old_full_eos_correct"]!=456
            or source_cert["new_full_eos_correct"]!=128
            or output_digest(joined)!=source_cert["protected_584_generation_digest"]
            or output_digest(novel)!=source_cert["train32_generation_digest"]
            or fp16_src!=s194["checkpoint_fingerprint_sha256"]):
        raise AssertionError("S12b original 43M/GT CPU J7 616 checkpoint drift")
    token_batch=collector.frozen_batch(tuple(r.text for r in rows))
    tok=real_tokenizer[0]
    alignments=[align_heldout16(
        tok,row,expected_input_ids=token_batch["input_ids"][i,
            :int(token_batch["token_lengths"][i])].tolist())
                for i,row in enumerate(rows)]
    for i,aligned in enumerate(alignments):
        if (aligned.tokenizer_token_count!=int(token_batch["token_lengths"][i])
                or aligned.complete_prefix_position!=
                   aligned.tokenizer_token_count-1):
            raise AssertionError("S12b actual EVAL tokenizer alignment drift")
    # Only after original J7 and native CPU prefills: store ACTUAL hidden
    # features, lengths and subtoken offsets; NEVER export gold labels.
    out.mkdir(parents=True,exist_ok=True)
    feature_file=out/"NRA3_S12B_REAL_43M_LAYER8_RU_EVAL16_HIDDEN320_FP16.safetensors"
    save_file(token_batch,str(feature_file))
    saved=load_file(str(feature_file),device="cpu")
    if (set(saved)!=set(token_batch)
            or any(not torch.equal(saved[k],v) for k,v in token_batch.items())):
        raise AssertionError("S12b exact FP16 read-only serialization/reload failed")
    # Physical word/subtoken maps contain NO hidden features or answers.
    maps=[{
      "text_sha256":row.sha256(),
      "token_count":aligned.tokenizer_token_count,
      "word_last_token_positions":[w.last_token for w in aligned.word_alignment],
      "word_token_positions":[list(w.token_positions)
                              for w in aligned.word_alignment],
    } for row,aligned in zip(rows,alignments)]
    report={
      "stage":"NRA3_S12B_TRUE_FROZEN_CALIBRI_43M_POST_GT_LAYER8_NEW_RUSSIAN_EVAL16_READONLY",
      "S12_RUSSIAN_UD_EVAL16_CANONICAL_SHA256":EVAL16_FULL_CANONICAL_SHA,
      "S12_PREREG_PHYSICAL_RUN_ID":43735385201,
      "S12_EVAL16_PREREG_BEFORE_MODEL":True,
      "S12_S11_FROZEN_BEFORE_EVAL":True,
      "S194_ORIGINAL_616_J7_BASELINE_PARENT_RUN_ID":43345267876,
      "S194_ORIGINAL_CHECKPOINT_SHA256":fp16_src,
      "S194_ORIGINAL_584_GENERATED_TOKEN_DIGEST":
        source_cert["protected_584_generation_digest"],
      "S194_ORIGINAL_32_NEW_GENERATED_TOKEN_DIGEST":
        source_cert["train32_generation_digest"],
      "all_616_native_CPU_generated_token_digests_unchanged":True,
      "original_456_full_EOS":456,
      "original_new128_full_EOS":128,
      "original_new32_full_EOS":0,
      "actual_43M_CPU_12_transformer_layers":True,
      "observed_layer_index_zero_based":7,
      "GT_residuals_before_readport_in_same_active_S93_scope":True,
      "NRA3_output_gate":0.0,
      "NRA3_candidate_accepted":False,
      "real_source_eval16_count":len(rows),
      "actual_genuine_43M_CPU_prefills":capture_count,
      "real_tokenizer_chat_offsets_verified":True,
      "true_real_Calibri_token_alignment_all_16":True,
      "S12_eval_labels_loaded_into_Calibri_forward":False,
      "S12_eval16_scored":False,
      "S12_original_43M_EVAL16_only_after_SHA_pinning":True,
      "no_training_or_optimizer_for_original_43M_and_GT":True,
      "S12_all_16_prompts_are_new_EVAL_only":True,
      "all_gold_labels_excluded_from_frozen_decoder":True,
      "actual_fp16_feature_file":feature_file.name,
      "actual_fp16_feature_sha256":sha_file(feature_file),
      "actual_hidden320_shape":list(token_batch["token_hidden320"].shape),
      "actual_token_ids_lengths_sha256":hashlib.sha256(json.dumps(
           [list(s.token_ids) for s in [
              collector.samples[row.sha256()] for row in rows]],
           sort_keys=True,separators=(",",":")).encode()).hexdigest(),
      "true_eval16_token_counts":token_batch["token_lengths"].tolist(),
      "wordpiece_alignment":maps,
      "actual_NRA3_6M_weights_trained":False,
      "actual_NRA3_grammar_semantic_accuracy_tested":False,
      "native_NRA3_live_generation_tested":False,
      "native_CPU_speed_relative_CB59_measured":False,
      "physical_all_inclusive_GT_NRA3_bpp_certified":False,
      "five_consumed_independent_EVAL128_not_loaded":True,
      "sixth_reserved_S3CI_EVAL128_not_loaded":True,
      "no_Render_Supabase_APK_LIVE_or_Jessica_chat":True,
    }
    (out/"NRA3_S12B_REAL_CPU_43M_LAYER8_EVAL16_J7_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("wordpiece_alignment","true_eval16_token_counts")},
                     sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
