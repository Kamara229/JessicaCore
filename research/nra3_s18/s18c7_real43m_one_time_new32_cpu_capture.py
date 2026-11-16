"""S18c7 ONE-TIME heldout new32 real original Calibri43M GT CPU physical.

Both frozen rank8 candidate FP16 blobs from S18c1 and S18c4 were
committed and trained BEFORE the new EVAL32 text or labels existed.
This stage loads exactly the S18c6 SHA-locked ONE-TIME 32 source,
verifies original S194 616 protected J7, then captures genuine
frozen original CPU 43M GT post-layer8 features on 32 raw texts.
Only row.text is passed into actual Calibri and its tokenizer.
NO labels, gold edges, predicate positions, UD or semantic roles
are passed to original model. No readout score, loss, optimizer or
checkpoint modification in this code. S18c7 later scores both
fully frozen candidates on same 32 one-time EVAL.
No prior consumed EVAL inputs or outputs are accessed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file,save_file

from calibri2.s194.actual_cpu_j7_bridge import (
    actual_fp16_checkpoint_sha, output_digest, real_j7_baseline,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s18.s18c6_new32_one_time_eval_prereg import (
    build_new_eval32, seal_new_eval32_source,
    EVAL32_COMPLETE_CANONICAL_SHA256, NEW_EVAL_SOURCE_CI,
)
from research.nra3_s18.s18c7_new32_original_token_alignment import (
    align_s18c7_raw_eval_text,
)

def sha_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    argv=list(sys.argv)
    if len(argv)<27:
        raise SystemExit("S18c7 requires true native S194 43M CPU 616-J7 CLI")
    out=Path(argv[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S18c7 do not overwrite any original eval capture")
    if sys.platform!="linux" or torch.cuda.is_available():
        raise RuntimeError("S18c7 frozen original43M+GT capture only on Linux CPU")
    rows=build_new_eval32()
    sealed=seal_new_eval32_source()
    if (len(rows)!=32 or NEW_EVAL_SOURCE_CI!=43919440242
            or EVAL32_COMPLETE_CANONICAL_SHA256!=
              "16cb3964a0c4220afc82ff67756534a180623f0303c92aede6eff48e0416a5fe"
            or sealed["new_eval_source_complete_sha256"]!=
               EVAL32_COMPLETE_CANONICAL_SHA256
            or sealed["first_prereg_source_only_CI"]!=43919440242
            or sealed["new_eval_gold_predicate_to_argument_links"]!=104
            or sealed["sha_locked_in_source"] is not True
            or sealed["candidate_scored_on_eval32"] is not False
            or sealed["original43M_GT_inference_executed_on_eval32"] is not False
            or sealed["source_prior_candidate_2597_FP16_sha"]!=
               "ea47451b8d34a8842d75529af4332717599fbd9862dc0cad97c4d58c1803d761"
            or sealed["source_prior_control_2309_FP16_sha"]!=
               "ec598cc7b5cb98fc8945101caee11c8ca4e589abca4500325ec7ac6eddc831df"):
        raise AssertionError("S18c7 one-time NEW32 source or frozen paid models not preregistered")
    source_manifest=json.loads(Path(
       "inputs/s18c6_source/NRA3_S18C6_NEW32_ONE_TIME_EVAL_SOURCE_ONLY_SEAL.json"
    ).read_text(encoding="utf-8"))
    if source_manifest!=sealed:
        raise AssertionError("S18c7 EVAL32 source-only independent CI full SHA drift")
    # NEVER load old consumed EVAL or any S18c4/S18c1 candidate until
    # all 32 original frozen CPU features are safely captured.
    s194=json.loads(Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s194.get("all_616_exact_real_CPU_generated_tokens_captured") is not True
            or s194.get("physical_FP16_reload_verified") is not True
            or s194.get("actual_43M_training_steps")!=0
            or s194.get("model_accepted") is not False):
        raise AssertionError("S18c7 original real S194 616 J7 source missing")

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
            raise AssertionError("S18c7 reloaded original 43M CPU twice")
        model,sources,*_=cpu
        if model.training or len(model.model.layers)!=12:
            raise AssertionError("S18c7 must read original frozen 12-layer 43M")
        actual_model.append(id(model))
        # Do NOT install S18c7 EVAL hook before original protected CPU J7.
        return cpu

    def extra_eval32_after_original_J7(
            cpu,old,old_ids,routes,tok,bank,copy_net,bridge,full,
            expansion,parent_old,head):
        nonlocal capture_count
        # Complete original 32 new EVAL CPU decode before any Russian
        # capture, *after* original 456+128 protected J7. Keep source
        # modules and full 616 original output captures untouched.
        original_probe(cpu,old,old_ids,routes,tok,bank,copy_net,bridge,
                       full,expansion,parent_old,head)
        if capture_count or len(actual_model)!=1 or id(cpu[0])!=actual_model[0]:
            raise AssertionError("S18c7 duplicated original CPU model")
        real_tokenizer.append(tok)
        model,sources,*_=cpu
        s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
        for row in rows:
            q=row.text # raw prereg EVAL prompt only; NO UD/semantic gold to Calibri
            ids=list(s160.prompt_tokens(tok,q))
            aligned=align_s18c7_raw_eval_text(tok,row.text,expected_input_ids=ids)
            if aligned.original_prompt_sha256!=row.text_sha256() or aligned.independent_eval_labels_loaded:
                raise AssertionError("S18c7 original Calibri EVAL tokenizer/source SHA drift")
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
                            raise AssertionError("S18c7 original 43M CPU train prefill drift")
                        del result
                        clear(sources)
            capture_count+=1

    s112.load_cpu_original=load_original_with_read_only_port
    s190.probe_true_frozen_cpu=extra_eval32_after_original_J7
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
            or capture_count!=32 or collector.observed_calls!=32
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
        raise AssertionError("S18c7 original 43M/GT CPU J7 616 checkpoint drift")
    token_batch=collector.frozen_batch(tuple(r.text for r in rows))
    tok=real_tokenizer[0]
    alignments=[align_s18c7_raw_eval_text(
        tok,row.text,expected_input_ids=token_batch["input_ids"][i,
            :int(token_batch["token_lengths"][i])].tolist())
                for i,row in enumerate(rows)]
    for i,aligned in enumerate(alignments):
        if (aligned.tokenizer_token_count!=int(token_batch["token_lengths"][i])
                or aligned.complete_prefix_position!=
                   aligned.tokenizer_token_count-1):
            raise AssertionError("S18c7 actual EVAL tokenizer alignment drift")
    # Only after original J7 and native CPU prefills: store ACTUAL hidden
    # features, lengths and subtoken offsets; NEVER export gold labels.
    out.mkdir(parents=True,exist_ok=True)
    feature_file=out/"NRA3_S18C7_TRUE43M_GT_NEW_EVAL32_GOLDFREE_HIDDEN320_FP16.safetensors"
    save_file(token_batch,str(feature_file))
    saved=load_file(str(feature_file),device="cpu")
    if (set(saved)!=set(token_batch)
            or any(not torch.equal(saved[k],v) for k,v in token_batch.items())):
        raise AssertionError("S18c7 exact FP16 read-only serialization/reload failed")
    # Save original true tokenizer source maps (no gold labels).
    maps=[{
        "text_sha256":row.text_sha256(),
        "token_count":aligned.tokenizer_token_count,
        "word_last_token_positions":[z.last_token for z in aligned.word_alignment],
        "word_token_positions":[list(z.token_positions)
                               for z in aligned.word_alignment],
    } for row,aligned in zip(rows,alignments)]
    report={
        "stage":"NRA3_S18C7_TRUE_ORIGINAL43M_GT_LAYER8_CPU_NEW32_ONE_TIME_EVAL_NO_GOLD",
        "S18c6_prereg_new32_complete_source_SHA":EVAL32_COMPLETE_CANONICAL_SHA256,
        "S18c6_first_source_only_CI":NEW_EVAL_SOURCE_CI,
        "S18c6_new32_source_gold_links_never_to_original43M":104,
        "S18c7_old_consumed_EVALs_never_loaded":True,
        "S18c4_candidate_FP16_SHA_frozen_BEFORE_EVAL":"ea47451b8d34a8842d75529af4332717599fbd9862dc0cad97c4d58c1803d761",
        "S18c1_control_FP16_SHA_frozen_BEFORE_EVAL":"ec598cc7b5cb98fc8945101caee11c8ca4e589abca4500325ec7ac6eddc831df",
        "S194_original_frozen_616_J7_parent_run":43345267876,
        "S194_original_checkpoint_SHA":fp16_src,
        "S194_original_584_J7_token_digest":source_cert["protected_584_generation_digest"],
        "S194_original_32_new_train_token_digest":source_cert["train32_generation_digest"],
        "original_616_J7_cpu_generated_tokens_unchanged":True,
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "original_32_train_source_correct":0,
        "actual_43M_CPU_12_layers":True,
        "actual_GreenTunnel_precedes_layer8_readport":True,
        "original_43M_GT_no_optimizer":True,
        "original_43M_GT_new32_original_CPU_prefills":capture_count,
        "native_C_or_rank8_used_during_capture":False,
        "original_tokenizer_true_chat_offsets_all_32":True,
        "actual_fp16_feature_file":feature_file.name,
        "actual_fp16_feature_sha256":sha_file(feature_file),
        "actual_hidden320_shape":list(token_batch["token_hidden320"].shape),
        "actual_token_ids_lengths_sha256":hashlib.sha256(json.dumps(
           [list(collector.samples[row.text_sha256()].token_ids) for row in rows],
           sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "true_new32_eval_token_counts":token_batch["token_lengths"].tolist(),
        "wordpiece_alignment":maps,
        "new_candidate_active_J7_EOS_tested":False,
        "new_candidate_full_system_bpp_le07_proven":False,
        "new_candidate_CPU_speed_ge90pct_CB59_proven":False,
        "original_43M_GT_and_readonly_616_J7_unchanged":True,
        "new_model_activated":False,"output_gate":0,
        "candidate_accepted":False,"activation_allowed":False,
        "LIVE_Jessica_Render_Supabase_APK_modified":False,
    }
    (out/"NRA3_S18C7_TRUE43M_GT_EVAL32_CPU_J7_CAPTURE_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
            if k not in ("wordpiece_alignment","true_new32_eval_token_counts")},
           sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
