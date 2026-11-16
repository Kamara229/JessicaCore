"""NRA3 S18b: original frozen Calibri43M+GT actual native CPU capture of NEW96 TRAIN.

Research-only; run one complete baseline S194 original CPU protected 616 J7,
then 96 genuine original Calibri43M post-GT layer8 read-only CPU prefills.
The original S18 TRAIN annotation SHA, UD and role labels are ONLY checked
against immutable source; only row.text is presented to original Calibri.
No S17c/S15c/S12c or any independent EVAL sources, no optimization,
no trained S16b reader, no model selection, no LIVE or main changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

from calibri2.s194.actual_cpu_j7_bridge import (
    actual_fp16_checkpoint_sha, output_digest, real_j7_baseline,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
    build_s18_new_train96, seal_s18_new96_train,
)
from research.nra3_s18.locked_original43m_new96_token_alignment import (
    align_s18_new96_train,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    argv = list(sys.argv)
    if len(argv) < 27:
        raise SystemExit("S18b requires full genuine S194 43M CPU 616 J7 CLI")
    out = Path(argv[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S18b never overwrites new96 TRAIN CPU features")
    if sys.platform != "linux" or torch.cuda.is_available():
        raise RuntimeError("S18b original43M capture must run on Linux CPU only")
    rows = build_s18_new_train96()
    evidence = seal_s18_new96_train()
    if (
        len(rows) != 96
        or evidence["S18_original_source_rows"] != 96
        or evidence["S18_new_directed_verb_to_argument_links"] != 300
        or evidence["S18_complete_original_96_new_TRAIN_full_annotation_SHA256"]
            != S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256
        or S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256
            != "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9"
        or evidence["S18_original_43M_GT_NRA_C_or_2309FP16_model_loaded"] is not False
        or evidence["S18_new_optimizer_steps"] != 0
        or evidence["S18_old_S17c_24_independent_EVAL_now_consumed_not_read"] is not True
    ):
        raise AssertionError("S18b new96 complete gold SHA original TRAIN prereg drift")
    previous = json.loads(Path(
        "inputs/nra3_s18/NRA3_S18_NEW96_MULTIVERB_GRAPH_TRAIN_ONLY_SOURCE_MANIFEST.json"
    ).read_text(encoding="utf-8"))
    if previous != evidence:
        raise AssertionError("S18b authenticated prior source-only TRAIN96 manifest mismatch")
    # Do not load original S18 TRAIN labels as model inputs or import any old EVAL.

    s194=json.loads(Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s194.get("all_616_exact_real_CPU_generated_tokens_captured") is not True
            or s194.get("physical_FP16_reload_verified") is not True
            or s194.get("actual_43M_training_steps")!=0
            or s194.get("model_accepted") is not False):
        raise AssertionError("S18b original real S194 616 J7 source missing")

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
            raise AssertionError("S18b reloaded original 43M CPU twice")
        model,sources,*_=cpu
        if model.training or len(model.model.layers)!=12:
            raise AssertionError("S18b must read original frozen 12-layer 43M")
        actual_model.append(id(model))
        # Do NOT install S18b TRAIN hook before original protected CPU J7.
        return cpu

    def extra_train96_after_original_J7(
            cpu,old,old_ids,routes,tok,bank,copy_net,bridge,full,
            expansion,parent_old,head):
        nonlocal capture_count
        # Complete original 32 new TRAIN CPU decode before any Russian
        # capture, *after* original 456+128 protected J7. Keep source
        # modules and full 616 original output captures untouched.
        original_probe(cpu,old,old_ids,routes,tok,bank,copy_net,bridge,
                       full,expansion,parent_old,head)
        if capture_count or len(actual_model)!=1 or id(cpu[0])!=actual_model[0]:
            raise AssertionError("S18b duplicated original CPU model")
        real_tokenizer.append(tok)
        model,sources,*_=cpu
        s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
        for row in rows:
            q=row.text # raw prereg TRAIN prompt only; NO UD/semantic gold to Calibri
            ids=list(s160.prompt_tokens(tok,q))
            aligned=align_s18_new96_train(tok,row,expected_input_ids=ids)
            if aligned.original_prompt_sha256!=row.text_sha256() or aligned.independent_eval_labels_loaded:
                raise AssertionError("S18b original Calibri TRAIN tokenizer/source SHA drift")
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
                            raise AssertionError("S18b original 43M CPU train prefill drift")
                        del result
                        clear(sources)
            capture_count+=1

    s112.load_cpu_original=load_original_with_read_only_port
    s190.probe_true_frozen_cpu=extra_train96_after_original_J7
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
            or capture_count!=96 or collector.observed_calls!=96
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
        raise AssertionError("S18b original 43M/GT CPU J7 616 checkpoint drift")
    token_batch=collector.frozen_batch(tuple(r.text for r in rows))
    tok=real_tokenizer[0]
    alignments=[align_s18_new96_train(
        tok,row,expected_input_ids=token_batch["input_ids"][i,
            :int(token_batch["token_lengths"][i])].tolist())
                for i,row in enumerate(rows)]
    for i,aligned in enumerate(alignments):
        if (aligned.tokenizer_token_count!=int(token_batch["token_lengths"][i])
                or aligned.complete_prefix_position!=
                   aligned.tokenizer_token_count-1):
            raise AssertionError("S18b actual TRAIN tokenizer alignment drift")
    # Only after original J7 and native CPU prefills: store ACTUAL hidden
    # features, lengths and subtoken offsets; NEVER export gold labels.
    out.mkdir(parents=True,exist_ok=True)
    feature_file=out/"NRA3_S18B_TRUE43M_GT_NEW_RU96_DIRECTED_HIDDEN320_FP16.safetensors"
    save_file(token_batch,str(feature_file))
    saved=load_file(str(feature_file),device="cpu")
    if (set(saved)!=set(token_batch)
            or any(not torch.equal(saved[k],v) for k,v in token_batch.items())):
        raise AssertionError("S18b exact FP16 read-only serialization/reload failed")
    # Physical word/subtoken maps contain NO hidden features or answers.
    maps=[{
      "text_sha256":row.text_sha256(),
      "token_count":aligned.tokenizer_token_count,
      "word_last_token_positions":[w.last_token for w in aligned.word_alignment],
      "word_token_positions":[list(w.token_positions)
                              for w in aligned.word_alignment],
    } for row,aligned in zip(rows,alignments)]
    report = {
        "stage": "NRA3_S18B_TRUE_ORIGINAL43M_GT_LAYER8_CPU_NEW96_TRAIN_ONLY",
        "S18_NEW96_TRAIN_COMPLETE_SOURCE_SHA256": S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        "S18_SOURCE_ONLY_CI_RUN_ID": 43904457245,
        "S18b0_original_tokenizer_contract_CI_RUN_ID": 43908545721,
        "S18_new96_train_gold_links_only_in_source": 300,
        "S18_research_only_TRAIN_not_independent_EVAL": True,
        "S18_no_old_consumed_S17c_S15c_S12c_or_EVAL128_sources": True,
        "S18_2309FP16_S16b_adapter_not_loaded": True,
        "S194_ORIGINAL_616_J7_BASELINE_PARENT_RUN_ID": 43345267876,
        "S194_ORIGINAL_CHECKPOINT_SHA256": fp16_src,
        "S194_ORIGINAL_584_GENERATED_TOKEN_DIGEST": source_cert["protected_584_generation_digest"],
        "S194_ORIGINAL_32_NEW_GENERATED_TOKEN_DIGEST": source_cert["train32_generation_digest"],
        "all_616_native_CPU_generated_token_digests_unchanged": True,
        "original_456_full_EOS": 456,
        "original_new128_full_EOS": 128,
        "original_new32_full_EOS": 0,
        "actual_43M_CPU_12_transformer_layers": True,
        "observed_layer_index_zero_based": 7,
        "GT_residuals_before_readport_in_same_active_S93_scope": True,
        "NRA3_output_gate": 0,
        "NRA3_candidate_accepted": False,
        "activation_allowed": False,
        "new96_TRAIN_rows": len(rows),
        "actual_genuine_43M_CPU_new96_TRAIN_prefills": capture_count,
        "real_tokenizer_chat_offsets_verified": True,
        "true_real_Calibri_token_alignment_all_96": True,
        "S18_no_optimizer_or_model_selection": True,
        "S18_gold_labels_not_passed_to_original43M": True,
        "actual_fp16_feature_file": feature_file.name,
        "actual_fp16_feature_sha256": sha_file(feature_file),
        "actual_hidden320_shape": list(token_batch["token_hidden320"].shape),
        "actual_token_ids_lengths_sha256": hashlib.sha256(json.dumps(
            [list(collector.samples[row.text_sha256()].token_ids) for row in rows],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "true_train96_token_counts": token_batch["token_lengths"].tolist(),
        "wordpiece_alignment": maps,
        "no_training_original43M_GT_NRA_or_S16b": True,
        "NRA3_grammar_or_semantic_accuracy_tested": False,
        "new_S18_candidate_616_active_J7_EOS_measured": False,
        "full_all_in_GT_NRA_bpp_le07_certified": False,
        "whole_system_CPU_speed_ge90pct_CB59_measured": False,
        "no_Render_Supabase_APK_LIVE_or_Jessica_chat": True,
    }
    (out / "NRA3_S18B_TRUE43M_CPU_NEW96_TRAINONLY_J7_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n",
        encoding="utf-8"
    )
    print(json.dumps({
        k: v for k, v in report.items()
        if k not in ("wordpiece_alignment", "true_train96_token_counts")
    }, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
