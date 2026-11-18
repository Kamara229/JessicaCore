"""S50: real original Calibri43M+GT CPU layer8 capture for 36 new TRAIN texts.

Replay protected original 456+128+32 CPU J7 before obtaining gold-blind
hidden vectors for SHA-locked synthetic TRAIN-only Russian prompts.
No source event labels, roles, predicate anchors, or coref enter model.
No original weights, Green Tunnel, decoder or LIVE updates or spent EVAL.
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
from research.nra3_s20.s50_quote36_original43m_native_alignment import (
    align_s50_raw_train36, s50_quote_train36_plaintexts,
)
from research.nra3_s20.s48_frozen_manual_train36_source import (
    digest as s48_train36_digest,
)
from research.nra3_s20.s49_real43m_gt_quote_sidecar_zero_gate import (
    Frozen43MGreenTunnelQuotePort, frozen_native_manifest_digest,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    argv = list(sys.argv)
    if len(argv) < 27:
        raise SystemExit("S50 requires full genuine original S194 CPU protected 616 J7 CLI")
    out = Path(argv[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S50 will not overwrite original S50 TRAIN36 CPU features")
    if sys.platform != "linux" or torch.cuda.is_available():
        raise RuntimeError("S50 genuine original43M capture only on Linux CPU")
    texts=s50_quote_train36_plaintexts()
    source_sha=s48_train36_digest()
    if len(texts)!=36 or len(source_sha)!=64 or not any('«' in q for q in texts):
        raise AssertionError('S50 requires sealed original S48 TRAIN36 quote plain texts')
    # Only raw text enters the original Calibri tokenizer and forward.
    # Human graph gold never enters any neural input or feature tensor.
    if any(q.strip()!=q or not q for q in texts):
        raise AssertionError("S50 raw TRAIN36 source invalid")

    s194=json.loads(Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s194.get("all_616_exact_real_CPU_generated_tokens_captured") is not True
            or s194.get("physical_FP16_reload_verified") is not True
            or s194.get("actual_43M_training_steps")!=0
            or s194.get("model_accepted") is not False):
        raise AssertionError("S50 original real S194 616 J7 source missing")

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
    align_failed=[]

    def load_original_with_read_only_port(*args,**kwargs):
        cpu=original_load(*args,**kwargs)
        if actual_model:
            raise AssertionError("S50 reloaded original 43M CPU twice")
        model,sources,*_=cpu
        if model.training or len(model.model.layers)!=12:
            raise AssertionError("S50 requires 12-layer original43M CPU")
        actual_model.append(id(model))
        # Do NOT install S50 TRAIN hook before original protected CPU J7.
        return cpu

    def extra_train36_after_original_J7(
            cpu,old,old_ids,routes,tok,bank,copy_net,bridge,full,
            expansion,parent_old,head):
        nonlocal capture_count
        # Complete original 32 new TRAIN CPU decode before any Russian
        # capture, *after* original 456+128 protected J7. Keep source
        # modules and full 616 original output captures untouched.
        original_probe(cpu,old,old_ids,routes,tok,bank,copy_net,bridge,
                       full,expansion,parent_old,head)
        if capture_count or len(actual_model)!=1 or id(cpu[0])!=actual_model[0]:
            raise AssertionError("S50 duplicated original CPU model")
        real_tokenizer.append(tok)
        model,sources,*_=cpu
        s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
        for q in texts:
            # Do not supply any graph, gold, predicate anchors or annotations.
            ids=list(s160.prompt_tokens(tok,q))
            try:
                aligned=align_s50_raw_train36(tok,q,expected_ids=ids)
                if aligned.original_prompt_sha256!=hashlib.sha256(q.encode('utf-8')).hexdigest() or aligned.independent_eval_labels_loaded:
                    raise ValueError('S50 original prompt SHA / BPE drift')
            except ValueError as ex:
                align_failed.append({'text_sha256':hashlib.sha256(q.encode('utf-8')).hexdigest(),
                                     'reason':str(ex)[:180]})
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
                            raise AssertionError("S50 original43M CPU raw36 prefill drift")
                        del result
                        clear(sources)
            capture_count+=1

    s112.load_cpu_original=load_original_with_read_only_port
    s190.probe_true_frozen_cpu=extra_train36_after_original_J7
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
            or capture_count!=36 or collector.observed_calls!=36
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
        raise AssertionError("S50 original 43M/GT CPU protected 616 J7 checkpoint drift")
    token_batch=collector.frozen_batch(texts)
    tok=real_tokenizer[0]
    alignments=[]
    for i,q in enumerate(texts):
        expected=token_batch["input_ids"][i,:int(token_batch["token_lengths"][i])].tolist()
        try:
            al=align_s50_raw_train36(tok,q,expected_ids=expected)
            if al.complete_prefix_position!=al.tokenizer_token_count-1:
                raise ValueError("S50 native prompt incomplete prefix")
            alignments.append(al)
        except ValueError:
            alignments.append(None)
    if len(alignments)!=36:
        raise ValueError("S50 real native 36 text alignment audit incomplete")
    # Only after original J7 and native CPU prefills: store ACTUAL hidden
    # features, lengths and subtoken offsets; NEVER export gold labels.
    out.mkdir(parents=True,exist_ok=True)
    feature_file=out/"NRA3_S50_REAL43M_GT_NEW_QUOTE_TRAIN36_HIDDEN320_FP16.safetensors"
    save_file(token_batch,str(feature_file))
    saved=load_file(str(feature_file),device="cpu")
    if (set(saved)!=set(token_batch)
            or any(not torch.equal(saved[k],v) for k,v in token_batch.items())):
        raise AssertionError("S50 actual raw36 FP16 CPU source corrupted")
    # Physical word/subtoken maps contain NO hidden features or answers.
    maps=[]
    supported=[]
    unsupported=[]
    for idx,(q,aligned) in enumerate(zip(texts,alignments)):
        if aligned is None:
            unsupported.append(idx)
            maps.append({'text_sha256':hashlib.sha256(q.encode('utf-8')).hexdigest(),
                         'native_bpe_word_alignment_supported':False})
            continue
        supported.append(idx)
        maps.append({
            'text_sha256':hashlib.sha256(q.encode('utf-8')).hexdigest(),
            'native_bpe_word_alignment_supported':True,
            'token_count':aligned.tokenizer_token_count,
            'word_last_token_positions':[w.last_token for w in aligned.word_alignment],
            'word_token_positions':[list(w.token_positions) for w in aligned.word_alignment],
        })
    if not supported:
        raise AssertionError('S50 no safely BPE-aligned original43M quote samples')
    max_words=max(len(raw_surface_grammar(texts[i])) for i in supported)
    raw338=torch.zeros((len(supported),max_words,338),dtype=torch.float32)
    valid=torch.zeros((len(supported),max_words),dtype=torch.bool)
    for out_row,source_id in enumerate(supported):
        words=raw_surface_grammar(texts[source_id])
        last_positions=maps[source_id]['word_last_token_positions']
        if len(last_positions)!=len(words):
            raise ValueError('S50 native BPE word offsets drift')
        for word_i,word in enumerate(words):
            raw338[out_row,word_i,:320]=token_batch['token_hidden320'][source_id,last_positions[word_i]].float()
            raw338[out_row,word_i,320:]=torch.tensor(word.features,dtype=torch.float32)
        valid[out_row,:len(words)]=True
    port=Frozen43MGreenTunnelQuotePort()
    before_native=frozen_native_manifest_digest(raw338,valid)
    original=port.forward_original_unchanged(raw338,valid,tuple(texts[i] for i in supported))
    augmented,quote_sidecar=port.shadow_augmented_features_only(
        raw338,valid,tuple(texts[i] for i in supported))
    if (original is not raw338 or original.data_ptr()!=raw338.data_ptr()
            or not torch.equal(augmented[...,:338],raw338)
            or before_native!=frozen_native_manifest_digest(raw338,valid)
            or port.output_gate!=0):
        raise AssertionError('S50 original43M GT vectors changed by S48 sidecar')
    quote_tokens=int(quote_sidecar.text_only_quote_mask.sum())
    if quote_tokens<1 or sum(quote_sidecar.per_row_matched_pairs)<1:
        raise AssertionError('S50 actual original43M+GT quote-bearing native vectors absent')
    report = {
        "stage":"NRA3_S50_GENUINE_ORIGINAL43M_GT_616J7_PLUS_S48_QUOTE_TRAIN36_NATIVE_LAYER8_CPU",
        "S50_NEW_S48_QUOTE_TRAIN36_SOURCE_SHA256":source_sha,
        "S48_TRAIN36_SOURCE_COMMIT":"95592557e95965b149ee9b7780ceba164f93a7ff",
        "S48_TRAIN36_SOURCE_SHA256":source_sha,
        "S48_TRAIN36_SOURCE_BLOB":"ed6a76001aa1b593ddf2e7877af9ac9cbb59c416",
        "S194_ORIGINAL_616_J7_BASELINE_PARENT_RUN_ID":43345267876,
        "S194_ORIGINAL_CHECKPOINT_SHA256":fp16_src,
        "S194_ORIGINAL_584_GENERATED_TOKEN_DIGEST":source_cert["protected_584_generation_digest"],
        "S194_ORIGINAL_32_NEW_GENERATED_TOKEN_DIGEST":source_cert["train32_generation_digest"],
        "all_616_native_CPU_generated_token_digests_unchanged":True,
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "original_new32_full_EOS":0,
        "actual_43M_CPU_12_transformer_layers":True,
        "observed_layer_index_zero_based":7,
        "GT_residuals_before_readport_in_same_active_S93_scope":True,
        "NRA3_output_gate":0,
        "NRA3_candidate_accepted":False,
        "activation_allowed":False,
        "quote36_TRAIN_rows":len(texts),
        "actual_genuine_43M_CPU_quote36_TRAIN_prefills":capture_count,
        "real_tokenizer_chat_offsets_verified":len(supported)>0,
        "true_real_Calibri_token_alignment_all_36":len(supported)==36,
        "real_BPE_aligned_quote_rows":len(supported),
        "real_BPE_unaligned_quote_row_ids":unsupported,
        "real_BPE_align_failure_diagnostics_without_input_repair":align_failed,
        "actual_original43M_word_vectors_post_GT_with_valid_quote_span":quote_tokens,
        "actual_original43M_quote_pair_completed_count":sum(quote_sidecar.per_row_matched_pairs),
        "original_native338_same_object_and_bytewise_unchanged":True,
        "S48_quote_sidecar_appended_only_in_separate_339D_workspace":True,
        "original_43M_616_J7_protected_baseline_replayed_before_S50_capture":True,
        "original_S38_and_S47_challenges_NOT_loaded":True,
        "S48_held26_NOT_loaded_or_scored":True,
        "no_optimizer_or_model_selection":True,
        "gold_event_roles_coreference_never_passed_to_original43M":True,
        "S48_TRAIN36_manual_quote_gold_NEVER_passed_to_model_or_adapter":True,
        "actual_fp16_feature_file":feature_file.name,
        "actual_fp16_feature_sha256":sha_file(feature_file),
        "actual_hidden320_shape":list(token_batch["token_hidden320"].shape),
        "actual_token_ids_lengths_sha256":hashlib.sha256(json.dumps(
            [list(collector.samples[hashlib.sha256(q.encode("utf-8")).hexdigest()].token_ids)
             for q in texts],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "true_train36_token_counts":token_batch["token_lengths"].tolist(),
        "wordpiece_alignment":maps,
        "no_training_original43M_GT_NRA_or_any_adapter":True,
        "new_candidate_active_616_J7_EOS_not_measured":True,
        "all_in_bpp_le07_NOT_CERTIFIED":True,
        "whole_CPU_ge90pct_CB59_NOT_CERTIFIED":True,
        "no_spent_EVAL32_EVAL128_S17c_S15c_S12c":True,
        "LIVE_Render_Supabase_APK_Jessica_unchanged":True,
    }
    (out/"NRA3_S50_GENUINE43M_GT_QUOTE36_NATIVE_CPU_J7_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8")
    print('S50_TRUE_ORIGINAL43M_GT_POST_J7_QUOTE_TRAIN36',json.dumps({k:v for k,v in report.items()
        if k not in ("wordpiece_alignment","true_train36_token_counts")},
        sort_keys=True,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
